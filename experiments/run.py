"""Unified entry point for training, Hessian analysis, and LDS scoring.

Auto-detects available GPUs and assigns sequential per-GPU queues round-robin.
Each GPU has one worker, so an idle GPU cannot accidentally launch a second
job assigned to a different, still-busy GPU.

Usage:
    # Train models
    python -m experiments.run train config1.yaml config2.yaml

    # Hessian analysis (each YAML has both models + analysis config)
    python -m experiments.run analyze analysis1.yaml analysis2.yaml

    # Analysis + LDS
    python -m experiments.run full analysis1.yaml analysis2.yaml

    # Pin to specific GPUs
    python -m experiments.run train config1.yaml --gpus 0,2

    # Extra flags forwarded to analyze_hessians
    python -m experiments.run analyze analysis.yaml --skip-if-exists
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "experiments" / "logs" / "runs"

logger = logging.getLogger(__name__)


def detect_gpus() -> list[int]:
    """Return list of GPU ids visible to the system."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raw = os.environ["CUDA_VISIBLE_DEVICES"]
        if raw.strip():
            return [int(g) for g in raw.split(",")]

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return [int(line.strip()) for line in out.stdout.strip().splitlines()]
        except (subprocess.TimeoutExpired, ValueError):
            pass

    return [0]


def run_train(gpu: int, config_path: Path) -> subprocess.CompletedProcess:
    config_path = config_path.resolve()
    stamp = _log_stamp(gpu)
    log_dir = f"experiments/logs/training/{config_path.stem}/{stamp}"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    cmd = [
        sys.executable, "-m", "experiments.train_models",
        f"--config-name={config_path.stem}",
        f"--config-path={config_path.parent}",
        f"hydra.run.dir={log_dir}",
    ]
    log_file = _make_log_path(config_path, "train", gpu)
    logger.info(f"GPU {gpu} | training {config_path.name} → {log_file}")
    with open(log_file, "x") as handle:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        logger.error(f"GPU {gpu} | train FAILED (exit {result.returncode}), see {log_file}")
    return result


def _log_stamp(gpu: int) -> str:
    # Repeated configs can finish within one second; never reuse their logs.
    return f"{time.strftime('%Y%m%d-%H%M%S')}-gpu{gpu}-{os.getpid()}-{time.time_ns()}"


def _make_log_path(config: Path, stage: str, gpu: int) -> Path:
    stamp = _log_stamp(gpu)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{config.stem}_{stage}_{stamp}.log"


def run_analyze(
    gpu: int,
    config: Path,
    lds: bool = False,
    lds_config: Path | None = None,
    extra_flags: list[str] | None = None,
) -> subprocess.CompletedProcess:
    config = config.resolve()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [
        sys.executable, "-m", "experiments.analyze_hessians",
        "--config", str(config),
        "--analysis-config", str(config),
        *(extra_flags or []),
    ]
    log_file = _make_log_path(config, "analyze", gpu)
    logger.info(f"GPU {gpu} | analyzing {config.name} → {log_file}")
    with open(log_file, "x") as lf:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdout=lf, stderr=subprocess.STDOUT, text=True,
        )

    if result.returncode != 0:
        logger.error(f"GPU {gpu} | analyze FAILED (exit {result.returncode}), see {log_file}")
        return result

    if lds:
        stdout_text = log_file.read_text()
        for line in stdout_text.splitlines():
            if "wrote results →" in line:
                results_path = Path(line.split("→")[-1].strip())
                if not results_path.is_absolute():
                    results_path = PROJECT_ROOT / results_path
                if results_path.is_file():
                    lds_log = _make_log_path(config, "lds", gpu)
                    logger.info(f"GPU {gpu} | running LDS → {lds_log}")
                    lds_cfg = str(lds_config or "experiments/configs/lds.yaml")
                    lds_cmd = [
                        sys.executable, "-m", "experiments.lds_analysis",
                        "--config", lds_cfg,
                        "--results-json", str(results_path),
                    ]
                    with open(lds_log, "x") as lf:
                        return subprocess.run(
                            lds_cmd, cwd=PROJECT_ROOT, env=env,
                            stdout=lf, stderr=subprocess.STDOUT, text=True,
                        )
        logger.error(f"GPU {gpu} | requested LDS but could not find results.json in {log_file}")
        return subprocess.CompletedProcess(cmd, returncode=1)

    return result


@dataclass(frozen=True)
class JobResult:
    gpu: int
    config: Path
    returncode: int
    error: str | None = None


def _run_gpu_queue(
    gpu: int, configs: list[Path], command: str, run_lds: bool,
    lds_config: Path | None, extra_flags: list[str],
) -> list[JobResult]:
    """One worker owns one GPU and completes each subprocess before the next."""
    outcomes = []
    for config in configs:
        label = f"GPU {gpu} | {config}"
        try:
            if command == "train":
                completed = run_train(gpu, config)
            else:
                completed = run_analyze(gpu, config, run_lds, lds_config, extra_flags)
            outcome = JobResult(gpu, config, completed.returncode)
        except Exception as exc:
            # A bad configuration or failed child launch must not abandon
            # later jobs in this GPU's queue or any other GPU's queue.
            outcome = JobResult(gpu, config, 1, f"{type(exc).__name__}: {exc}")
        outcomes.append(outcome)
        if outcome.returncode == 0:
            logger.info("DONE  %s", label)
        else:
            logger.error("FAIL  %s (exit %d%s)", label, outcome.returncode,
                         f", {outcome.error}" if outcome.error else "")
    return outcomes


def run_jobs(
    command: str, configs: list[Path], gpu_list: list[int], *,
    run_lds: bool = False, lds_config: Path | None = None,
    extra_flags: list[str] | None = None,
) -> list[JobResult]:
    """Validate all jobs, then run independent, strictly serial GPU queues.

    No numerical libraries are imported here. Threads supervise child
    processes only; CUDA initialization happens in each assigned subprocess.
    Exclusivity is within this runner invocation, not a system-wide GPU lock.
    """
    if command not in ("train", "analyze", "full"):
        raise ValueError(f"Unknown runner command: {command}")
    if not gpu_list or len(set(gpu_list)) != len(gpu_list) or any(gpu < 0 for gpu in gpu_list):
        raise ValueError("GPU ids must be a nonempty list of unique nonnegative integers")
    resolved = [Path(config).resolve() for config in configs]
    if not resolved:
        raise ValueError("At least one configuration is required")
    for config in resolved:
        if not config.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config}")
    if lds_config is not None:
        lds_config = lds_config.resolve()
        if not lds_config.is_file():
            raise FileNotFoundError(f"LDS configuration file not found: {lds_config}")

    queues = {gpu: resolved[index::len(gpu_list)] for index, gpu in enumerate(gpu_list)}
    queues = {gpu: queue for gpu, queue in queues.items() if queue}
    logger.info("queued %d job(s) across GPUs: %s", len(resolved), list(queues))
    outcomes = []
    with ThreadPoolExecutor(max_workers=len(queues), thread_name_prefix="gpu-queue") as pool:
        futures = {
            pool.submit(_run_gpu_queue, gpu, queue, command, run_lds or command == "full",
                        lds_config, list(extra_flags or [])): (gpu, queue)
            for gpu, queue in queues.items()
        }
        for future in as_completed(futures):
            gpu, queue = futures[future]
            try:
                outcomes.extend(future.result())
            except Exception as exc:
                # Defensive fallback for a failure in queue supervision itself.
                logger.exception("GPU %d queue supervisor failed", gpu)
                outcomes.extend(JobResult(gpu, config, 1, str(exc)) for config in queue)
    return outcomes


def main():
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run",
        description="Unified runner for train / analyze / full pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train --
    p_train = sub.add_parser("train", help="Train models from YAML configs")
    p_train.add_argument("configs", nargs="+", help="Training YAML config files")
    p_train.add_argument("--gpus", default=None, help="Comma-separated GPU ids (default: auto-detect)")

    # -- analyze --
    p_analyze = sub.add_parser("analyze", help="Run Hessian analysis")
    p_analyze.add_argument("configs", nargs="+", help="Analysis YAML configs (must contain models + hessian_analysis)")
    p_analyze.add_argument("--gpus", default=None, help="Comma-separated GPU ids (default: auto-detect)")
    p_analyze.add_argument("--lds", action="store_true", help="Run LDS after analysis")
    p_analyze.add_argument("--lds-config", default=None, help="LDS config YAML")
    p_analyze.add_argument("--skip-if-exists", action="store_true")
    p_analyze.add_argument("--override", action="append", default=[])

    # -- full (= analyze + lds) --
    p_full = sub.add_parser("full", help="Analyze + LDS end-to-end")
    p_full.add_argument("configs", nargs="+", help="Analysis YAML configs (must contain models + hessian_analysis)")
    p_full.add_argument("--gpus", default=None, help="Comma-separated GPU ids (default: auto-detect)")
    p_full.add_argument("--lds-config", default=None, help="LDS config YAML")
    p_full.add_argument("--skip-if-exists", action="store_true")
    p_full.add_argument("--override", action="append", default=[])

    args = parser.parse_args()
    try:
        gpu_list = [int(g.strip()) for g in args.gpus.split(",")] if args.gpus else detect_gpus()
    except ValueError:
        parser.error("--gpus must contain comma-separated integer GPU ids")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [run] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(f"using GPUs: {gpu_list}")

    # -- ensure DB exists before parallel jobs race to create it --
    if args.command in ("analyze", "full"):
        from experiments.results import init_db
        init_db().close()

    # -- build extra flags for analyze_hessians --
    extra_flags = []
    if args.command in ("analyze", "full"):
        if args.skip_if_exists:
            extra_flags.append("--skip-if-exists")
        for ov in args.override:
            extra_flags.extend(["--override", ov])

    # -- dispatch: one sequential queue per GPU, never one shared job queue --
    try:
        outcomes = run_jobs(
            args.command, [Path(config) for config in args.configs], gpu_list,
            run_lds=args.command == "full" or getattr(args, "lds", False),
            lds_config=Path(args.lds_config) if getattr(args, "lds_config", None) else None,
            extra_flags=extra_flags,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    failed = sum(outcome.returncode != 0 for outcome in outcomes)

    if failed:
        sys.exit(f"{failed} job(s) failed")
    logger.info(f"all {len(outcomes)} job(s) completed successfully")


if __name__ == "__main__":
    main()
