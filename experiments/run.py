"""Unified entry point for training, Hessian analysis, and LDS scoring.

Auto-detects available GPUs and assigns jobs round-robin across them.

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
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-gpu{gpu}"
    log_dir = f"experiments/logs/training/{config_path.stem}/{stamp}"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [
        sys.executable, "-m", "experiments.train_models",
        f"--config-name={config_path.stem}",
        f"--config-path={config_path.parent}",
        f"hydra.run.dir={log_dir}",
    ]
    logger.info(f"GPU {gpu} | training {config_path.name}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)


def _make_log_path(config: Path, stage: str, gpu: int) -> Path:
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-gpu{gpu}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{config.stem}_{stage}_{stamp}.log"


def run_analyze(
    gpu: int,
    config: Path,
    lds: bool = False,
    lds_config: Path | None = None,
    extra_flags: list[str] | None = None,
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    }
    cmd = [
        sys.executable, "-m", "experiments.analyze_hessians",
        "--config", str(config),
        "--analysis-config", str(config),
        *(extra_flags or []),
    ]
    log_file = _make_log_path(config, "analyze", gpu)
    logger.info(f"GPU {gpu} | analyzing {config.name} → {log_file}")
    with open(log_file, "w") as lf:
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
                results_json = line.split("→")[-1].strip()
                if Path(results_json).is_file():
                    lds_log = _make_log_path(config, "lds", gpu)
                    logger.info(f"GPU {gpu} | running LDS → {lds_log}")
                    lds_cfg = str(lds_config or "experiments/configs/lds.yaml")
                    lds_cmd = [
                        sys.executable, "-m", "experiments.lds_analysis",
                        "--config", lds_cfg,
                        "--results-json", results_json,
                    ]
                    with open(lds_log, "w") as lf:
                        return subprocess.run(
                            lds_cmd, cwd=PROJECT_ROOT, env=env,
                            stdout=lf, stderr=subprocess.STDOUT, text=True,
                        )
        logger.warning(f"GPU {gpu} | could not find results.json, skipping LDS")

    return result


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
    if args.gpus:
        gpu_list = [int(g) for g in args.gpus.split(",")]
    else:
        gpu_list = detect_gpus()
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

    # -- dispatch --
    from concurrent.futures import ProcessPoolExecutor, as_completed

    futures = {}
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pool:
        for i, cfg in enumerate(args.configs):
            gpu = gpu_list[i % len(gpu_list)]
            path = Path(cfg)

            if not path.is_file():
                sys.exit(f"error: file not found: {cfg}")

            if args.command == "train":
                fut = pool.submit(run_train, gpu, path)
            else:
                run_lds = args.command == "full" or getattr(args, "lds", False)
                lds_cfg = Path(args.lds_config) if args.lds_config else None
                fut = pool.submit(run_analyze, gpu, path, run_lds, lds_cfg, extra_flags)

            futures[fut] = f"GPU {gpu} | {cfg}"
            if i < len(args.configs) - 1:
                time.sleep(10)

    logger.info(f"launched {len(futures)} job(s) across GPUs: {gpu_list}")

    failed = 0
    for fut in as_completed(futures):
        label = futures[fut]
        try:
            result = fut.result()
            if result.returncode == 0:
                logger.info(f"DONE  {label}")
            else:
                logger.error(f"FAIL  {label} (exit {result.returncode})")
                failed += 1
        except Exception as e:
            logger.error(f"FAIL  {label} ({e})")
            failed += 1

    if failed:
        sys.exit(f"{failed} job(s) failed")
    logger.info(f"all {len(futures)} job(s) completed successfully")


if __name__ == "__main__":
    main()
