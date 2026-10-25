"""Runner scheduling/logging tests; every child process is mocked."""

import logging
import subprocess
import threading
from pathlib import Path

import pytest

from experiments import run as runner


def _configs(tmp_path, count=4):
    paths = []
    for index in range(count):
        path = tmp_path / f"job{index}.yaml"
        path.write_text("models: []\n")
        paths.append(path)
    return paths


def test_one_sequential_queue_per_gpu_even_when_other_gpu_finishes(tmp_path, monkeypatch):
    configs = _configs(tmp_path)
    gpu_zero_started = threading.Event()
    gpu_one_second_job_started = threading.Event()
    active = {0: 0, 1: 0}
    seen = {0: [], 1: []}
    lock = threading.Lock()

    def fake_train(gpu, config):
        with lock:
            active[gpu] += 1
            assert active[gpu] == 1, "Two jobs overlapped on the same GPU"
            seen[gpu].append(config.stem)
        if config.stem == "job0":
            gpu_zero_started.set()
            assert gpu_one_second_job_started.wait(timeout=5)
        elif config.stem == "job1":
            assert gpu_zero_started.wait(timeout=5)
        elif config.stem == "job3":
            assert active[0] == 1  # GPU 0 is still running its first job.
            gpu_one_second_job_started.set()
        with lock:
            active[gpu] -= 1
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(runner, "run_train", fake_train)
    outcomes = runner.run_jobs("train", configs, [0, 1])
    assert all(outcome.returncode == 0 for outcome in outcomes)
    assert seen == {0: ["job0", "job2"], 1: ["job1", "job3"]}


def test_failed_job_does_not_abandon_either_gpu_queue(tmp_path, monkeypatch):
    configs = _configs(tmp_path)
    seen = []

    def fake_train(gpu, config):
        seen.append(config.stem)
        if config.stem == "job0":
            raise OSError("mock child could not start")
        return subprocess.CompletedProcess([], 7 if config.stem == "job1" else 0)

    monkeypatch.setattr(runner, "run_train", fake_train)
    outcomes = runner.run_jobs("train", configs, [0, 1])
    assert set(seen) == {"job0", "job1", "job2", "job3"}
    assert sum(outcome.returncode != 0 for outcome in outcomes) == 2
    assert next(outcome for outcome in outcomes if outcome.config.stem == "job0").error.startswith("OSError:")


@pytest.mark.parametrize("gpus", [[], [0, 0], [-1]])
def test_invalid_gpu_list_refused_before_launch(tmp_path, monkeypatch, gpus):
    configs = _configs(tmp_path, 1)
    monkeypatch.setattr(runner, "run_train", lambda *args: pytest.fail("No child should launch"))
    with pytest.raises(ValueError, match="unique nonnegative"):
        runner.run_jobs("train", configs, gpus)


def test_all_configs_validated_before_any_launch(tmp_path, monkeypatch):
    configs = _configs(tmp_path, 1) + [tmp_path / "missing.yaml"]
    monkeypatch.setattr(runner, "run_train", lambda *args: pytest.fail("No child should launch"))
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        runner.run_jobs("train", configs, [0, 1])


def test_training_captures_child_stdout_stderr_and_announces_unique_log(tmp_path, monkeypatch, caplog):
    config = _configs(tmp_path, 1)[0]
    monkeypatch.setattr(runner, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    calls = []

    def fake_subprocess(cmd, **kwargs):
        calls.append((cmd, kwargs))
        kwargs["stdout"].write("mock progress and training output\n")
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess)
    with caplog.at_level(logging.INFO):
        runner.run_train(3, config)
        runner.run_train(3, config)
    logs = sorted((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 2
    assert all(path.read_text() == "mock progress and training output\n" for path in logs)
    assert all(str(path) in caplog.text for path in logs)
    assert all("experiments.train_models" in cmd for cmd, _ in calls)
    assert all(any(arg.startswith("hydra.run.dir=") for arg in cmd) for cmd, _ in calls)


def test_full_queue_preserves_analysis_and_lds_flags(tmp_path, monkeypatch):
    config, lds_config = _configs(tmp_path, 2)
    calls = []

    def fake_analyze(gpu, path, lds, lds_cfg, flags):
        calls.append((gpu, path, lds, lds_cfg, flags))
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(runner, "run_analyze", fake_analyze)
    flags = ["--skip-if-exists", "--override", "analysis_seed=12"]
    outcomes = runner.run_jobs("full", [config], [6], lds_config=lds_config, extra_flags=flags)
    assert outcomes[0].returncode == 0
    assert calls == [(6, config.resolve(), True, lds_config.resolve(), flags)]


def test_analysis_lds_handoff_keeps_gpu_and_propagates_lds_failure(tmp_path, monkeypatch):
    config, lds_config = _configs(tmp_path, 2)
    results = tmp_path / "results.json"
    results.write_text("{}")
    monkeypatch.setattr(runner, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    calls = []

    def fake_subprocess(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "6"
        if "experiments.analyze_hessians" in cmd:
            kwargs["stdout"].write("wrote results → results.json\n")
            return subprocess.CompletedProcess(cmd, 0)
        kwargs["stdout"].write("mock LDS failure\n")
        return subprocess.CompletedProcess(cmd, 3)

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess)
    result = runner.run_analyze(6, config, lds=True, lds_config=lds_config,
                                extra_flags=["--skip-if-exists"])
    assert result.returncode == 3
    assert len(calls) == 2
    assert "--skip-if-exists" in calls[0][0]
    assert calls[1][0][-1] == str(results)
    assert str(lds_config) in calls[1][0]


def test_requested_lds_without_result_marker_is_failure(tmp_path, monkeypatch):
    config = _configs(tmp_path, 1)[0]
    monkeypatch.setattr(runner, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0))
    assert runner.run_analyze(0, config, lds=True).returncode != 0


def test_cli_returns_nonzero_when_any_queued_job_failed(tmp_path, monkeypatch):
    config = _configs(tmp_path, 1)[0]
    monkeypatch.setattr(runner.sys, "argv", ["run", "train", str(config), "--gpus", "2"])
    monkeypatch.setattr(runner, "run_jobs", lambda *args, **kwargs: [runner.JobResult(2, config, 7)])
    with pytest.raises(SystemExit, match="1 job\\(s\\) failed") as error:
        runner.main()
    assert error.value.code != 0
