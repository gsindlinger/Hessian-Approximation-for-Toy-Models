"""Training plot helpers use snapshots and do not launch numerical models."""

import json

import pytest

from experiments.toy_pairs.plot_training import TrainingRun, load_history, summarize


def test_history_ignores_only_trailing_partial_append(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 1}\n{"epoch": 2}\n{"epoch":')
    with pytest.warns(UserWarning, match="incomplete final"):
        assert [row["epoch"] for row in load_history(path)] == [1, 2]
    path.write_text('{"epoch": 1}\nnot-json\n{"epoch": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        load_history(path)


def test_history_rejects_repeated_epochs(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 1}\n{"epoch": 1}\n')
    with pytest.raises(ValueError, match="increasing epochs"):
        load_history(path)


def test_summary_reports_late_instability_not_only_final_accuracy(tmp_path):
    run = TrainingRun(
        directory=tmp_path,
        config={
            "architecture": "transformer",
            "metadata": {"dataset": {"name": "modular_addition"}, "model_seed": 42, "training_status": "running"},
            "training": {"epochs": 4, "learning_rate": 0.001, "weight_decay": 0.1, "lr_schedule": "cosine", "beta2": 0.999},
        },
        history=[
            {"epoch": 1, "train/loss": 4.0, "val/loss": 4.5, "val/accuracy": 0.01},
            {"epoch": 2, "train/loss": 0.1, "val/loss": 0.2, "val/accuracy": 0.99},
            {"epoch": 3, "train/loss": 2.9, "val/loss": 3.0, "val/accuracy": 0.15},
            {"epoch": 4, "train/loss": 0.01, "val/loss": 0.02, "val/accuracy": 1.0},
        ],
        selected=False,
    )
    summary = summarize(run, tail_epochs=3)
    assert summary["status"] == "completed"
    assert summary["tail_start_epoch"] == 2
    assert summary["latest_val_accuracy"] == 1.0
    assert summary["tail_val_accuracy_min"] == 0.15
    assert summary["tail_val_loss_max"] == 3.0
    assert summary["minimum_val_loss_epoch"] == 4
    assert summary["tail_grad_norm_max"] is None
    assert summary["all_train_val_losses_finite"]
    json.dumps(summary, allow_nan=False)
