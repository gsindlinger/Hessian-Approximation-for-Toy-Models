"""Selection and plot windows must not silently mix in superseded runs."""

import json

import numpy as np
import pytest
import yaml

from experiments.toy_pairs.plot_selected_losses import PAIRS, SEEDS, load_selected_runs, visible_mask


def test_selection_uses_stable_manifests(tmp_path):
    for dataset, _, folder, _ in PAIRS:
        for seed in SEEDS:
            model = tmp_path / "models" / f"{dataset}_{seed}"
            model.mkdir(parents=True)
            config = {"metadata": {"dataset": {"name": dataset}, "model_seed": seed}, "training": {"epochs": 1}}
            (model / "model.json").write_text(json.dumps(config))
            (model / "training_metrics.jsonl").write_text(json.dumps({"epoch": 1, "train/loss": .1, "val/loss": .2}) + "\n")
            manifests = tmp_path / folder
            manifests.mkdir(exist_ok=True)
            (manifests / f"{dataset}_s{seed}_validation.yaml").write_text(yaml.safe_dump({"model_directory": str(model)}))
    # A stale Energy manifest must never be consulted.
    (tmp_path / "selected/energy_s42_validation.yaml").write_text("model_directory: /nonexistent/obsolete\n")
    runs = load_selected_runs(tmp_path)
    assert len(runs) == 12
    assert runs["energy", 42].directory.name == "energy_42"
    model = runs["energy", 42].directory
    config = json.loads((model / "model.json").read_text())
    config["metadata"]["model_seed"] = 99
    (model / "model.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Selection metadata mismatch"):
        load_selected_runs(tmp_path)


@pytest.mark.parametrize("dataset,end,window", [("mnist", 1000, 200), ("modular_addition", 2500, 500)])
def test_windows_preserve_unsmoothed_epoch_indices(dataset, end, window):
    epochs = np.arange(1, end + 1)
    assert visible_mask(epochs, "full", dataset).sum() == end
    assert np.array_equal(epochs[visible_mask(epochs, "early", dataset)], epochs[:window])
    assert np.array_equal(epochs[visible_mask(epochs, "tail", dataset)], epochs[-window:])
