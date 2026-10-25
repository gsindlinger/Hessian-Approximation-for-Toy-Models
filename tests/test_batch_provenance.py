"""Provenance snapshots use only temporary files; no model jobs or devices."""

import hashlib
import json
import pytest
import yaml

from experiments.toy_pairs import record_batch as recorder


def _write(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def _fixture(tmp_path, monkeypatch):
    _write(tmp_path, "src/model.py", "# current uncommitted source\n")
    _write(tmp_path, "src/secrets.py", "EXCLUDED\n")
    _write(tmp_path, "src/.env", "EXCLUDED\n")
    _write(tmp_path, "experiments/driver.py", "# driver\n")
    _write(tmp_path, recorder.CAMPAIGN / "record_batch.py", "# recorder\n")
    _write(tmp_path, recorder.CAMPAIGN / "PROTOCOL.rst", "Protocol\n")
    _write(tmp_path, recorder.CAMPAIGN / "RUNS.rst", "Current campaign decisions\n")
    _write(tmp_path, recorder.CAMPAIGN / "notebooks/hidden.py", "EXCLUDED\n")
    _write(tmp_path, recorder.CAMPAIGN / "artifacts/provenance/old/hidden.py", "EXCLUDED\n")
    _write(tmp_path, recorder.REFERENCE_ANALYSIS, "models: [reference]\nhessian_analysis: {}\n")
    _write(tmp_path, recorder.NATIVE_LDS, "num_subsets: 100\nreps_per_model: 50\n")
    monkeypatch.setattr(recorder, "git_identity", lambda root: {"head": "abc", "dirty": True})
    monkeypatch.setattr(recorder, "resolve_native_recipe", lambda raw, kind: {**raw, "mock_default": kind})
    return tmp_path


def _selection(root, pair, seed, directory=None, model_id=None):
    directory = directory or recorder.SELECTION_DIRECTORIES[pair]
    model_id = model_id or f"model_{pair}_{seed}"
    base = recorder.CAMPAIGN / directory
    _write(root, base / f"{pair}_s{seed}_validation.yaml", yaml.safe_dump({
        "model_seed": seed, "model_directory": f"/unused/models/{model_id}",
        "test_used_for_selection": False, "passes_basic_training_gate": True,
    }))
    _write(root, base / f"{pair}_s{seed}_analysis_final.yaml", yaml.safe_dump({
        "models": [model_id], "hessian_analysis": {}, "epochs": [1000],
    }))


def test_partial_scope_source_hashes_native_recipes_and_superseded_exclusion(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    _selection(root, "concrete", 42)
    _selection(root, "energy", 42)
    _selection(root, "energy", 42, directory="selected", model_id="SUPERSEDED")
    with pytest.raises(ValueError, match="use --allow-partial"):
        recorder.record_batch(root)
    assert not list((root / recorder.CAMPAIGN / "artifacts/provenance").glob("*/manifest.json"))
    destination = recorder.record_batch(root, allow_partial=True)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["complete_selection_records"] is False
    assert len(manifest["pending_validation_records"]) == 10
    assert {row["model_id"] for row in manifest["selected_models"]} == {"model_concrete_42", "model_energy_42"}
    assert manifest["git"] == {"head": "abc", "dirty": True}
    assert manifest["files"]["src/model.py"]["sha256"] == hashlib.sha256(b"# current uncommitted source\n").hexdigest()
    assert not any("secret" in path or "hidden.py" in path or ".env" in path for path in manifest["files"])
    lds = manifest["recipes"][str(recorder.NATIVE_LDS)]
    assert lds["resolved"] == {"num_subsets": 100, "reps_per_model": 50, "mock_default": "lds"}
    assert lds["raw_sha256"] == hashlib.sha256(lds["raw_text"].encode()).hexdigest()
    assert lds["resolved_json_sha256"] == hashlib.sha256(recorder._json(lds["resolved"]).encode()).hexdigest()
    superseded = str(recorder.CAMPAIGN / "selected/energy_s42_validation.yaml")
    assert manifest["recipes"][superseded]["superseded_selection"] is True
    assert "does not establish" in manifest["snapshot_notice"]


def test_complete_snapshot_is_unique_and_preserves_previous_artifacts(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    for pair in recorder.SELECTION_DIRECTORIES:
        for seed in (42, 43, 44):
            _selection(root, pair, seed)
    first = recorder.record_batch(root)
    original = (first / "manifest.json").read_bytes()
    second = recorder.record_batch(root)
    assert first != second
    assert (first / "manifest.json").read_bytes() == original
    manifest = json.loads(original)
    assert manifest["complete_selection_records"] is True
    assert len(manifest["selected_models"]) == 12
    assert manifest["pending_validation_records"] == []
    assert manifest["file_manifest_sha256"] == json.loads((second / "manifest.json").read_text())["file_manifest_sha256"]


def test_native_resolution_is_default_complete_cpu_only(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    lds = recorder.resolve_native_recipe({"reps_per_model": 50}, "lds")
    assert lds["reps_per_model"] == 50
    assert lds["num_subsets"] == 100
    assert lds["filter"] == {"model_ids": None, "epochs": None, "methods": None}
    analysis = recorder.resolve_native_recipe({"models": ["mock"], "hessian_analysis": {}}, "analysis")
    assert analysis["models"] == ["mock"]
    assert analysis["hessian_analysis"]["analysis_seed"] == 42
    assert analysis["hessian_analysis"]["computation_config"]["parameter_dtype"] is None
