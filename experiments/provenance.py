"""Run/result provenance hashes.

Three identifiers, all short hex prefixes of full SHA-256:

* `code_hash` — git SHA + branch. Per-run; tells you which version of the
  source code produced a result. Only committed state is pinned — uncommitted
  edits and untracked files are invisible.
* `config_hash` — canonicalized hash of the inputs that determine numerical
  output for one `(model_id, epoch)`. Per-result; the dedupe key.
* `model_hash` — SHA of the checkpoint file. Per-result; pins the exact
  parameters that were loaded, independent of the model_id directory name.

The hashes are deliberately short (12 hex chars). Collision risk is negligible
at any plausible number of runs and short hashes are easier to eyeball.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from experiments.paths import PROJECT_ROOT

HASH_LEN = 12


def _short(h: str) -> str:
    return h[:HASH_LEN]


# -- code provenance --------------------------------------------------------

def git_info() -> Dict[str, Any]:
    """Current commit SHA + branch name. Empty dict if not a git checkout
    (so this never blocks running the pipeline).

    Note: only committed state is pinned. Uncommitted edits, staged changes,
    and untracked files are invisible. If you want stronger guarantees,
    commit before running.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    return {"sha": _short(sha), "branch": branch}


# -- config hash ------------------------------------------------------------

def _canonicalize(value: Any) -> Any:
    """Recursively turn dataclasses, enums, tuples, sets into hashable JSON.
    Lists are *sorted* if they look like enum-valued sets (approximators,
    metrics, etc.), so order in the YAML doesn't change the hash."""
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value  # Enum
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        items = [_canonicalize(v) for v in value]
        # If every element is a primitive, sort to make the hash order-insensitive.
        if all(isinstance(it, (str, int, float, bool)) for it in items):
            items = sorted(items)
        return items
    return value


def config_hash(
    dataset_cfg,
    analysis_cfg,
    seed: int,
    model_id: str,
    epoch: int | None,
) -> str:
    """Hash of the inputs that determine numerical output for one result.

    Excluded by design: anything cosmetic, anything that doesn't affect the
    numbers (output dirs, experiment_name, etc.).
    """
    inputs = {
        "dataset": {
            "name": dataset_cfg.name.value,
            "test_size": dataset_cfg.test_size,
            "store_on_disk": dataset_cfg.store_on_disk,
        },
        "seed": seed,
        "model_id": model_id,
        "epoch": epoch,
        "computation_config": _canonicalize(analysis_cfg.computation_config),
        "matrix_config": {
            "metrics": _canonicalize(analysis_cfg.matrix_config.metrics),
        },
        "vector_config": {
            "metrics": _canonicalize(analysis_cfg.vector_config.metrics),
            "num_samples": analysis_cfg.vector_config.num_samples,
            "sampling_method": _canonicalize(analysis_cfg.vector_config.sampling_method),
        },
    }
    blob = json.dumps(inputs, sort_keys=True, default=str).encode()
    return _short(hashlib.sha256(blob).hexdigest())


# -- model hash -------------------------------------------------------------

def model_hash(checkpoint_path: Path | str) -> str:
    """SHA of the checkpoint file bytes. Pins the exact parameters loaded,
    independent of the directory name (so two model_dirs with the same
    contents would hash identically — and a corrupted file would be caught)."""
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return _short(h.hexdigest())
