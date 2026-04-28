"""Single source of truth for filesystem layout.

All scripts construct paths via this module. Paths are repo-relative and
resolved from PROJECT_ROOT, so absolute paths never leak into configs or
output files.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "experiments" / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "experiments" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RUNS_DIR = OUTPUTS_DIR / "runs"
RESULTS_DB = OUTPUTS_DIR / "runs.db"


# -- model artifacts ---------------------------------------------------------

def models_base_dir(dataset: str) -> Path:
    """outputs/models/<dataset>/  — parent dir for all models on this dataset."""
    return MODELS_DIR / dataset


def model_dir(dataset: str, model_id: str) -> Path:
    """outputs/models/<dataset>/<model_id>/"""
    return MODELS_DIR / dataset / model_id


def epoch_dir(dataset: str, model_id: str, epoch: int) -> Path:
    """outputs/models/<dataset>/<model_id>/epoch_<e>/"""
    return model_dir(dataset, model_id) / f"epoch_{epoch}"


def collector_dir(
    dataset: str,
    model_id: str,
    epoch: int,
    strategy: str,
    repetitions: int,
    corr: bool = False,
) -> Path:
    """outputs/models/<dataset>/<model_id>/epoch_<e>/collector_<strategy>_r<reps>[_corr]/

    The cache key is encoded in the dir name so different (strategy, reps)
    settings produce parallel caches instead of clobbering.
    """
    suffix = "_corr" if corr else ""
    return epoch_dir(dataset, model_id, epoch) / f"collector_{strategy}_r{repetitions}{suffix}"


# -- training-sweep bookkeeping ---------------------------------------------

def training_run_dir(experiment_name: str, timestamp: str) -> Path:
    """outputs/training/<experiment_name>/<timestamp>/  — one dir per training
    invocation. Holds full_results.json and best_models.yaml side by side.
    """
    return OUTPUTS_DIR / "training" / experiment_name / timestamp


# -- analysis-run artifacts -------------------------------------------------

def run_dir(run_id: str) -> Path:
    """outputs/runs/<run_id>/"""
    return RUNS_DIR / run_id


def influence_dir(run_id: str) -> Path:
    return run_dir(run_id) / "influence"


def influence_path(run_id: str, model_tag: str, method: str) -> Path:
    """Deterministic from (run_id, model_tag, method) — db rows store this
    triple, not raw paths, so reorganizing only touches this file.

    `model_tag` disambiguates when one run analyzes several (model, epoch)
    pairs (e.g. `mlp_fec568fb74dc_epoch_1000`).
    """
    return influence_dir(run_id) / f"{model_tag}_{method}.npy"
