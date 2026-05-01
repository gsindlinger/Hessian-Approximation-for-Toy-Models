"""Self-contained replacement for the broken `analyze_hessians.main()`.

Run via `experiments/new_run.sh` or directly:
    python -m experiments.analyze_hessians_v2 \\
        --config experiments/shared_models.yaml \\
        --analysis-config experiments/configs/hessian_analysis.yaml

Once verified, this file's contents can supersede `analyze_hessians.py`
entirely — no imports from the legacy module anywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import yaml
from jax.random import PRNGKey, permutation
from tqdm.auto import tqdm

from experiments import paths, provenance, results as results_db
from experiments.utils import block_tree, json_safe, to_dataclass
from src.config import (
    ComputationType,
    DampingStrategy,
    HessianAnalysisConfig,
    HessianApproximationMethod,
    HessianComputationConfig,
    LossType,
    ProbeSource,
    PseudoTargetGenerationStrategy,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator, KroneckerEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import sample_vectors
from src.utils.data.data import (
    Dataset,
    DatasetEnum,
    load_split_from_disk,
    normalize_for_loss,
)
from src.utils.influence import compute_influence_matrix, compute_per_example_flat_grads
from src.utils.loss import get_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.train import load_model_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model + dataset metadata helpers
# ---------------------------------------------------------------------------

def _find_model_dir(model_id: str) -> str:
    """Resolve a bare `model_id` to its on-disk directory by globbing
    `outputs/models/*/<model_id>/`. Errors if missing or ambiguous."""
    matches = list(paths.MODELS_DIR.glob(f"*/{model_id}"))
    if not matches:
        raise FileNotFoundError(
            f"model_id {model_id!r} not found under {paths.MODELS_DIR}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"ambiguous model_id {model_id!r}: matches in datasets "
            f"{[m.parent.name for m in matches]}"
        )
    return str(matches[0])


def _read_model_json(model_dir: str) -> Dict:
    with open(os.path.join(model_dir, "model.json"), "r") as f:
        return json.load(f)


def _read_model_dataset_info(model_dir: str) -> Optional[Dict]:
    """Read the dataset block written into `model.json["metadata"]["dataset"]`
    at training time. None if the field is absent."""
    return (_read_model_json(model_dir).get("metadata") or {}).get("dataset")


def _read_saved_epochs(model_dir: str) -> Optional[List[int]]:
    """Read `metadata.saved_epochs` from `model.json`. None if absent (e.g.,
    model trained before this field existed) — callers should treat None as
    'trust the request, don't filter'."""
    saved = (_read_model_json(model_dir).get("metadata") or {}).get("saved_epochs")
    return None if saved is None else [int(e) for e in saved]


def resolve_shared_dataset(model_dirs: List[str]) -> Dict:
    """Read each model's saved dataset block; require agreement across all
    listed models on (name, split_id, train_dir). Returns the first model's
    block."""
    if not model_dirs:
        raise ValueError("no model directories provided")
    infos: List[Tuple[str, Optional[Dict]]] = [
        (m, _read_model_dataset_info(m)) for m in model_dirs
    ]
    missing = [m for m, info in infos if info is None]
    if missing:
        raise FileNotFoundError(
            f"missing metadata.dataset in model.json for: {missing}"
        )
    reference = infos[0][1]
    assert reference is not None
    ref_key = (reference["name"], reference["split_id"], reference["train_dir"])
    for model_dir, info in infos[1:]:
        assert info is not None
        key = (info["name"], info["split_id"], info["train_dir"])
        if key != ref_key:
            raise ValueError(
                f"model {model_dir} trained on {key} disagrees with "
                f"first model's {ref_key}"
            )
    return reference


def load_train_test_for_loss(
    dataset_info: Dict, loss: LossType
) -> Tuple[Dataset, Dataset]:
    """Load the cached split from disk and apply per-loss normalization."""
    train, test = load_split_from_disk(
        DatasetEnum(dataset_info["name"]),
        Path(dataset_info["split_dir"]),
    )
    return normalize_for_loss(train, test, loss)


def model_epoch_pairs(
    model_dirs: List[str], requested_epochs: Optional[List[int]]
) -> List[Tuple[str, Optional[int]]]:
    """Build (model_dir, epoch) pairs to analyze.

    If `requested_epochs` is empty/None, each model is analyzed at its final
    checkpoint only. Otherwise, requested epochs are intersected with each
    model's saved_epochs metadata; misses are logged and skipped. Models
    without saved_epochs metadata fall back to honoring the request as-is.
    """
    if not requested_epochs:
        return [(m, None) for m in model_dirs]
    pairs: List[Tuple[str, Optional[int]]] = []
    for m in model_dirs:
        saved = _read_saved_epochs(m)
        for e in requested_epochs:
            if saved is None or e in saved:
                pairs.append((m, e))
            else:
                logger.warning(
                    "model %s has no checkpoint at epoch=%d (saved=%s); skipping",
                    m, e, saved,
                )
    return pairs


# ---------------------------------------------------------------------------
# Collector + Hessian context
# ---------------------------------------------------------------------------

def _has_cached_collection(save_dir: str) -> bool:
    return os.path.exists(
        os.path.join(save_dir, CollectorActivationsGradients.MANIFEST_FILENAME)
    )


def collect_activations_gradients(
    model,
    params,
    dataset: Dataset,
    loss_fn,
    analysis_cfg: HessianAnalysisConfig,
    collector_dir: str,
    collector_dir_corr: str,
    analysis_seed: int,
    cache_inputs_base: Dict[str, Any],
):
    """Run the primary collector; also run `_corr` when strategy is MCMC.

    `cache_inputs_base` is the per-run provenance the collector stamps into
    its manifest and validates on reload. The `_corr` pass uses
    `analysis_seed + 1` and `role="corr"`, so primary and corr each pin
    their own inputs.
    """
    strategy = analysis_cfg.computation_config.pseudo_target_generation_strategy
    reps = analysis_cfg.computation_config.pseudo_target_generation_repetitions

    def _make() -> CollectorActivationsGradients:
        return CollectorActivationsGradients(
            model=model,
            params=params,
            loss_fn=loss_fn,
            pseudo_target_repetitions=reps,
            pseudo_target_strategy=strategy,
        )

    def _collect(save_dir: str, rng_key, cache_inputs):
        return _make().collect(
            dataset=dataset,
            save_directory=save_dir,
            try_load=_has_cached_collection(save_dir),
            rng_key=rng_key,
            cache_inputs=cache_inputs,
        )

    primary_inputs = {
        **cache_inputs_base,
        "analysis_rng_seed": int(analysis_seed),
        "role": "primary",
    }
    primary = _collect(collector_dir, PRNGKey(analysis_seed), primary_inputs)
    if strategy == PseudoTargetGenerationStrategy.MCMC:
        corr_inputs = {
            **cache_inputs_base,
            "analysis_rng_seed": int(analysis_seed + 1),
            "role": "corr",
        }
        corr = _collect(
            collector_dir_corr, PRNGKey(analysis_seed + 1), corr_inputs
        )
    else:
        corr = primary
    logger.info(
        "  collected acts/grads → %s%s",
        collector_dir,
        " (+ _corr)" if strategy == PseudoTargetGenerationStrategy.MCMC else "",
    )
    return primary, corr


@dataclass
class HessianCtx:
    """Per-(model, epoch) state: collector data + model ctx + cached computers."""

    collector_data: DataActivationsGradients
    collector_data_corr: DataActivationsGradients
    model_ctx: ModelContext
    build_base_dir: str
    _cache: Dict[HessianApproximationMethod, HessianEstimator] = field(
        default_factory=dict
    )

    def get(self, approx: HessianApproximationMethod) -> HessianEstimator:
        if approx not in self._cache:
            data = HessianComputerRegistry.get_compute_context(
                approx, self.collector_data, self.model_ctx
            )
            comp = HessianComputerRegistry.get_computer(
                approx, data, corr_context=self.collector_data_corr
            )
            comp.build(base_directory=self.build_base_dir)
            self._cache[approx] = comp
        return self._cache[approx]


DampingValue = "Optional[float | Dict[str, float]]"


def _damping_values(
    comp_cfg: HessianComputationConfig,
) -> List[Optional[float]]:
    """Resolve `damping_value` (scalar | list | None) to the per-iteration list.

    A list expands to a sweep of scalars sharing the (model, epoch) precomputes;
    a scalar wraps to a single-element list; None is one iteration with no λ
    (strategy-dependent fallback inside `resolve_damping`).
    """
    v = comp_cfg.damping_value
    if v is None:
        return [None]
    if isinstance(v, list):
        if not v:
            raise ValueError(
                "damping_value: [] is not allowed; use null or a non-empty list"
            )
        return [float(x) for x in v]
    return [float(v)]


def _format_damping_for_tag(value: float) -> str:
    """`repr(float(...))` with the PyYAML-1.1 dot-in-mantissa fixup, so the
    string round-trips through both Python and yaml without ambiguity. Mirrors
    the normalizer in `run_damping_sweep.sh`."""
    s = repr(float(value))
    if "e" in s:
        head, _, tail = s.partition("e")
        if "." not in head:
            head += ".0"
        s = f"{head}e{tail}"
    return s


def _damping_tag(strategy: DampingStrategy, value: Optional[float]) -> str:
    """Filename-safe tag identifying a (strategy, λ) combination, e.g.
    `uniform_1.0e-06`, `auto_mean_0.1`, `pseudo_inverse_1.0`. `λ=None` means
    the strategy uses a derived value (uniform with EKFAC mean fallback) and
    is rendered as `<strategy>_auto`."""
    if value is None:
        return f"{strategy.value}_auto"
    return f"{strategy.value}_{_format_damping_for_tag(value)}"


def resolve_damping(
    ctx: HessianCtx,
    analysis_cfg: HessianAnalysisConfig,
    approximators: List[HessianApproximationMethod],
) -> Tuple[Dict[str, "Optional[float | Dict[str, float]]"], Optional[float]]:
    """Per-approximator damping table.

    Returns `(damping_table, pseudo_inverse_factor)` where `damping_table`
    maps each approximator's value-string to one of:
      - `None` (PSEUDO_INVERSE strategy — pif used instead)
      - a scalar (UNIFORM for everyone; AUTO* for non-Kronecker estimators)
      - a `{layer: λ}` dict (AUTO* for `KroneckerEstimator` subclasses)

    Today AUTO* on non-Kronecker estimators falls back to EKFAC's
    mean-aggregated scalar (preserves prior behavior). A future patch can
    derive estimator-specific dampings for the block_fim / block_hessian
    paths too.
    """
    comp = analysis_cfg.computation_config
    strat = comp.damping_strategy

    if strat == DampingStrategy.PSEUDO_INVERSE:
        return {a.value: None for a in approximators}, comp.damping_value
    if strat == DampingStrategy.UNIFORM:
        # `damping_value: null` → fall back to EKFAC's cross-layer mean as
        # the global scalar. Otherwise use the specified value as the
        # absolute λ for every approximator and every layer.
        if comp.damping_value is None:
            ekfac = ctx.get(HessianApproximationMethod.EKFAC)
            assert isinstance(ekfac, EKFACComputer)
            ekfac_per_layer = ekfac.get_damping(
                damping_strategy=DampingStrategy.AUTO_MEAN, factor=1.0
            )
            value = float(jnp.mean(jnp.stack(list(ekfac_per_layer.values()))))
        else:
            value = comp.damping_value
        return {a.value: value for a in approximators}, None
    if strat == DampingStrategy.AUTO_MEAN:
        ekfac = ctx.get(HessianApproximationMethod.EKFAC)
        assert isinstance(ekfac, EKFACComputer)
        ekfac_per_layer = ekfac.get_damping(
            damping_strategy=strat, factor=comp.damping_value
        )
        ekfac_scalar = float(
            jnp.mean(jnp.stack(list(ekfac_per_layer.values())))
        )
        out: Dict[str, "Optional[float | Dict[str, float]]"] = {}
        for approx in approximators:
            est = ctx.get(approx)
            if isinstance(est, KroneckerEstimator):
                out[approx.value] = est.get_damping(
                    damping_strategy=strat, factor=comp.damping_value
                )
            else:
                out[approx.value] = ekfac_scalar
        return out, None

    return {a.value: None for a in approximators}, None


# ---------------------------------------------------------------------------
# Comparison + influence
# ---------------------------------------------------------------------------

def compare_matrices(
    ctx: HessianCtx,
    reference: HessianApproximationMethod,
    approximators: List[HessianApproximationMethod],
    metrics: List[FullMatrixMetric],
) -> Dict[str, Dict[str, float]]:
    ref_H = block_tree(
        ctx.get(reference).estimate_hessian(), f"{reference.value}_matrix"
    )
    out: Dict[str, Dict[str, float]] = {m.value: {} for m in metrics}
    others = [a for a in approximators if a != reference]
    pbar = tqdm(others, desc=f"matrix vs {reference.value}")
    for approx in pbar:
        pbar.set_postfix_str(approx.value[:5].ljust(5))
        comp = ctx.get(approx)
        for metric in metrics:
            score = comp.compare_full_hessian_estimates(
                comparison_matrix=ref_H, metric=metric
            )
            out[metric.value][approx.value] = float(score)
    return out


def compare_hvps(
    ctx: HessianCtx,
    reference: HessianApproximationMethod,
    approximators: List[HessianApproximationMethod],
    metrics: List[VectorMetric],
    grads_1,
    grads_2,
) -> Dict[str, Dict[str, float]]:
    ref_hvp = block_tree(
        ctx.get(reference).estimate_hvp(grads_1), f"{reference.value}_hvp"
    )
    out: Dict[str, Dict[str, float]] = {m.name: {} for m in metrics}
    others = [a for a in approximators if a != reference]
    pbar = tqdm(others, desc=f"hvp vs {reference.value}")
    for approx in pbar:
        pbar.set_postfix_str(approx.value[:5].ljust(5))
        approx_hvp = block_tree(
            ctx.get(approx).estimate_hvp(grads_1), f"{approx.value}_hvp"
        )
        for metric in metrics:
            out[metric.name][approx.value] = float(
                metric.compute(ref_hvp, approx_hvp, grads_2)
            )
    return out


def compare_ihvps(
    ctx: HessianCtx,
    reference: HessianApproximationMethod,
    approximators: List[HessianApproximationMethod],
    metrics: List[VectorMetric],
    grads_1,
    grads_2,
    damping_table: Dict[str, "Optional[float | Dict[str, float]]"],
    pseudo_inverse_factor: Optional[float],
    compute_approximation_error: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    ref = ctx.get(reference)
    ref_ihvp = block_tree(
        ref.estimate_ihvp(
            grads_1,
            damping=damping_table[reference.value],
            pseudo_inverse_factor=pseudo_inverse_factor,
        ),
        f"{reference.value}_ihvp",
    )
    out: Dict[str, Dict[str, float]] = {m.name: {} for m in metrics}
    round_trip: Dict[str, float] = {}
    others = [a for a in approximators if a != reference]
    pbar = tqdm(others, desc=f"ihvp vs {reference.value}")
    for approx in pbar:
        pbar.set_postfix_str(approx.value[:5].ljust(5))
        approx_ihvp = block_tree(
            ctx.get(approx).estimate_ihvp(
                grads_1,
                damping=damping_table[approx.value],
                pseudo_inverse_factor=pseudo_inverse_factor,
            ),
            f"{approx.value}_ihvp",
        )
        for metric in metrics:
            out[metric.name][approx.value] = float(
                metric.compute(ref_ihvp, approx_ihvp, grads_2)
            )
        if compute_approximation_error:
            round_trip[approx.value] = float(
                VectorMetric.RELATIVE_ERROR.compute(
                    grads_1, ref.estimate_hvp(approx_ihvp), x=None, power=2.0
                )
            )
    return out, round_trip


def compute_influence_scores(
    ctx: HessianCtx,
    methods: List[HessianApproximationMethod],
    train_flat_grads,
    test_flat_grads,
    damping_table: Dict[str, "Optional[float | Dict[str, float]]"],
    pseudo_inverse_factor: Optional[float],
    run_id: str,
    model_tag: str,
) -> Dict[str, str]:
    """Per-method `(n_query, n_train)` influence score matrix, saved as `.npy`
    under `outputs/runs/<run_id>/influence/`. No aggregation over queries."""
    paths.influence_dir(run_id).mkdir(parents=True, exist_ok=True)
    influence_paths: Dict[str, str] = {}
    pbar = tqdm(methods, desc="influence")
    for approx in pbar:
        pbar.set_postfix_str(approx.value)
        matrix = compute_influence_matrix(
            test_flat_grads=test_flat_grads,
            train_flat_grads=train_flat_grads,
            computer=ctx.get(approx),
            damping=damping_table[approx.value],
            pseudo_inverse_factor=pseudo_inverse_factor,
        )  # (n_query, n_train)
        path = paths.influence_path(run_id, model_tag, approx.value)
        np.save(str(path), np.asarray(matrix))
        influence_paths[approx.value] = str(path)
    return influence_paths


# ---------------------------------------------------------------------------
# YAML / overrides
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_overrides(
    overrides: List[str],
    models_cfg: Dict[str, Any],
    analysis_raw: Dict[str, Any],
) -> None:
    """Apply `path=value` overrides in-place.

    Path conventions:
      `analysis.X.Y` → analysis_raw["hessian_analysis"]["X"]["Y"]
      `models.X.Y`   → models_cfg["X"]["Y"]
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be 'path=value', got {ov!r}")
        key, raw_value = ov.split("=", 1)
        value = yaml.safe_load(raw_value)
        parts = key.split(".")
        if parts[0] == "analysis":
            target = analysis_raw["hessian_analysis"]
            tail = parts[1:]
        elif parts[0] == "models":
            target = models_cfg
            tail = parts[1:]
        else:
            raise ValueError(
                f"override path must start with 'analysis.' or 'models.', got {ov!r}"
            )
        for p in tail[:-1]:
            target = target[p]
        target[tail[-1]] = value
        logger.info("override: %s = %r", key, value)


def _write_rerun_artifacts(
    run_dir: Path,
    models_cfg: Dict[str, Any],
    analysis_raw: Dict[str, Any],
) -> None:
    """Drop post-override configs + a runnable `rerun.sh` into the run dir.

    The script anchors paths to its own location and to a relative project
    root, so the entire `outputs/runs/<run_id>/` folder stays self-contained
    and portable across moves of the project tree.
    """
    models_path = run_dir / "models_config.yaml"
    analysis_path = run_dir / "analysis_config.yaml"
    with open(models_path, "w") as f:
        yaml.safe_dump(models_cfg, f, sort_keys=False)
    with open(analysis_path, "w") as f:
        yaml.safe_dump(analysis_raw, f, sort_keys=False)

    rerun = run_dir / "rerun.sh"
    rerun.write_text(
        '#!/bin/bash\n'
        'set -e\n'
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"\n'
        'cd "$PROJECT_ROOT"\n'
        'export TF_CPP_MIN_LOG_LEVEL=3\n'
        'python -m experiments.analyze_hessians \\\n'
        '    --config "$SCRIPT_DIR/models_config.yaml" \\\n'
        '    --analysis-config "$SCRIPT_DIR/analysis_config.yaml" \\\n'
        '    "$@"\n'
    )
    rerun.chmod(0o755)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(
    models_config_path: str,
    analysis_config_path: str,
    skip_if_exists: bool = False,
    overrides: Optional[List[str]] = None,
) -> None:
    models_cfg = load_yaml(models_config_path)
    analysis_raw = load_yaml(analysis_config_path)
    if overrides:
        _apply_overrides(overrides, models_cfg, analysis_raw)

    model_ids: List[str] = list(models_cfg["models"])
    model_dirs: List[str] = [_find_model_dir(m) for m in model_ids]

    analysis_cfg: HessianAnalysisConfig = to_dataclass(
        HessianAnalysisConfig, analysis_raw["hessian_analysis"]
    )  # type: ignore
    analysis_seed: int = analysis_cfg.analysis_seed
    epochs: Optional[List[int]] = analysis_raw.get("epochs")

    dataset_info = resolve_shared_dataset(model_dirs)
    dataset_name: str = dataset_info["name"]
    logger.info(
        "dataset %s (split_id=%s, train_dir=%s)",
        dataset_name, dataset_info["split_id"], dataset_info["train_dir"],
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    code_info = provenance.git_info()

    # Reproducibility: dump post-override configs + a runnable rerun.sh next
    # to the results, so `bash outputs/runs/<run_id>/rerun.sh` reruns this
    # exact analysis (including any overrides that were on the CLI).
    _write_rerun_artifacts(run_dir, models_cfg, analysis_raw)

    db = results_db.init_db()
    results_db.append_run(
        db,
        run_id=run_id,
        code_info=code_info,
        models_config_path=models_config_path,
        analysis_config_path=analysis_config_path,
        params={
            "dataset": dataset_info,
            "analysis_seed": analysis_seed,
            "model_ids": list(model_ids),
            "epochs": epochs,
            "hessian_config": asdict(analysis_cfg),
        },
    )

    comp_cfg = analysis_cfg.computation_config
    strategy_str = comp_cfg.pseudo_target_generation_strategy.value
    reps = comp_cfg.pseudo_target_generation_repetitions

    cached_splits: Dict[LossType, Tuple[Dataset, Dataset]] = {}
    all_results: List[Dict] = []

    for model_dir, epoch in model_epoch_pairs(model_dirs, epochs):
        model_id = os.path.basename(model_dir)
        params, model, model_config, metadata = load_model_checkpoint(
            model_dir, epoch=epoch
        )
        tag = f"epoch_{epoch}" if epoch is not None else "final"

        ckpt_path = (
            paths.epoch_dir(dataset_name, model_id, epoch) / "checkpoint.msgpack"
            if epoch is not None
            else paths.model_dir(dataset_name, model_id) / "checkpoint.msgpack"
        )
        result_model_hash = provenance.model_hash(ckpt_path)

        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "  %s  (%s)  num_params=%d",
            model_config.get_model_display_name(), tag, model.num_params,
        )
        logger.info("=" * 70)

        if model_config.loss not in cached_splits:
            cached_splits[model_config.loss] = load_train_test_for_loss(
                dataset_info, model_config.loss
            )
        train_ds, test_ds = cached_splits[model_config.loss]
        logger.info("  N_train=%d, N_test=%d", len(train_ds), len(test_ds))

        # Resolve `collector_subset_size: null` to actual train size *now*, so
        # config_hash / cache_inputs / db rows always carry the concrete int.
        if comp_cfg.collector_subset_size is None:
            comp_cfg.collector_subset_size = int(len(train_ds))

        # Subset is collector-only — probes, model_ctx, and influence still use
        # the full train. Deterministic permutation seeded by analysis_seed.
        subset_size = comp_cfg.collector_subset_size
        n_full = len(train_ds)
        if subset_size > n_full:
            raise ValueError(
                f"collector_subset_size={subset_size} exceeds train size {n_full}"
            )
        if subset_size < n_full:
            perm = permutation(PRNGKey(analysis_seed), n_full)[:subset_size]
            dataset_for_collector = type(train_ds)(
                train_ds.inputs[perm], train_ds.targets[perm]
            )
            logger.info(
                "  collector subset: %d / %d (analysis_seed=%d)",
                subset_size, n_full, analysis_seed,
            )
        else:
            dataset_for_collector = train_ds

        loss_fn = get_loss(model_config.loss)
        collector_dir = str(
            paths.collector_dir(
                dataset_name, model_id, epoch, strategy_str, reps, analysis_seed
            )
        )
        collector_dir_corr = str(
            paths.collector_dir(
                dataset_name, model_id, epoch, strategy_str, reps, analysis_seed,
                corr=True,
            )
        )
        cache_inputs_base = {
            "dataset_name": dataset_name,
            "split_id": dataset_info["split_id"],
            "collector_subset_size": subset_size,
            "model_id": model_id,
            "epoch": epoch,
            "strategy": strategy_str,
            "repetitions": int(reps),
        }

        collector_data, collector_data_corr = collect_activations_gradients(
            model=model,
            params=params,
            dataset=dataset_for_collector,
            loss_fn=loss_fn,
            analysis_cfg=analysis_cfg,
            collector_dir=collector_dir,
            collector_dir_corr=collector_dir_corr,
            analysis_seed=analysis_seed,
            cache_inputs_base=cache_inputs_base,
        )

        # Probes: grads_1 from train (the vector we ask the approximation to
        # apply H / H^-1 to). grads_2 source is config-driven via
        # `vector_config.comparison_probe_source` (default test). Cap
        # num_samples at min(train, comparison) so both sides match.
        comparison_ds = (
            train_ds
            if analysis_cfg.vector_config.comparison_probe_source == ProbeSource.TRAIN
            else test_ds
        )
        requested = analysis_cfg.vector_config.num_samples
        max_n = min(len(train_ds), len(comparison_ds))
        n_samples = min(requested, max_n)
        if requested > max_n:
            logger.warning(
                "vector_config.num_samples=%d exceeds available data "
                "(min(train=%d, comparison=%d)); clamping to %d",
                requested, len(train_ds), len(comparison_ds), max_n,
            )
        clamped_vec_cfg = replace(analysis_cfg.vector_config, num_samples=n_samples)

        grads_1 = sample_vectors(
            vector_config=clamped_vec_cfg,
            model=model,
            params=params,
            inputs=train_ds.inputs,
            targets=train_ds.targets,
            loss_fn=loss_fn,
            analysis_seed=analysis_seed,
            repetitions=1,
        )
        grads_2 = sample_vectors(
            vector_config=clamped_vec_cfg,
            model=model,
            params=params,
            inputs=comparison_ds.inputs,
            targets=comparison_ds.targets,
            loss_fn=loss_fn,
            analysis_seed=analysis_seed + 1,
            repetitions=1,
        )

        model_ctx = ModelContext.create(
            model=model, params=params, dataset=train_ds, loss_fn=loss_fn,
        )

        ctx = HessianCtx(
            collector_data=collector_data,
            collector_data_corr=collector_data_corr,
            model_ctx=model_ctx,
            build_base_dir=os.path.join(collector_dir, "factors"),
        )

        all_approx = list(dict.fromkeys(
            list(comp_cfg.approximators) + list(comp_cfg.comparison_references)
        ))

        # Train/test flat grads for influence are λ-independent; compute once
        # per (model, epoch) and reuse across the sweep.
        influence_grads: Optional[Tuple[Any, Any]] = None
        if comp_cfg.compute_influence:
            train_flat_grads = compute_per_example_flat_grads(
                model, params, train_ds.inputs, train_ds.targets, loss_fn,
            )
            test_flat_grads = compute_per_example_flat_grads(
                model, params, test_ds.inputs, test_ds.targets, loss_fn,
            )
            influence_grads = (train_flat_grads, test_flat_grads)

        damping_values = _damping_values(comp_cfg)
        if len(damping_values) > 1:
            logger.info(
                "  damping sweep: %d value(s) — %s",
                len(damping_values),
                ", ".join("auto" if v is None else f"{v:g}" for v in damping_values),
            )

        for lam in damping_values:
            comp_cfg_lam = replace(comp_cfg, damping_value=lam)
            analysis_cfg_lam = replace(
                analysis_cfg, computation_config=comp_cfg_lam
            )
            damping_tag = _damping_tag(comp_cfg.damping_strategy, lam)

            result_config_hash = provenance.config_hash(
                dataset_info, analysis_cfg_lam, analysis_seed, model_id, epoch,
            )

            logger.info("--- damping: %s ---", damping_tag)

            if (
                skip_if_exists
                and code_info.get("sha")
                and results_db.result_exists(
                    db, config_hash=result_config_hash, code_sha=code_info["sha"],
                )
            ):
                logger.info(
                    "  SKIP — config_hash=%s already computed on code_sha=%s",
                    result_config_hash, code_info["sha"],
                )
                continue

            damping_table, pif = resolve_damping(ctx, analysis_cfg_lam, all_approx)
            for k, v in damping_table.items():
                if v is None:
                    continue
                if isinstance(v, dict):
                    layer_strs = ", ".join(
                        f"{layer}={d:.4g}" for layer, d in v.items()
                    )
                    logger.info("  damping[%s] per-layer: %s", k, layer_strs)
                else:
                    logger.info("  damping[%s]=%.6f", k, v)
            if pif is not None:
                logger.info("  pseudo_inverse_factor=%.6f", pif)

            matrix_comparisons: Dict[str, Dict[str, Dict[str, float]]] = {}
            hvp_comparisons: Dict[str, Dict[str, Dict[str, float]]] = {}
            ihvp_comparisons: Dict[str, Dict[str, Dict[str, float]]] = {}
            round_trip_errors: Dict[str, Dict[str, float]] = {}

            for reference in comp_cfg.comparison_references:
                logger.info("    reference: %s", reference.value)
                if ComputationType.MATRIX in comp_cfg.computation_types:
                    res = compare_matrices(
                        ctx, reference, comp_cfg.approximators,
                        analysis_cfg.matrix_config.metrics,
                    )
                    for metric, scores in res.items():
                        matrix_comparisons.setdefault(metric, {})[reference.value] = scores
                if ComputationType.HVP in comp_cfg.computation_types:
                    res = compare_hvps(
                        ctx, reference, comp_cfg.approximators,
                        analysis_cfg.vector_config.metrics, grads_1, grads_2,
                    )
                    for metric, scores in res.items():
                        hvp_comparisons.setdefault(metric, {})[reference.value] = scores
                if ComputationType.IHVP in comp_cfg.computation_types:
                    res, rt = compare_ihvps(
                        ctx, reference, comp_cfg.approximators,
                        analysis_cfg.vector_config.metrics, grads_1, grads_2,
                        damping_table, pif,
                    )
                    for metric, scores in res.items():
                        ihvp_comparisons.setdefault(metric, {})[reference.value] = scores
                    round_trip_errors[reference.value] = rt

            influence_section: Optional[Dict] = None
            if influence_grads is not None:
                train_flat_grads, test_flat_grads = influence_grads
                model_tag = f"{model_id}_{tag}_{damping_tag}"
                influence_paths = compute_influence_scores(
                    ctx=ctx,
                    methods=all_approx,
                    train_flat_grads=train_flat_grads,
                    test_flat_grads=test_flat_grads,
                    damping_table=damping_table,
                    pseudo_inverse_factor=pif,
                    run_id=run_id,
                    model_tag=model_tag,
                )
                influence_section = {"paths": influence_paths}

            all_results.append({
                "model_id": model_id,
                "model_name": model_config.get_model_display_name(),
                "model_directory": model_dir,
                "epoch": epoch,
                "config_hash": result_config_hash,
                "model_hash": result_model_hash,
                "model_config": asdict(model_config),
                "num_parameters": model.num_params,
                "metadata": metadata or {},
                "damping_value": lam,
                "damping_strategy": comp_cfg.damping_strategy.value,
                "damping_table": damping_table,
                "pseudo_inverse_factor": pif,
                "hessian_analysis": {
                    "matrix_comparisons": matrix_comparisons,
                    "hvp_comparisons": hvp_comparisons,
                    "ihvp_comparisons": ihvp_comparisons,
                    "ihvp_round_trip_approximation_errors": round_trip_errors,
                    "influence": influence_section,
                },
            })

            # Persist this result incrementally — ctrl-C mid-sweep keeps what's done.
            # Same (config_hash, code_sha) on rerun overwrites this row.
            result_id = results_db.append_result(
                db,
                run_id=run_id,
                model_id=model_id,
                epoch=epoch,
                config_hash=result_config_hash,
                model_hash=result_model_hash,
                code_sha=code_info.get("sha"),
                analysis_seed=analysis_seed,
                dataset_name=dataset_name,
                dataset_test_size=(
                    float(dataset_info["test_size"])
                    if dataset_info.get("test_size") is not None else None
                ),
                collector_subset_size=int(comp_cfg.collector_subset_size),
                damping_value=(None if lam is None else float(lam)),
                damping_strategy=comp_cfg.damping_strategy.value,
                pseudo_target_strategy=strategy_str,
                pseudo_target_repetitions=int(reps),
                vector_num_samples=int(analysis_cfg.vector_config.num_samples),
                sampling_method=analysis_cfg.vector_config.sampling_method.value,
                comparison_probe_source=analysis_cfg.vector_config.comparison_probe_source.value,
                pseudo_inverse_factor=pif,
                num_parameters=int(model.num_params),
                val_loss=(
                    float((metadata or {}).get("val_loss"))
                    if metadata and metadata.get("val_loss") is not None else None
                ),
                val_accuracy=(
                    float((metadata or {}).get("val_accuracy"))
                    if metadata and metadata.get("val_accuracy") is not None else None
                ),
                params={
                    "model_config": asdict(model_config),
                    "model_directory": model_dir,
                    "metadata": metadata or {},
                    "damping_table": damping_table,
                },
            )
            results_db.append_metrics(
                db,
                result_id=result_id,
                rows=results_db.flatten_metric_dicts(
                    matrix_comparisons=matrix_comparisons,
                    hvp_comparisons=hvp_comparisons,
                    ihvp_comparisons=ihvp_comparisons,
                    round_trip_errors=round_trip_errors,
                ),
            )
            if influence_section is not None:
                results_db.append_influence(
                    db, result_id=result_id,
                    method_to_path=influence_section["paths"],
                )

    output_file = run_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "code": code_info,
                "models_config": models_config_path,
                "analysis_config": analysis_config_path,
                "hessian_config": asdict(analysis_cfg),
                "results": all_results,
            },
            f, indent=2, default=json_safe,
        )
    logger.info("wrote results → %s", output_file)
    db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="experiments/shared_models.yaml",
        help="Models YAML (`models` field with list of model_ids).",
    )
    parser.add_argument(
        "--analysis-config", default="experiments/configs/hessian_analysis.yaml",
        help="Analysis YAML (HessianAnalysisConfig).",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip (model_id, epoch) pairs whose config_hash already exists "
             "in runs.db on the current code_sha.",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Repeatable. Override a YAML field, e.g. "
             "--override analysis.computation_config.damping_value=0.05.",
    )
    args = parser.parse_args()
    main(
        args.config, args.analysis_config,
        skip_if_exists=args.skip_if_exists, overrides=args.override,
    )
