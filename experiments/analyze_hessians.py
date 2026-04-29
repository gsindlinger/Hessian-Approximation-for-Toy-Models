"""Hessian analysis pipeline (stepwise rewrite of experiments/hessian_analysis.py).

Step 1: load models from a models YAML.
Step 2: load the on-disk train/test split that training cached, and collect
        activations/gradients (plus a second independent `_corr` run for MCMC
        eigenvalue correction).
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jax.random import PRNGKey, permutation

from tqdm.auto import tqdm

from experiments import paths, provenance, results as results_db
from experiments.utils import block_tree, json_safe, to_dataclass
from src.config import (
    ComputationType,
    DatasetEnum,
    HessianAnalysisConfig,
    HessianApproximationMethod,
    LossType,
    PseudoTargetGenerationStrategy,
    RegularizationStrategy,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import sample_vectors
from src.utils.data.data import (
    Dataset,
    load_split_from_disk,
    normalize_for_loss,
)
from src.utils.influence import compute_influence_matrix, compute_per_example_flat_grads
from src.utils.loss import get_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.train import load_model_checkpoint

logger = logging.getLogger(__name__)
logging.getLogger("src.hessians.collector").setLevel(logging.WARNING)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _read_saved_epochs(model_dir: str) -> Optional[List[int]]:
    """Read the `saved_epochs` list from model.json metadata.

    Returns None if the field is absent (e.g., model trained before this field
    was added) — callers should treat that as "trust the request, don't filter."
    """
    model_json = os.path.join(model_dir, "model.json")
    if not os.path.exists(model_json):
        return None
    with open(model_json, "r") as f:
        data = json.load(f)
    metadata = data.get("metadata") or {}
    saved = metadata.get("saved_epochs")
    return None if saved is None else [int(e) for e in saved]


def model_epoch_pairs(
    models: List[str], requested_epochs: Optional[List[int]]
) -> List[Tuple[str, Optional[int]]]:
    """Build (model_dir, epoch) pairs to analyze.

    `requested_epochs` is the analysis YAML's `epochs:` list. If empty/None,
    each model is analyzed at its final checkpoint only. Otherwise, requested
    epochs are intersected with each model's `saved_epochs` from model.json
    metadata; any miss is logged and skipped. Models without `saved_epochs`
    metadata fall back to honoring the request as-is.
    """
    if not requested_epochs:
        return [(m, None) for m in models]
    pairs: List[Tuple[str, Optional[int]]] = []
    for m in models:
        saved = _read_saved_epochs(m)
        for e in requested_epochs:
            if saved is None or e in saved:
                pairs.append((m, e))
            else:
                logger.warning(
                    "epoch %d not in saved_epochs=%s for model %s; skipping",
                    e,
                    saved,
                    m,
                )
    return pairs


def _read_model_dataset_info(model_dir: str) -> Optional[Dict]:
    """Read the dataset block written into model.json's metadata at training time."""
    model_json = os.path.join(model_dir, "model.json")
    if not os.path.exists(model_json):
        return None
    with open(model_json, "r") as f:
        data = json.load(f)
    metadata = data.get("metadata") or {}
    return metadata.get("dataset")


def resolve_shared_dataset(model_dirs: List[str]) -> Dict:
    """Read each model's saved dataset block; warn if any disagree, return the
    first one."""
    if not model_dirs:
        raise ValueError(
            "No model directories were provided for Hessian analysis. "
            "Check that the models YAML contains a non-empty 'models' list."
        )

    infos: List[Tuple[str, Optional[Dict]]] = [
        (m, _read_model_dataset_info(m)) for m in model_dirs
    ]
    missing = [m for m, info in infos if info is None]
    if missing:
        raise FileNotFoundError(
            "Models missing dataset metadata in model.json (was the model trained "
            f"with the updated train_models.py?): {missing}"
        )

    reference = infos[0][1]
    assert reference is not None
    ref_key = (reference["name"], reference["split_id"], reference["train_dir"])
    for model_dir, info in infos[1:]:
        assert info is not None
        key = (info["name"], info["split_id"], info["train_dir"])
        if key != ref_key:
            logger.error(
                "model %s was trained on a different dataset/split (%s) than the "
                "first model (%s); analysis will use the first model's split.",
                model_dir,
                key,
                ref_key,
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
    seed: int,
    cache_inputs_base: Dict[str, Any],
):
    """Run the primary collector; also run `_corr` when strategy is MCMC.

    `cache_inputs_base` is the dataset/seed-level provenance the collector
    stamps into its manifest and validates on reload. The `_corr` pass uses
    `seed + 1` and `role="corr"`, so primary and corr each pin their own
    inputs.
    """
    strategy = analysis_cfg.computation_config.pseudo_target_generation_strategy
    reps = analysis_cfg.computation_config.pseudo_target_generation_repetitions

    def _make():
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

    primary_inputs = {**cache_inputs_base, "rng_seed": int(seed), "role": "primary"}
    primary = _collect(collector_dir, PRNGKey(seed), primary_inputs)
    if strategy == PseudoTargetGenerationStrategy.MCMC:
        corr_inputs = {**cache_inputs_base, "rng_seed": int(seed + 1), "role": "corr"}
        corr = _collect(collector_dir_corr, PRNGKey(seed + 1), corr_inputs)
    else:
        corr = primary
    logger.info(
        "collected acts/grads → %s%s",
        collector_dir,
        " (+ _corr)" if strategy == PseudoTargetGenerationStrategy.MCMC else "",
    )

    return primary, corr


@dataclass
class HessianCtx:
    """Per-model state: collector data + model ctx + build dir + computer cache."""

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


def resolve_damping(
    ctx: HessianCtx, analysis_cfg: HessianAnalysisConfig
) -> Tuple[Optional[float], Optional[float]]:
    """Return (damping, pseudo_inverse_factor)."""
    comp = analysis_cfg.computation_config
    strat = comp.regularization_strategy
    if strat in (
        RegularizationStrategy.AUTO_MEAN_EIGENVALUE,
        RegularizationStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION,
    ):
        ekfac = ctx.get(HessianApproximationMethod.EKFAC)
        assert isinstance(ekfac, EKFACComputer)
        return ekfac.get_damping(
            damping_strategy=strat, factor=comp.regularization_value
        ), None
    if strat == RegularizationStrategy.FIXED:
        return comp.regularization_value, None
    if strat == RegularizationStrategy.PSEUDO_INVERSE:
        return None, comp.regularization_value
    return None, None


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
    damping: Optional[float],
    pseudo_inverse_factor: Optional[float],
    compute_approximation_error: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    ref = ctx.get(reference)
    ref_ihvp = block_tree(
        ref.estimate_ihvp(
            grads_1, damping=damping, pseudo_inverse_factor=pseudo_inverse_factor
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
                grads_1, damping=damping, pseudo_inverse_factor=pseudo_inverse_factor
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
    damping: Optional[float],
    pseudo_inverse_factor: Optional[float],
    run_id: str,
    model_tag: str,
) -> Dict[str, str]:
    """Per-method `(n_train,)` influence score (mean over query dim), saved
    as `.npy` under `outputs/runs/<run_id>/influence/`.
    """
    paths.influence_dir(run_id).mkdir(parents=True, exist_ok=True)
    influence_paths: Dict[str, str] = {}
    pbar = tqdm(methods, desc="influence")
    for approx in pbar:
        pbar.set_postfix_str(approx.value)
        matrix = compute_influence_matrix(
            test_flat_grads=test_flat_grads,
            train_flat_grads=train_flat_grads,
            computer=ctx.get(approx),
            damping=damping,
            pseudo_inverse_factor=pseudo_inverse_factor,
        )  # (n_test, n_train)
        vec = np.asarray(jnp.mean(matrix, axis=0))  # (n_train,)
        path = paths.influence_path(run_id, model_tag, approx.value)
        np.save(str(path), vec)
        influence_paths[approx.value] = str(path)

    return influence_paths


def _apply_overrides(
    overrides: List[str],
    models_cfg: Dict[str, Any],
    analysis_raw: Dict[str, Any],
) -> None:
    """Apply `path=value` overrides in-place to the loaded YAML dicts.

    Path conventions:
      `analysis.X.Y` → analysis_raw["hessian_analysis"]["X"]["Y"]
      `models.X.Y`   → models_cfg["X"]["Y"]

    Value is yaml.safe_load'd, so `0.05`, `null`, `[a,b,c]`, `true` parse as
    expected.
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be 'path=value', got {ov!r}")
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
                f"Override path must start with 'analysis.' or 'models.', got {ov!r}"
            )
        for p in tail[:-1]:
            target = target[p]
        target[tail[-1]] = value
        logger.info("override: %s = %r", key, value)


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

    seed: int = models_cfg["seed"]
    model_ids: List[str] = models_cfg["models"]
    epochs: Optional[List[int]] = (
        models_cfg.get("intermediate_epochs") or models_cfg.get("epochs")
    )

    dataset_cfg: DatasetConfig = to_dataclass(DatasetConfig, models_cfg["dataset"])  # type: ignore
    analysis_cfg: HessianAnalysisConfig = to_dataclass(
        HessianAnalysisConfig, analysis_raw["hessian_analysis"]
    )  # type: ignore
    dataset_name = dataset_cfg.name.value

    base_train, base_test = load_train_test(dataset_cfg, seed)
    logger.info(
        "dataset %s: N_train=%d, N_test=%d",
        dataset_name,
        base_train.inputs.shape[0],
        base_test.inputs.shape[0],
    )

    # Resolve `collector_subset_size: null` to the actual train size *now*, so
    # config_hash / cache_inputs / db rows always carry the concrete int.
    if analysis_cfg.computation_config.collector_subset_size is None:
        analysis_cfg.computation_config.collector_subset_size = int(
            base_train.inputs.shape[0]
        )

    all_results: List[Dict] = []
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    code_info = provenance.git_info()

    db = results_db.init_db()
    results_db.append_run(
        db,
        run_id=run_id,
        code_info=code_info,
        models_config_path=models_config_path,
        analysis_config_path=analysis_config_path,
        params={
            "dataset": asdict(dataset_cfg),
            "seed": seed,
            "model_ids": list(model_ids),
            "epochs": epochs,
            "hessian_config": asdict(analysis_cfg),
        },
    )

    comp_cfg = analysis_cfg.computation_config
    strategy_str = comp_cfg.pseudo_target_generation_strategy.value
    reps = comp_cfg.pseudo_target_generation_repetitions

    for model_id, epoch in model_epoch_pairs(model_ids, epochs):
        model_dir = str(paths.model_dir(dataset_name, model_id))
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
        result_config_hash = provenance.config_hash(
            dataset_cfg, analysis_cfg, seed, model_id, epoch
        )
        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "  %s  (%s)  num_params=%d",
            model_config.get_model_display_name(),
            tag,
            model.num_params,
        )
        logger.info("=" * 70)

        if (
            skip_if_exists
            and code_info.get("sha")
            and results_db.result_exists(
                db,
                config_hash=result_config_hash,
                code_sha=code_info["sha"],
            )
        ):
            logger.info(
                "  SKIP — config_hash=%s already computed on code_sha=%s",
                result_config_hash, code_info["sha"],
            )
            continue

        dataset, test_dataset = prepare_datasets_for_model(
            base_train, base_test, model_config.loss
        )

        # Subset is collector-only — probes, model_ctx, and influence still use
        # the full train. Deterministic permutation seeded by `seed` so identical
        # (size, seed) → identical subset. When subset == full size, skip the
        # permutation and use the dataset as-is.
        subset_size = comp_cfg.collector_subset_size  # already resolved to int
        n_full = int(dataset.inputs.shape[0])
        if subset_size > n_full:
            raise ValueError(
                f"collector_subset_size={subset_size} exceeds train size {n_full}"
            )
        if subset_size < n_full:
            perm = permutation(PRNGKey(seed), n_full)[:subset_size]
            dataset_for_collector = type(dataset)(
                dataset.inputs[perm], dataset.targets[perm]
            )
            logger.info(
                "collector subset: %d / %d (seed=%d)", subset_size, n_full, seed
            )
        else:
            dataset_for_collector = dataset

        loss_fn = get_loss(model_config.loss)
        collector_dir = str(
            paths.collector_dir(dataset_name, model_id, epoch, strategy_str, reps)
        )
        collector_dir_corr = str(
            paths.collector_dir(
                dataset_name, model_id, epoch, strategy_str, reps, corr=True
            )
        )
        cache_inputs_base = {
            "dataset_name": dataset_name,
            "dataset_test_size": dataset_cfg.test_size,
            "dataset_store_on_disk": dataset_cfg.store_on_disk,
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
            seed=seed,
            cache_inputs_base=cache_inputs_base,
        )

        probes = sample_vectors(
            vector_config=analysis_cfg.vector_config,
            model=model,
            params=params,
            inputs=test_ds.inputs,
            targets=test_ds.targets,
            loss_fn=loss_fn,
            seed=analysis_seed + 1,
            repetitions=1,
        )

        # Use train_ds as the canonical "dataset" for the rest of the pipeline.
        dataset = train_ds

        model_ctx = ModelContext.create(
            model=model,
            params=params,
            dataset=dataset,
            loss_fn=loss_fn,
        )

        build_base_dir = os.path.join(model_config.directory, "collector")  # type: ignore
        if epoch is not None:
            build_base_dir = os.path.join(build_base_dir, f"epoch_{epoch}")

        ctx = HessianCtx(
            collector_data=collector_data,
            collector_data_corr=collector_data_corr,
            model_ctx=model_ctx,
            build_base_dir=os.path.join(collector_dir, "factors"),
        )

        damping, pif = resolve_damping(ctx, analysis_cfg)
        if damping is not None:
            logger.info("damping=%.6f", damping)
        elif pif is not None:
            logger.info("pseudo_inverse_factor=%.6f", pif)

        matrix_comparisons: Dict[str, Dict[str, Dict[str, float]]] = {}
        hvp_comparisons: Dict[str, Dict[str, Dict[str, float]]] = {}
        ihvp_comparisons: Dict[str, Dict[str, Dict[str, float]]] = {}
        round_trip_errors: Dict[str, Dict[str, float]] = {}

        for reference in comp_cfg.comparison_references:
            logger.info("--- reference: %s ---", reference.value)
            if ComputationType.MATRIX in comp_cfg.computation_types:
                res = compare_matrices(
                    ctx,
                    reference,
                    comp_cfg.approximators,
                    analysis_cfg.matrix_config.metrics,
                )
                for metric, scores in res.items():
                    matrix_comparisons.setdefault(metric, {})[reference.value] = scores
            if ComputationType.HVP in comp_cfg.computation_types:
                res = compare_hvps(
                    ctx,
                    reference,
                    comp_cfg.approximators,
                    analysis_cfg.vector_config.metrics,
                    grads_1,
                    grads_2,
                )
                for metric, scores in res.items():
                    hvp_comparisons.setdefault(metric, {})[reference.value] = scores
            if ComputationType.IHVP in comp_cfg.computation_types:
                res, rt = compare_ihvps(
                    ctx,
                    reference,
                    comp_cfg.approximators,
                    analysis_cfg.vector_config.metrics,
                    grads_1,
                    grads_2,
                    damping,
                    pif,
                )
                for metric, scores in res.items():
                    ihvp_comparisons.setdefault(metric, {})[reference.value] = scores
                round_trip_errors[reference.value] = rt

        influence_section: Optional[Dict] = None
        if comp_cfg.compute_influence:
            train_flat_grads = compute_per_example_flat_grads(
                model, params, dataset.inputs, dataset.targets, loss_fn
            )
            test_flat_grads = compute_per_example_flat_grads(
                model, params, test_dataset.inputs, test_dataset.targets, loss_fn
            )
            all_methods = list(dict.fromkeys(
                list(comp_cfg.approximators) + list(comp_cfg.comparison_references)
            ))
            model_tag = f"{model_id}_{tag}"
            influence_paths = compute_influence_scores(
                ctx=ctx,
                methods=all_methods,
                train_flat_grads=train_flat_grads,
                test_flat_grads=test_flat_grads,
                damping=damping,
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
            "damping": damping,
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
        # Same (config_hash, code_sha) on rerun overwrites this row via
        # INSERT OR REPLACE (cascades to metrics/influence).
        result_id = results_db.append_result(
            db,
            run_id=run_id,
            model_id=model_id,
            epoch=epoch,
            config_hash=result_config_hash,
            model_hash=result_model_hash,
            code_sha=code_info.get("sha"),
            seed=int(seed),
            dataset_name=dataset_name,
            dataset_test_size=float(dataset_cfg.test_size),
            collector_subset_size=int(comp_cfg.collector_subset_size),
            regularization_value=float(comp_cfg.regularization_value),
            regularization_strategy=comp_cfg.regularization_strategy.value,
            pseudo_target_strategy=strategy_str,
            pseudo_target_repetitions=int(reps),
            vector_num_samples=int(analysis_cfg.vector_config.num_samples),
            sampling_method=analysis_cfg.vector_config.sampling_method.value,
            damping=damping,
            pseudo_inverse_factor=pif,
            num_parameters=int(model.num_params),
            val_loss=float((metadata or {}).get("val_loss")) if metadata and metadata.get("val_loss") is not None else None,
            val_accuracy=float((metadata or {}).get("val_accuracy")) if metadata and metadata.get("val_accuracy") is not None else None,
            params={
                "model_config": asdict(model_config),
                "model_directory": model_dir,
                "metadata": metadata or {},
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
                db, result_id=result_id, method_to_path=influence_section["paths"],
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
            f,
            indent=2,
            default=json_safe,
        )
    logger.info("wrote results → %s", output_file)
    db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/shared_models.yaml",
        help="Models YAML (`models` field with list of model directories).",
    )
    parser.add_argument(
        "--analysis-config",
        default="experiments/configs/hessian_analysis.yaml",
        help="Analysis YAML (HessianAnalysisConfig).",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip (model_id, epoch) pairs whose config_hash already exists "
             "in runs.db on the current code_sha (and code is clean).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable. Override a YAML field, e.g. "
             "--override analysis.computation_config.regularization_value=0.05 "
             "or --override models.seed=43. Value is YAML-parsed, so lists "
             "(e.g. [kfac,ekfac]) and null work.",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.analysis_config,
        skip_if_exists=args.skip_if_exists,
        overrides=args.override,
    )
