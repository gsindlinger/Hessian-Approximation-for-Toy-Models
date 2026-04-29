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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from jax.random import PRNGKey
from tqdm.auto import tqdm

from experiments import paths, provenance
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
):
    """Run the primary collector; also run `_corr` when strategy is MCMC."""
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

    def _collect(save_dir: str, rng_key):
        return _make().collect(
            dataset=dataset,
            save_directory=save_dir,
            try_load=_has_cached_collection(save_dir),
            rng_key=rng_key,
        )

    primary = _collect(collector_dir, PRNGKey(seed))
    if strategy == PseudoTargetGenerationStrategy.MCMC:
        corr = _collect(collector_dir_corr, PRNGKey(seed + 1))
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


def main(models_config_path: str, analysis_config_path: str) -> None:
    models_cfg = load_yaml(models_config_path)
    analysis_raw = load_yaml(analysis_config_path)

    models: List[str] = models_cfg["models"]
    # Epoch filter comes from the analysis YAML; final-only if absent.
    epochs: Optional[List[int]] = analysis_raw.get("epochs")

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
    dataset_info = resolve_shared_dataset(models)
    logger.info(
        "dataset %s: split_id=%s, split_dir=%s",
        dataset_info["name"],
        dataset_info["split_id"],
        dataset_info["split_dir"],
    )

    all_results: List[Dict] = []
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Per-loss caching of normalized train/test, since loading + normalizing only
    # depends on the loss type.
    cached_splits: Dict[LossType, Tuple[Dataset, Dataset]] = {}

    for model_dir, epoch in model_epoch_pairs(models, epochs):
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

        if model_config.loss not in cached_splits:
            cached_splits[model_config.loss] = load_train_test_for_loss(
                dataset_info, model_config.loss
            )
        train_ds, test_ds = cached_splits[model_config.loss]

        loss_fn = get_loss(model_config.loss)
        collector_dir = resolve_collector_dir(model_dir, epoch)

        # Collector runs over the training set (Hessian/Fisher are estimated on
        # train data).
        analysis_seed = analysis_cfg.analysis_seed
        collector_data, collector_data_corr = collect_activations_gradients(
            model=model,
            params=params,
            dataset=train_ds,
            loss_fn=loss_fn,
            analysis_cfg=analysis_cfg,
            collector_dir=collector_dir,
            seed=analysis_seed,
        )

        # Probes: grads_1 from train, grads_2 from test. Cap num_samples at the
        # smaller of the two so both sides have matching counts.
        requested = analysis_cfg.vector_config.num_samples
        max_n = min(len(train_ds), len(test_ds))
        n_samples = min(requested, max_n)
        if requested > max_n:
            logger.warning(
                "vector_config.num_samples=%d exceeds available data "
                "(min(train=%d, test=%d)); clamping to %d",
                requested,
                len(train_ds),
                len(test_ds),
                max_n,
            )
        clamped_vec_cfg = replace(analysis_cfg.vector_config, num_samples=n_samples)

        grads_1 = sample_vectors(
            vector_config=clamped_vec_cfg,
            model=model,
            params=params,
            inputs=train_ds.inputs,
            targets=train_ds.targets,
            loss_fn=loss_fn,
            seed=analysis_seed,
            repetitions=1,
        )
        grads_2 = sample_vectors(
            vector_config=clamped_vec_cfg,
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

        all_results.append(
            {
                "model_name": model_config.get_model_display_name(),
                "model_directory": model_dir,
                "epoch": epoch,
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
                },
            }
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
    args = parser.parse_args()
    main(args.config, args.analysis_config)
