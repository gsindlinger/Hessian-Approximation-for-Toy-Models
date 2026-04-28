"""Hessian analysis pipeline (stepwise rewrite of experiments/hessian_analysis.py).

Step 1: load models from a models YAML.
Step 2: load dataset + collect activations/gradients (plus a second
        independent `_corr` run for MCMC eigenvalue correction).
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import yaml
from jax.random import PRNGKey

from tqdm.auto import tqdm

from experiments import paths, provenance
from experiments.utils import block_tree, json_safe, to_dataclass
from src.config import (
    ComputationType,
    DatasetConfig,
    HessianAnalysisConfig,
    HessianApproximationMethod,
    LossType,
    PseudoTargetGenerationStrategy,
    RegularizationStrategy,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import sample_vectors
from src.utils.data.data import Dataset, DownloadableDataset
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


def model_epoch_pairs(
    models: List[str], epochs: Optional[List[int]]
) -> List[Tuple[str, Optional[int]]]:
    if not epochs:
        return [(m, None) for m in models]
    return [(m, e) for m in models for e in epochs]


def load_train_test(
    dataset_cfg: DatasetConfig, seed: int
) -> Tuple[Dataset, Dataset]:
    """Reproduce the model's train/test split from (seed, test_size)."""
    full = DownloadableDataset.load(
        dataset=dataset_cfg.name,
        directory=dataset_cfg.path,
        store_on_disk=dataset_cfg.store_on_disk,
    )
    return full.train_test_split(test_size=dataset_cfg.test_size, seed=seed)


def prepare_datasets_for_model(
    train: Dataset, test: Dataset, loss: LossType
) -> Tuple[Dataset, Dataset]:
    """Fresh (train, test) pair, MSE-normalized with a shared scaler."""
    if loss != LossType.MSE:
        return Dataset(train.inputs, train.targets), Dataset(test.inputs, test.targets)
    train_inputs, test_inputs = Dataset.normalize_data(train.inputs, test.inputs)
    train_targets, test_targets = Dataset.normalize_data(train.targets, test.targets)
    return (
        Dataset(train_inputs, train_targets),
        Dataset(test_inputs, test_targets),
    )


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
    logger.info("collected acts/grads → %s%s", collector_dir,
                " (+ _corr)" if strategy == PseudoTargetGenerationStrategy.MCMC else "")

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
        return ekfac.get_damping(damping_strategy=strat, factor=comp.regularization_value), None
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
    ref_H = block_tree(ctx.get(reference).estimate_hessian(), f"{reference.value}_matrix")
    out: Dict[str, Dict[str, float]] = {m.value: {} for m in metrics}
    others = [a for a in approximators if a != reference]
    pbar = tqdm(others, desc=f"matrix vs {reference.value}")
    for approx in pbar:
        pbar.set_postfix_str(approx.value[:5].ljust(5))
        comp = ctx.get(approx)
        for metric in metrics:
            score = comp.compare_full_hessian_estimates(comparison_matrix=ref_H, metric=metric)
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
    ref_hvp = block_tree(ctx.get(reference).estimate_hvp(grads_1), f"{reference.value}_hvp")
    out: Dict[str, Dict[str, float]] = {m.name: {} for m in metrics}
    others = [a for a in approximators if a != reference]
    pbar = tqdm(others, desc=f"hvp vs {reference.value}")
    for approx in pbar:
        pbar.set_postfix_str(approx.value[:5].ljust(5))
        approx_hvp = block_tree(ctx.get(approx).estimate_hvp(grads_1), f"{approx.value}_hvp")
        for metric in metrics:
            out[metric.name][approx.value] = float(metric.compute(ref_hvp, approx_hvp, grads_2))
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
        ref.estimate_ihvp(grads_1, damping=damping, pseudo_inverse_factor=pseudo_inverse_factor),
        f"{reference.value}_ihvp",
    )
    out: Dict[str, Dict[str, float]] = {m.name: {} for m in metrics}
    round_trip: Dict[str, float] = {}
    others = [a for a in approximators if a != reference]
    pbar = tqdm(others, desc=f"ihvp vs {reference.value}")
    for approx in pbar:
        pbar.set_postfix_str(approx.value[:5].ljust(5))
        approx_ihvp = block_tree(
            ctx.get(approx).estimate_ihvp(grads_1, damping=damping, pseudo_inverse_factor=pseudo_inverse_factor),
            f"{approx.value}_ihvp",
        )
        for metric in metrics:
            out[metric.name][approx.value] = float(metric.compute(ref_ihvp, approx_ihvp, grads_2))
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

    all_results: List[Dict] = []
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    code_info = provenance.git_info()

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

        dataset, test_dataset = prepare_datasets_for_model(
            base_train, base_test, model_config.loss
        )
        loss_fn = get_loss(model_config.loss)
        collector_dir = str(
            paths.collector_dir(dataset_name, model_id, epoch, strategy_str, reps)
        )
        collector_dir_corr = str(
            paths.collector_dir(
                dataset_name, model_id, epoch, strategy_str, reps, corr=True
            )
        )

        collector_data, collector_data_corr = collect_activations_gradients(
            model=model,
            params=params,
            dataset=dataset,
            loss_fn=loss_fn,
            analysis_cfg=analysis_cfg,
            collector_dir=collector_dir,
            collector_dir_corr=collector_dir_corr,
            seed=seed,
        )

        probes = sample_vectors(
            vector_config=analysis_cfg.vector_config,
            model=model,
            params=params,
            inputs=dataset.inputs,
            targets=dataset.targets,
            loss_fn=loss_fn,
            seed=seed,
            repetitions=2,
        )
        grads_1, grads_2 = probes[0], probes[1]

        model_ctx = ModelContext.create(
            model=model,
            params=params,
            dataset=dataset,
            loss_fn=loss_fn,
        )

        # Hessian computer factors (kfac, ekfac, ...) cache one level inside
        # the collector dir, so the raw collector data and the derived
        # per-method factors don't share a namespace.
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
                    damping, pif,
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
        help="Models YAML (models list, dataset, seed, epochs).",
    )
    parser.add_argument(
        "--analysis-config",
        default="experiments/configs/hessian_analysis.yaml",
        help="Analysis YAML (HessianAnalysisConfig).",
    )
    args = parser.parse_args()
    main(args.config, args.analysis_config)
