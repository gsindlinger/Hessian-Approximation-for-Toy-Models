"""Unified analysis entry point: dispatches ``error_analysis``,
``attribution``, and ``lds`` stages over a shared ``(models × epochs)``
iteration axis.

A single Hydra config (``AnalysisConfig``) drives everything — stage-specific
sub-configs (``error_analysis``, ``attribution``, ``lds``) are only consulted
when their stage name is listed in ``stages``. One combined JSON is produced
per invocation; it carries only the keys for stages that actually ran.

Pipelines:

* **Error analysis only**         ``stages=[error_analysis]``
* **Attribution only**            ``stages=[attribution]``
* **LDS only** (pre-existing dirs)``stages=[lds]``
* **Attribution → LDS**           ``stages=[attribution,lds]``
* **Full pipeline** (all three)   ``stages=[error_analysis,attribution,lds]``

Stage execution order is fixed (error_analysis → attribution → lds)
regardless of how they are listed in the config.

Usage:
    # End-to-end from a list of pretrained models
    uv run python -m experiments.analysis \\
        --config-name=analysis \\
        --config-path=../configs \\
        +override_config=path/to/best_models.yaml \\
        stages=[error_analysis,attribution,lds] \\
        epochs=[10,100,1000]

    # LDS-only on pre-existing attribution_scores/*/ directories
    uv run python -m experiments.analysis \\
        --config-name=analysis \\
        --config-path=../configs \\
        stages=[lds] \\
        models=[path/to/model] \\
        epochs=[100]

Attribution scores are stored as per-method ``.npy`` files inside
``{model}/attribution_scores/{epoch_label}/`` — the ``attribution`` stage
writes them and the ``lds`` stage memory-maps them.
"""

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import hydra
import jax
from hydra.core.config_store import ConfigStore
from jax.random import PRNGKey
from jaxtyping import Array, Float
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from experiments.utils import (
    json_safe,
    load_experiment_override_from_yaml,
    to_dataclass,
)
from src.config import (
    AnalysisConfig,
    AnalysisStage,
    ComputationType,
    HessianAnalysisConfig,
    LossType,
    ModelConfig,
    PseudoTargetGenerationStrategy,
    RegularizationStrategy,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import sample_vectors
from src.utils.attribution import (
    compute_attribution_scores_for_model,
    save_attribution_scores,
)
from src.utils.data.data import Dataset, DownloadableDataset
from src.utils.lds import compute_lds_for_model
from src.utils.loss import get_loss
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.train import (
    check_saved_model,
    evaluate_loss_and_classification_accuracy,
    load_model_checkpoint,
)
from src.utils.utils import cleanup_memory

logger = logging.getLogger(__name__)

# Quiet chatter from deep library code when the error-analysis stage is running.
logging.getLogger("experiments.utils").setLevel(logging.WARNING)
logging.getLogger("src.hessians.computer.computer").setLevel(logging.WARNING)

cs = ConfigStore.instance()
cs.store(name="analysis_experiment", node=AnalysisConfig)


def _attribution_output_dir(
    model_dir: str, epoch: Optional[int], base_dir: Optional[str] = None
) -> str:
    """Attribution-score directory for ``(model, epoch)``.

    Each call resolves to a directory holding one ``{method}.npy`` per
    attribution method.

    ``base_dir=None`` → ``{model_dir}/attribution_scores/{epoch_label}/``.
    Otherwise → ``{base_dir}/{model_basename}/{epoch_label}/`` (per-model
    subdir preserved so multiple models can share one output root).
    """
    label = f"epoch_{epoch}" if epoch is not None else "final"
    if base_dir is None:
        return os.path.join(model_dir, "attribution_scores", label)
    return os.path.join(base_dir, os.path.basename(model_dir), label)


def _attribution_read_dirs(
    model_dir: str,
    epoch: Optional[int],
    lds_override_attribution_dirs: Optional[List[str]],
    attribution_dir: Optional[str],
) -> "str | List[str]":
    """Resolve the LDS stage's attribution-read directory(ies) for one model/epoch.

    Precedence: explicit ``lds.override_attribution_dirs`` → ``attribution_dir``
    → default per-model path. A list override yields a list (one resolved
    directory per base dir); otherwise a single string is returned.
    """
    if lds_override_attribution_dirs:
        return [
            _attribution_output_dir(model_dir, epoch, base_dir=d)
            for d in lds_override_attribution_dirs
        ]
    return _attribution_output_dir(model_dir, epoch, base_dir=attribution_dir)


def _expand_pairs(
    models: List[str], epochs: Optional[List[int]]
) -> List[Tuple[str, Optional[int]]]:
    """Expand ``models × epochs`` to a flat list of ``(model, epoch)`` pairs."""
    if not epochs:
        return [(m, None) for m in models]
    return [(m, e) for m in models for e in epochs]


def _checkpoint_metrics(
    model_directory: str,
    epoch: Optional[int],
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> Dict:
    """Loss / accuracy / parameter count for the loaded checkpoint."""
    params, model, model_config, _ = load_model_checkpoint(model_directory, epoch=epoch)
    loss_fn = get_loss(model_config.loss)

    train_inputs = train_dataset.inputs
    train_targets = train_dataset.targets
    test_inputs = test_dataset.inputs
    test_targets = test_dataset.targets
    if model_config.loss == LossType.MSE:
        train_inputs, test_inputs = Dataset.normalize_data(train_inputs, test_inputs)
        train_targets, test_targets = Dataset.normalize_data(train_targets, test_targets)
        assert test_inputs is not None and test_targets is not None

    test_loss, test_acc = evaluate_loss_and_classification_accuracy(
        model, params, test_inputs, test_targets, loss_fn
    )
    train_loss, train_acc = evaluate_loss_and_classification_accuracy(
        model, params, train_inputs, train_targets, loss_fn
    )
    num_parameters = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    return {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "num_parameters": num_parameters,
    }


def _collect_data(
    model,
    params,
    model_config: ModelConfig,
    dataset: Dataset,
    collector_dir: str,
    hessian_config: HessianAnalysisConfig,
    seed: int,
):
    """Prepare gradients, collector data, and the model context."""

    train_inputs, train_targets = dataset.inputs, dataset.targets
    loss_fn = get_loss(model_config.loss)

    logger.info("[HESSIAN] Sampling gradient vectors")
    grads: Float[Array, "2 n_vectors num_params"] = sample_vectors(
        vector_config=hessian_config.vector_config,
        model=model,
        params=params,
        inputs=train_inputs,
        targets=train_targets,
        loss_fn=loss_fn,
        seed=seed,
        repetitions=2,
    )
    grads_1, grads_2 = grads[0], grads[1]
    cleanup_memory("gradient_sampling")

    logger.info("[HESSIAN] Collecting Activations & Gradients")
    strategy = hessian_config.computation_config.estimators_config.pseudo_target_generation_strategy

    def _make_collector():
        return CollectorActivationsGradients(
            model=model,
            params=params,
            loss_fn=loss_fn,
            pseudo_target_repetitions=hessian_config.computation_config.estimators_config.pseudo_target_generation_repetitions,
            pseudo_target_strategy=strategy,
        )

    collected_data: DataActivationsGradients = _make_collector().collect(
        dataset=Dataset(train_inputs, train_targets),
        save_directory=collector_dir,
        try_load=True,
        rng_key=PRNGKey(seed),
    )

    # For MCMC, use an independent sample set for EKFAC's eigenvalue correction
    # so the correction is not fit on the same samples as the covariances.
    if strategy == PseudoTargetGenerationStrategy.MCMC:
        collected_data_corr = _make_collector().collect(
            dataset=Dataset(train_inputs, train_targets),
            save_directory=f"{collector_dir}_corr",
            try_load=True,
            rng_key=PRNGKey(seed + 1),
        )
    else:
        collected_data_corr = collected_data

    model_ctx = ModelContext.create(
        dataset=Dataset(train_inputs, train_targets),
        model=model,
        params=params,
        loss_fn=loss_fn,
    )
    return grads_1, grads_2, collected_data, collected_data_corr, model_ctx


def _compute_hessian_comparison_for_single_model(
    hessian_config: HessianAnalysisConfig,
    collector_data: DataActivationsGradients,
    collector_data_corr: DataActivationsGradients,
    model_ctx: ModelContext,
    grads_1: Float[Array, "*batch_size n_params"],
    grads_2: Float[Array, "*batch_size n_params"],
    build_base_dir: str,
    compute_approximation_error: bool = True,
) -> Dict:
    """Compute the Hessian comparison specified in ``hessian_config``."""
    from experiments.utils import block_tree

    results: Dict = {
        "matrix_comparisons": {},
        "hvp_comparisons": {},
        "ihvp_comparisons": {},
    }

    pseudo_inverse_factor: Optional[float] = None
    damping: Optional[float] = None
    reg_strategy = (
        hessian_config.computation_config.estimators_config.regularization_strategy
    )
    reg_value = hessian_config.computation_config.estimators_config.regularization_value

    if reg_strategy in (
        RegularizationStrategy.AUTO_MEAN_EIGENVALUE,
        RegularizationStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION,
    ):
        logger.info("[HESSIAN] Using EKFAC to estimate damping for other methods.")
        ekfac_computer = EKFACComputer(
            compute_context=collector_data, corr_context=collector_data_corr
        )
        ekfac_computer.build(base_directory=build_base_dir)
        damping = ekfac_computer.get_damping(
            damping_strategy=reg_strategy, factor=reg_value
        )
        logger.info(f"[HESSIAN] Using damping: {damping:.6f}")
        results["damping"] = damping
    elif reg_strategy == RegularizationStrategy.FIXED:
        damping = reg_value
        logger.info(f"[HESSIAN] Using fixed damping: {damping:.6f}")
        results["damping"] = damping
    elif reg_strategy == RegularizationStrategy.PSEUDO_INVERSE:
        pseudo_inverse_factor = reg_value
        logger.info(
            f"[HESSIAN] Using pseudo-inverse with factor: {pseudo_inverse_factor:.6f}"
        )
        results["pseudo_inverse_factor"] = pseudo_inverse_factor

    comp_config = hessian_config.computation_config

    for reference_approx in comp_config.comparison_references:
        logger.info(f"[HESSIAN] Using {reference_approx.value} as reference")

        reference_data = HessianComputerRegistry.get_compute_context(
            reference_approx, collector_data, model_ctx
        )
        reference_computer = HessianComputerRegistry.get_computer(
            reference_approx, reference_data, corr_context=collector_data_corr
        )
        reference_computer.build(base_directory=build_base_dir)

        if ComputationType.MATRIX in comp_config.computation_types:
            ref_hessian = block_tree(
                reference_computer.estimate_hessian(),
                f"{reference_approx.value}_matrix",
            )
            matrix_approxs = [
                a
                for a in comp_config.estimators_config.approximators
                if a != reference_approx
            ]
            bar = tqdm(
                matrix_approxs,
                desc=f"matrix vs {reference_approx.value}",
                leave=True,
            )
            for approx in bar:
                bar.set_postfix_str(approx.value)
                approx_data = HessianComputerRegistry.get_compute_context(
                    approximator=approx,
                    collector_data=collector_data,
                    model_ctx=model_ctx,
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data, corr_context=collector_data_corr
                )
                approx_computer.build(base_directory=build_base_dir)

                for metric in hessian_config.matrix_config.metrics:
                    results["matrix_comparisons"].setdefault(metric.value, {})
                    results["matrix_comparisons"][metric.value].setdefault(
                        reference_approx.value, {}
                    )
                    score = approx_computer.compare_full_hessian_estimates(
                        comparison_matrix=ref_hessian,
                        metric=metric,
                    )
                    results["matrix_comparisons"][metric.value][reference_approx.value][
                        approx.value
                    ] = float(score)
            del ref_hessian
            cleanup_memory(f"{reference_approx.value}_matrix")

        if ComputationType.HVP in comp_config.computation_types:
            ref_hvp = block_tree(
                reference_computer.estimate_hvp(grads_1),
                f"{reference_approx.value}_hvp",
            )
            hvp_approxs = [
                a
                for a in comp_config.estimators_config.approximators
                if a != reference_approx
            ]
            bar = tqdm(
                hvp_approxs,
                desc=f"hvp vs {reference_approx.value}",
                leave=True,
            )
            for approx in bar:
                bar.set_postfix_str(approx.value)
                approx_data = HessianComputerRegistry.get_compute_context(
                    approximator=approx,
                    collector_data=collector_data,
                    model_ctx=model_ctx,
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data, corr_context=collector_data_corr
                )
                assert isinstance(approx_computer, HessianEstimator), (
                    "HVP comparisons require HessianEstimator"
                )
                approx_computer.build(base_directory=build_base_dir)
                approx_hvp = block_tree(
                    approx_computer.estimate_hvp(grads_1),
                    f"{approx.value}_hvp",
                )
                for metric in hessian_config.vector_config.metrics:
                    results["hvp_comparisons"].setdefault(metric.name, {})
                    results["hvp_comparisons"][metric.name].setdefault(
                        reference_approx.value, {}
                    )
                    score = metric.compute(ref_hvp, approx_hvp, grads_2)
                    results["hvp_comparisons"][metric.name][reference_approx.value][
                        approx.value
                    ] = float(score)
                del approx_hvp
            del ref_hvp
            cleanup_memory(f"{reference_approx.value}_hvp")

        if ComputationType.IHVP in comp_config.computation_types:
            ref_ihvp = block_tree(
                reference_computer.estimate_ihvp(
                    grads_1,
                    damping=damping,
                    pseudo_inverse_factor=pseudo_inverse_factor,
                ),
                f"{reference_approx.value}_ihvp",
            )
            ihvp_approxs = [
                a
                for a in comp_config.estimators_config.approximators
                if a != reference_approx
            ]
            bar = tqdm(
                ihvp_approxs,
                desc=f"ihvp vs {reference_approx.value}",
                leave=True,
            )
            for approx in bar:
                bar.set_postfix_str(approx.value)
                approx_data = HessianComputerRegistry.get_compute_context(
                    approximator=approx,
                    collector_data=collector_data,
                    model_ctx=model_ctx,
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data, corr_context=collector_data_corr
                )
                assert isinstance(approx_computer, HessianEstimator), (
                    "IHVP comparisons require HessianEstimator"
                )
                approx_computer.build(base_directory=build_base_dir)
                approx_ihvp = block_tree(
                    approx_computer.estimate_ihvp(
                        grads_1,
                        damping=damping,
                        pseudo_inverse_factor=pseudo_inverse_factor,
                    ),
                    f"{approx.value}_ihvp",
                )
                for metric in hessian_config.vector_config.metrics:
                    results["ihvp_comparisons"].setdefault(metric.name, {})
                    results["ihvp_comparisons"][metric.name].setdefault(
                        reference_approx.value, {}
                    )
                    score = metric.compute(ref_ihvp, approx_ihvp, grads_2)
                    results["ihvp_comparisons"][metric.name][reference_approx.value][
                        approx.value
                    ] = float(score)

                if compute_approximation_error:
                    round_trip_V = reference_computer.estimate_hvp(approx_ihvp)
                    approx_error = VectorMetric.RELATIVE_ERROR.compute(
                        grads_1, round_trip_V, x=None, power=2.0
                    )
                    results.setdefault("ihvp_round_trip_approximation_errors", {})
                    results["ihvp_round_trip_approximation_errors"].setdefault(
                        reference_approx.value, {}
                    )
                    results["ihvp_round_trip_approximation_errors"][
                        reference_approx.value
                    ][approx.value] = float(approx_error)

                del approx_ihvp
            del ref_ihvp
            cleanup_memory(f"{reference_approx.value}_ihvp")

    return results


def analyze_single_model(
    model_directory: str,
    dataset: Dataset,
    hessian_config: HessianAnalysisConfig,
    seed: int,
    epoch: Optional[int] = None,
) -> Dict:
    """Run the Hessian comparison on a single ``(model, epoch)`` pair."""
    params, model, model_config, metadata = load_model_checkpoint(
        model_directory, epoch=epoch
    )

    epoch_str = f"epoch_{epoch}" if epoch is not None else "final"
    logger.info("=" * 70)
    logger.info("[HESSIAN] %s (%s)", model_config.get_model_display_name(), epoch_str)
    logger.info("Model directory: %s", model_config.directory)
    logger.info("=" * 70)

    if metadata:
        logger.info("Val loss: %s", metadata.get("val_loss", "N/A"))
        if model_config.loss == LossType.CROSS_ENTROPY:
            logger.info("Val accuracy: %s", metadata.get("val_accuracy", "N/A"))

    # In-place normalisation for regression tasks — mirrors the previous
    # behaviour of the standalone hessian_analysis entry point.
    if model_config.loss == LossType.MSE:
        dataset.inputs, _ = Dataset.normalize_data(dataset.inputs, None)
        dataset.targets, _ = Dataset.normalize_data(dataset.targets, None)

    assert model_config.directory is not None, (
        "directory must be set in model_config for Hessian analysis."
    )

    # Collector cache is shared across worktrees: anchor to the absolute
    # caller-supplied model_directory. Build outputs are per-tree: anchor to
    # the model_config.directory which may differ by worktree.
    collector_dir = os.path.join(model_directory, "collector")
    build_base_dir = os.path.join(model_config.directory, "collector")
    if epoch is not None:
        collector_dir = os.path.join(collector_dir, f"epoch_{epoch}")
        build_base_dir = os.path.join(build_base_dir, f"epoch_{epoch}")

    grads_1, grads_2, collector_data, collector_data_corr, model_ctx = _collect_data(
        model=model,
        params=params,
        model_config=model_config,
        dataset=Dataset(dataset.inputs, dataset.targets),
        collector_dir=collector_dir,
        hessian_config=hessian_config,
        seed=seed,
    )

    hessian_results = _compute_hessian_comparison_for_single_model(
        hessian_config=hessian_config,
        collector_data=collector_data,
        collector_data_corr=collector_data_corr,
        model_ctx=model_ctx,
        grads_1=grads_1,
        grads_2=grads_2,
        build_base_dir=build_base_dir,
    )

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_config.directory,
        "epoch": epoch,
        "model_config": asdict(model_config),
        "num_parameters": model.num_params,
        "metadata": metadata or {},
        "hessian_analysis": hessian_results,
    }


@hydra.main(
    version_base="1.3", config_name="analysis_experiment", config_path="../configs"
)
def main(cfg: DictConfig) -> Dict:
    OmegaConf.resolve(cfg)

    override_file = cfg.get("override_config", None)
    if override_file:
        logger.info("[CONFIG] Overriding from: %s", override_file)
        model_dirs, dataset_config, seed, epochs = load_experiment_override_from_yaml(
            override_file
        )
        cfg.models = model_dirs
        if dataset_config is not None:
            cfg.dataset = asdict(dataset_config)
        if seed is not None:
            cfg.seed = seed
        if epochs is not None:
            cfg.epochs = epochs

    config: AnalysisConfig = to_dataclass(AnalysisConfig, cfg)  # type: ignore[assignment]

    if not config.models:
        raise ValueError(
            "AnalysisConfig.models is empty — pass models=[...] or supply an "
            "+override_config YAML that provides them."
        )
    if not config.stages:
        raise ValueError(
            "AnalysisConfig.stages is empty — pick at least one of "
            "[error_analysis, attribution, lds]."
        )

    # Canonicalise stage order (error_analysis → attribution → lds)
    # independently of how the user listed them.
    stage_order = [
        AnalysisStage.ERROR_ANALYSIS,
        AnalysisStage.ATTRIBUTION,
        AnalysisStage.LDS,
    ]
    stages_set = set(config.stages)
    ordered_stages = [s for s in stage_order if s in stages_set]

    pairs = _expand_pairs(config.models, config.epochs)
    if config.epochs:
        for m in config.models:
            if not check_saved_model(m, config.epochs):
                raise FileNotFoundError(
                    f"Model at {m} is missing one or more checkpoints for "
                    f"epochs={config.epochs}"
                )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info("=" * 70)
    logger.info("Analysis: %s  [%s]", config.experiment_name, timestamp)
    logger.info(
        "Stages: %s  |  Seed: %d  |  Dataset: %s",
        [s.value for s in ordered_stages],
        config.seed,
        config.dataset.name.value,
    )
    logger.info(
        "Models: %d  |  Epochs: %s  |  Pairs to analyse: %d",
        len(config.models),
        config.epochs,
        len(pairs),
    )
    logger.info("=" * 70)

    full_dataset = DownloadableDataset.load(
        dataset=config.dataset.name,
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )
    train_dataset, test_dataset = full_dataset.train_test_split(
        test_size=config.dataset.test_size, seed=config.seed
    )
    logger.info(
        "Split: %d train / %d test",
        len(train_dataset.inputs),
        len(test_dataset.inputs),
    )

    # Query slice shared by attribution + lds stages (same query count).
    n_query = min(config.lds.num_test_examples, len(test_dataset.inputs))
    query_inputs = test_dataset.inputs[:n_query]
    query_targets = test_dataset.targets[:n_query]

    results: List[Dict] = []
    for i, (model_dir, epoch) in enumerate(pairs, 1):
        label = f"epoch={epoch}" if epoch is not None else "final"
        logger.info(
            "[%d/%d] %s (%s)", i, len(pairs), os.path.basename(model_dir), label
        )

        item: Dict = {
            "model_directory": model_dir,
            "epoch": epoch,
            "metadata": _checkpoint_metrics(
                model_dir, epoch, train_dataset, test_dataset
            ),
        }

        if AnalysisStage.ERROR_ANALYSIS in stages_set:
            logger.info("[ERROR_ANALYSIS] %s (%s)", model_dir, label)
            hessian_result = analyze_single_model(
                model_directory=model_dir,
                dataset=Dataset(train_dataset.inputs, train_dataset.targets),
                hessian_config=config.error_analysis,
                seed=config.seed,
                epoch=epoch,
            )
            item["model_name"] = hessian_result["model_name"]
            item["error_analysis"] = hessian_result["hessian_analysis"]
            cleanup_memory(f"error_analysis_{i}")

        if AnalysisStage.ATTRIBUTION in stages_set:
            out_dir = _attribution_output_dir(
                model_dir, epoch, base_dir=config.attribution_dir
            )
            logger.info("[ATTRIBUTION] %s → %s", model_dir, out_dir)
            attributions = compute_attribution_scores_for_model(
                model_directory=model_dir,
                train_dataset=train_dataset,
                query_inputs=query_inputs,
                query_targets=query_targets,
                hessian_config=config.attribution,
                seed=config.seed,
                epoch=epoch,
            )
            save_attribution_scores(attributions, out_dir)
            item["attribution"] = {
                "output_dir": out_dir,
                "methods": sorted(attributions.keys()),
            }
            cleanup_memory(f"attribution_{i}")

        if AnalysisStage.LDS in stages_set:
            attribution_dir = _attribution_read_dirs(
                model_dir,
                epoch,
                lds_override_attribution_dirs=config.lds.override_attribution_dirs,
                attribution_dir=config.attribution_dir,
            )
            missing = [
                d
                for d in (
                    [attribution_dir]
                    if isinstance(attribution_dir, str)
                    else attribution_dir
                )
                if not os.path.isdir(d)
            ]
            if missing:
                raise FileNotFoundError(
                    f"[LDS] Missing attribution directory(ies): {missing}. "
                    f"Run with stages=[...,attribution,lds] to generate them, "
                    f"or place per-method .npy files at those paths by hand."
                )
            lds_result = compute_lds_for_model(
                model_directory=model_dir,
                attribution_dirs=attribution_dir,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                lds_config=config.lds,
                dataset_config=config.dataset,
                seed=config.seed,
                epoch=epoch,
            )
            item.setdefault("model_name", lds_result["model_name"])
            item["lds"] = {
                "lds_scores": lds_result["lds_scores"],
                "num_subsets": lds_result["num_subsets"],
                "subset_fraction": lds_result["subset_fraction"],
                "reps_per_model": lds_result["reps_per_model"],
                "num_queries": lds_result["num_queries"],
                "attribution_dirs": lds_result["attribution_dirs"],
            }
            cleanup_memory(f"lds_{i}")

        results.append(item)

    # Only serialise stage configs that actually ran.
    config_dump: Dict = {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "dataset": asdict(config.dataset),
        "models": list(config.models),
        "epochs": config.epochs,
        "stages": [s.value for s in ordered_stages],
    }
    if AnalysisStage.ERROR_ANALYSIS in stages_set:
        config_dump["error_analysis"] = asdict(config.error_analysis)
    if AnalysisStage.ATTRIBUTION in stages_set:
        config_dump["attribution"] = asdict(config.attribution)
    if AnalysisStage.LDS in stages_set:
        config_dump["lds"] = asdict(config.lds)

    full_results = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "stages": [s.value for s in ordered_stages],
        "config": json_safe(config_dump),
        "results": json_safe(results),
    }

    os.makedirs(config.results_output_dir, exist_ok=True)
    output_file = os.path.join(config.results_output_dir, f"{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info("=" * 70)
    logger.info("Analysis complete — results saved to %s", output_file)

    if AnalysisStage.LDS in stages_set:
        logger.info("LDS summary:")
        for item in results:
            if "lds" not in item:
                continue
            logger.info(
                "  %s (epoch=%s):",
                item.get("model_name", item["model_directory"]),
                item["epoch"],
            )
            for method, s in item["lds"]["lds_scores"].items():
                logger.info(
                    "    %-12s %.4f ± %.4f  (95%% CI [%.4f, %.4f])",
                    method,
                    s["mean_lds"],
                    s["std_lds"],
                    s["ci_low"],
                    s["ci_high"],
                )
    logger.info("=" * 70)

    return full_results


if __name__ == "__main__":
    main()
