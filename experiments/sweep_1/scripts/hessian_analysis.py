"""
Standalone Hessian analysis script.

Loads pre-trained models and performs Hessian approximation comparisons.
Can override models from command line with models_file parameter.

Usage: TODO
"""

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Dict, List

import hydra
from hydra.core.config_store import ConfigStore
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

from experiments.sweep_1.scripts.utils import (
    block_tree,
    cleanup_memory,
    load_experiment_override_from_yaml,
    to_dataclass,
)
from src.config import (
    ComputationType,
    ExperimentConfig,
    HessianAnalysisConfig,
    HessianApproximator,
    LossType,
    ModelConfig,
)
from src.hessians.approximator.ekfac import EKFACApproximator
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import EKFACData, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_vectors
from src.utils.data.data import Dataset, DownloadableDataset
from src.utils.loss import get_loss
from src.utils.train import load_model_checkpoint

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="hessian_experiment", node=ExperimentConfig)


def collect_approximation_data(
    model,
    params,
    model_config: ModelConfig,
    dataset: Dataset,
    collector_dirs: List[str],
    ekfac_dir: str,
    hessian_config: HessianAnalysisConfig,
    seed: int,
):
    """Prepare all data needed for Hessian analysis."""
    train_ds, _ = dataset.train_test_split(test_size=0.1, seed=seed)
    train_inputs, train_targets = train_ds.inputs, train_ds.targets
    loss_fn = get_loss(model_config.loss)

    # Sample gradient vectors
    logger.info("[HESSIAN] Sampling gradient vectors")
    grads_1 = sample_vectors(
        vector_config=hessian_config.vector_config,
        model=model,
        params=params,
        inputs=train_inputs,
        targets=train_targets,
        loss_fn=loss_fn,
        seed=seed,
    )

    grads_2 = sample_vectors(
        vector_config=hessian_config.vector_config,
        model=model,
        params=params,
        inputs=train_inputs,
        targets=train_targets,
        loss_fn=loss_fn,
        seed=seed + 1,
    )
    cleanup_memory("gradient_sampling")

    # Collect Activation & Gradients (2 runs)
    logger.info("[HESSIAN] Collecting Activations & Gradients")
    for run_idx, collector_dir in enumerate(collector_dirs):
        run_seed = seed + run_idx
        pseudo_targets = generate_pseudo_targets(
            model=model,
            inputs=train_inputs,
            params=params,
            loss_fn=loss_fn,
            rng_key=PRNGKey(run_seed),
        )
        collector_data = Dataset(train_inputs, train_targets).replace_targets(
            pseudo_targets
        )
        cleanup_memory("pseudo_target_generation")

        collector = CollectorActivationsGradients(
            model=model, params=params, loss_fn=loss_fn
        )
        collector.collect(
            inputs=collector_data.inputs,
            targets=collector_data.targets,
            save_directory=collector_dir,
            try_load=True,
        )
        cleanup_memory(f"collection_run_{run_idx}")

    # Build/Load EKFAC
    logger.info("[HESSIAN] Building EKFAC approximation")
    ekfac = EKFACApproximator(collector_dirs[0], collector_dirs[1])

    if not ekfac.data_exists(ekfac_dir):
        ekfac.build(config=model_config, save_directory=ekfac_dir)

    ekfac_data, _ = EKFACApproximator.load_data(ekfac_dir)
    assert isinstance(ekfac_data, EKFACData)
    cleanup_memory("ekfac_build")

    # Load collector data
    collector_single_data = CollectorActivationsGradients.load(collector_dirs[0])

    # Create model context
    model_ctx = ModelContext.create(
        dataset=Dataset(train_inputs, train_targets),
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    return grads_1, grads_2, ekfac_data, collector_single_data, model_ctx


def get_approximator_data(
    approximator: HessianApproximator,
    ekfac_data,
    collector_data,
    model_ctx,
):
    """Get the appropriate data for each approximator type."""
    if approximator in [HessianApproximator.FIM, HessianApproximator.BLOCK_FIM]:
        return collector_data
    elif approximator in [HessianApproximator.EKFAC, HessianApproximator.KFAC]:
        return ekfac_data
    elif approximator in [
        HessianApproximator.GNH,
        HessianApproximator.BLOCK_HESSIAN,
        HessianApproximator.EXACT,
    ]:
        return model_ctx
    else:
        raise ValueError(f"Unsupported approximator: {approximator}")


def compute_hessian_comparisons(
    hessian_config: HessianAnalysisConfig,
    ekfac_data,
    collector_data,
    model_ctx,
    grads_1,
    grads_2,
    damping: float,
) -> Dict:
    """Compute all Hessian comparisons specified in the config."""
    results = {
        "damping": float(damping),
        "matrix_comparisons": {},
        "hvp_comparisons": {},
        "ihvp_comparisons": {},
    }

    comp_config = hessian_config.computation_config

    # For each reference method
    for reference_approx in comp_config.comparison_references:
        logger.info(f"[HESSIAN] Using {reference_approx.value} as reference")

        reference_data = get_approximator_data(
            reference_approx, ekfac_data, collector_data, model_ctx
        )
        reference_computer = HessianComputerRegistry.get_computer(
            reference_approx, reference_data
        )

        # Matrix comparisons
        if ComputationType.MATRIX in comp_config.computation_types:
            logger.info(f"[HESSIAN] Computing {reference_approx.value} matrix")

            if isinstance(reference_computer, HessianComputer):
                ref_hessian = block_tree(
                    reference_computer.compute_hessian(damping=damping),
                    f"{reference_approx.value}_matrix",
                )
            elif isinstance(reference_computer, HessianEstimator):
                ref_hessian = block_tree(
                    reference_computer.estimate_hessian(damping),
                    f"{reference_approx.value}_matrix",
                )

            for approx in comp_config.approximators:
                if approx == reference_approx:
                    continue
                logger.info(
                    f"[HESSIAN] Comparing {reference_approx.value} vs {approx.value} (matrix)"
                )
                approx_data = get_approximator_data(
                    approx, ekfac_data, collector_data, model_ctx
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data
                )

                for metric in hessian_config.matrix_config.metrics:
                    results["matrix_comparisons"].setdefault(metric.value, {})
                    results["matrix_comparisons"][metric.value].setdefault(
                        reference_approx.value, {}
                    )

                    assert isinstance(approx_computer, HessianEstimator), (
                        "Matrix comparisons require HessianEstimator"
                    )
                    score = approx_computer.compare_full_hessian_estimates(
                        ref_hessian, damping, metric
                    )
                    results["matrix_comparisons"][metric.value][reference_approx.value][
                        approx.value
                    ] = float(score)

            del ref_hessian
            cleanup_memory(f"{reference_approx.value}_matrix")

        # HVP comparisons
        if ComputationType.HVP in comp_config.computation_types:
            logger.info(f"[HESSIAN] Computing {reference_approx.value} HVP")

            if isinstance(reference_computer, HessianComputer):
                ref_hvp = block_tree(
                    reference_computer.compute_hvp(grads_1, damping=damping),
                    f"{reference_approx.value}_hvp",
                )
            elif isinstance(reference_computer, HessianEstimator):
                ref_hvp = block_tree(
                    reference_computer.estimate_hvp(grads_1, damping),
                    f"{reference_approx.value}_hvp",
                )

            for approx in comp_config.approximators:
                if approx == reference_approx:
                    continue
                logger.info(
                    f"[HESSIAN] Comparing {reference_approx.value} vs {approx.value} (HVP)"
                )
                approx_data = get_approximator_data(
                    approx, ekfac_data, collector_data, model_ctx
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data
                )

                assert isinstance(approx_computer, HessianEstimator), (
                    "HVP comparisons require HessianEstimator"
                )

                approx_hvp = block_tree(
                    approx_computer.estimate_hvp(grads_1, damping),
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

        # IHVP comparisons
        if ComputationType.IHVP in comp_config.computation_types:
            logger.info(f"[HESSIAN] Computing {reference_approx.value} IHVP")

            if isinstance(reference_computer, HessianComputer):
                ref_ihvp = block_tree(
                    reference_computer.compute_ihvp(grads_1, damping=damping),
                    f"{reference_approx.value}_ihvp",
                )
            elif isinstance(reference_computer, HessianEstimator):
                ref_ihvp = block_tree(
                    reference_computer.estimate_ihvp(grads_1, damping),
                    f"{reference_approx.value}_ihvp",
                )

            for approx in comp_config.approximators:
                if approx == reference_approx:
                    continue
                logger.info(
                    f"[HESSIAN] Comparing {reference_approx.value} vs {approx.value} (IHVP)"
                )
                approx_data = get_approximator_data(
                    approx, ekfac_data, collector_data, model_ctx
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data
                )

                assert isinstance(approx_computer, HessianEstimator), (
                    "IHVP comparisons require HessianEstimator"
                )

                approx_ihvp = block_tree(
                    approx_computer.estimate_ihvp(grads_1, damping),
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

                del approx_ihvp

            del ref_ihvp
            cleanup_memory(f"{reference_approx.value}_ihvp")

    return results


def analyze_single_model(
    model_directory: str,
    dataset: Dataset,
    hessian_config: HessianAnalysisConfig,
    seed: int,
) -> Dict:
    """Run Hessian analysis on a single model."""
    # Load model and parameters
    params, model, model_config, metadata = load_model_checkpoint(model_directory)

    logger.info(f"{'=' * 70}")
    logger.info(f"[HESSIAN] Analyzing: {model_config.get_model_display_name()}")
    logger.info(f"Model directory: {model_config.directory}")
    logger.info(f"{'=' * 70}")

    # Log training metrics if available
    if metadata:
        val_loss = metadata.get("val_loss", "N/A")
        logger.info(f"Val loss: {val_loss}")
        if model_config.loss == LossType.CROSS_ENTROPY:
            val_acc = metadata.get("val_accuracy", "N/A")
            logger.info(f"Val accuracy: {val_acc}")

    # Prepare data
    assert model_config.directory is not None, (
        "directory must be set in model_config for Hessian analysis."
    )
    grads_1, grads_2, ekfac_data, collector_data, model_ctx = (
        collect_approximation_data(
            model=model,
            params=params,
            model_config=model_config,
            dataset=dataset,
            collector_dirs=[
                os.path.join(model_config.directory, "collector", f"run_{i + 1}")
                for i, _ in enumerate(range(2))
            ],
            ekfac_dir=os.path.join(model_config.directory, "ekfac"),
            hessian_config=hessian_config,
            seed=seed,
        )
    )

    # Compute damping
    damping = EKFACApproximator.get_damping(
        ekfac_data,
        hessian_config.computation_config.damping_strategy,
        hessian_config.computation_config.damping,
    )
    logger.info(f"[HESSIAN] Damping: {damping:.6e}")

    # Run comparisons
    hessian_results = compute_hessian_comparisons(
        hessian_config,
        ekfac_data,
        collector_data,
        model_ctx,
        grads_1,
        grads_2,
        damping,
    )

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_config.directory,
        "model_config": asdict(model_config),
        "num_parameters": model.num_params,
        "metadata": metadata or {},
        "hessian_analysis": hessian_results,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


@hydra.main(
    version_base="1.3", config_name="hessian_experiment", config_path="../configs"
)
def main(cfg: DictConfig) -> Dict:
    OmegaConf.resolve(cfg)

    # Check if models should be loaded from file
    override_file = cfg.get("override_config", None)

    if override_file:
        logger.info(f"[CONFIG] Overriding config data from: {override_file}")
        model_directories, dataset, seed = load_experiment_override_from_yaml(
            override_file
        )

        # Update config with loaded models
        cfg.models = model_directories
        if dataset is not None:
            cfg.dataset = asdict(dataset)
        if seed is not None:
            cfg.seed = seed

    config: ExperimentConfig = to_dataclass(ExperimentConfig, cfg)  # type: ignore

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info(f"{'=' * 70}")
    logger.info(f"Starting Hessian Analysis: {config.experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Dataset: {config.dataset.name.value}")
    logger.info(f"Models to analyze: {len(config.models)}")
    logger.info(f"{'=' * 70}")

    # Load dataset
    dataset = DownloadableDataset.load(
        dataset=config.dataset.name,
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )
    logger.info(f"Loaded dataset: {config.dataset.name.value}")

    # Run Hessian analysis
    logger.info(f"{'#' * 70}")
    logger.info("HESSIAN ANALYSIS")
    logger.info(f"{'#' * 70}")

    hessian_results = []
    for i, model_config in enumerate(config.models, 1):
        logger.info(f"[MODEL {i}/{len(config.models)}]")

        result = analyze_single_model(
            model_config,
            dataset,
            config.hessian_analysis,
            config.seed,
        )
        hessian_results.append(result)

        cleanup_memory(f"model_{i}")

    # Save results
    results_dir = config.hessian_analysis.results_output_dir
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(
        results_dir,
        f"{timestamp}.json",
    )

    full_results = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "hessian_config": asdict(config.hessian_analysis),
        "results": hessian_results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"{'=' * 70}")
    logger.info("Hessian Analysis Complete!")
    logger.info(f"Models analyzed: {len(hessian_results)}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'=' * 70}")

    return full_results


if __name__ == "__main__":
    main()
