"""
Standalone Hessian analysis script.

Loads pre-trained models and performs Hessian approximation comparisons.
Supports analyzing models at specific training epochs or final checkpoints.

Usage:
    # Basic run with final checkpoints only
    uv run python -m experiments.hessian_analysis \\
        --config-name=hessian_analysis \\
        --config-path=../configs
    
    # Analyze specific epochs across all models
    uv run python -m experiments.hessian_analysis \\
        --config-name=hessian_analysis \\
        --config-path=../configs \\
        epochs=[10,50,100]
    
    # Use models from training output with epoch analysis
    uv run python -m experiments.hessian_analysis \\
        --config-name=hessian_analysis \\
        +override_config=path/to/best_models.yaml \\
        epochs=[25,50,75,100]
    
    # Override vector config parameters
    uv run python -m experiments.hessian_analysis \\
        --config-name=hessian_analysis \\
        hessian_analysis.vector_config.num_samples=500 \\
        epochs=[100]

Epoch Analysis:
    - If epochs=None (default): analyzes only final checkpoints
    - If epochs=[10,50,100]: analyzes each model at epochs 10, 50, and 100
    - Total analyses = num_models × num_epochs (or num_models if epochs=None)
    - All specified epoch checkpoints must exist before analysis begins

Notes:
    - The override_config parameter expects a YAML file with model paths, dataset config, and seed
    - Epoch-specific collector data is stored separately to avoid conflicts
    - Results include epoch metadata for tracking training progression
    - Vector config (num_samples) can be adjusted based on dataset size
    - Results are saved as timestamped JSON files in hessian_analysis.results_output_dir
"""

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from jax.random import PRNGKey
from jaxtyping import Array, Float
from omegaconf import DictConfig, OmegaConf

from experiments.utils import (
    block_tree,
    cleanup_memory,
    json_safe,
    load_experiment_override_from_yaml,
    to_dataclass,
)
from src.config import (
    ComputationType,
    RegularizationStrategy,
    ExperimentConfig,
    HessianAnalysisConfig,
    LossType,
    ModelConfig,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import (
    generate_pseudo_targets_dataset,
    sample_vectors,
)
from src.utils.data.data import Dataset, DownloadableDataset
from src.utils.loss import get_loss
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.train import check_saved_model, load_model_checkpoint

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="hessian_experiment", node=ExperimentConfig)


def collect_data(
    model,
    params,
    model_config: ModelConfig,
    dataset: Dataset,
    collector_dirs: Tuple[str, str],
    hessian_config: HessianAnalysisConfig,
    seed: int,
):
    """Prepare all data needed for Hessian analysis."""
    train_inputs, train_targets = dataset.inputs, dataset.targets
    loss_fn = get_loss(model_config.loss)

    # Sample gradient vectors
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

    # Collect Activation & Gradients (2 runs)
    collected_data_list = []
    logger.info("[HESSIAN] Collecting Activations & Gradients")
    for run_idx, collector_dir in enumerate(collector_dirs):
        monte_carlo_repetitions = 3
        run_seed = seed + (run_idx * monte_carlo_repetitions)
        collector_data = generate_pseudo_targets_dataset(
            model=model,
            params=params,
            dataset=dataset,
            loss_fn=loss_fn,
            rng_key=PRNGKey(run_seed),
            monte_carlo_repetitions=monte_carlo_repetitions,
        )
        cleanup_memory("pseudo_target_generation")

        collector = CollectorActivationsGradients(
            model=model, params=params, loss_fn=loss_fn
        )
        collected_data_list.append(
            collector.collect(
                inputs=collector_data.inputs,
                targets=collector_data.targets,
                save_directory=collector_dir,
                try_load=True,
            )
        )
        cleanup_memory(f"collection_run_{run_idx}")

    assert len(collected_data_list) == 2, "Expected 2 runs of collected data."

    collected_data: Tuple[DataActivationsGradients, DataActivationsGradients] = (
        collected_data_list[0],
        collected_data_list[1],
    )

    # Create model context
    model_ctx = ModelContext.create(
        dataset=Dataset(train_inputs, train_targets),
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    return grads_1, grads_2, collected_data, model_ctx


def compute_hessian_comparison_for_single_model(
    hessian_config: HessianAnalysisConfig,
    collector_data: Tuple[DataActivationsGradients, DataActivationsGradients],
    model_ctx: ModelContext,
    grads_1: Float[Array, "*batch_size n_params"],
    grads_2: Float[Array, "*batch_size n_params"],
    model_directory: str,
    compute_approximation_error: bool = True,
) -> Dict:
    """Compute all Hessian comparisons specified in the config."""
    results = {
        "matrix_comparisons": {},
        "hvp_comparisons": {},
        "ihvp_comparisons": {},
    }

    # Use EKFAC as base for damping selection
    pseudo_inverse_factor: Optional[float] = None
    damping: Optional[float] = None
    if hessian_config.computation_config.regularization_strategy in (RegularizationStrategy.AUTO_MEAN_EIGENVALUE, RegularizationStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION):
        logger.info(
            "[HESSIAN] Using EKFAC to estimate damping for other methods."
        )
        ekfac_computer = EKFACComputer(compute_context=collector_data)
        ekfac_computer.build(base_directory=model_directory)
        damping = EKFACComputer.get_damping(
            ekfac_data=ekfac_computer.precomputed_data,
            damping_strategy=hessian_config.computation_config.regularization_strategy,
            factor=hessian_config.computation_config.regularization_value,
        )
        logger.info(f"[HESSIAN] Using damping: {damping:.6f}")
        results.setdefault("damping", damping) # type: ignore
    elif hessian_config.computation_config.regularization_strategy == RegularizationStrategy.FIXED:
        damping = hessian_config.computation_config.regularization_value
        logger.info(f"[HESSIAN] Using fixed damping: {damping:.6f}")
        results.setdefault("damping", damping) # type: ignore
    elif hessian_config.computation_config.regularization_strategy == RegularizationStrategy.PSEUDO_INVERSE:
        pseudo_inverse_factor = hessian_config.computation_config.regularization_value
        logger.info(
            f"[HESSIAN] Using pseudo-inverse with factor: {pseudo_inverse_factor:.6f}"
        )
        results.setdefault("pseudo_inverse_factor", pseudo_inverse_factor) # type: ignore
        

    comp_config = hessian_config.computation_config

    # For each reference method
    for reference_approx in comp_config.comparison_references:
        logger.info(f"[HESSIAN] Using {reference_approx.value} as reference")

        reference_data = HessianComputerRegistry.get_compute_context(
            reference_approx, collector_data, model_ctx
        )
        reference_computer = HessianComputerRegistry.get_computer(
            reference_approx, reference_data
        )
        if isinstance(reference_computer, HessianEstimator):
            reference_computer.build(base_directory=model_directory)

        # Matrix comparisons
        if ComputationType.MATRIX in comp_config.computation_types:
            logger.info(f"[HESSIAN] Computing {reference_approx.value} matrix")

            if isinstance(reference_computer, HessianComputer):
                ref_hessian = block_tree(
                    reference_computer.compute_hessian(),
                    f"{reference_approx.value}_matrix",
                )
            elif isinstance(reference_computer, HessianEstimator):
                ref_hessian = block_tree(
                    reference_computer.estimate_hessian(),
                    f"{reference_approx.value}_matrix",
                )

            for approx in comp_config.approximators:
                if approx == reference_approx:
                    continue
                logger.info(
                    f"[HESSIAN] Comparing {reference_approx.value} vs {approx.value} (matrix)"
                )
                approx_data = HessianComputerRegistry.get_compute_context(
                    approximator=approx,
                    collector_data=collector_data,
                    model_ctx=model_ctx,
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data
                )
                if isinstance(approx_computer, HessianEstimator):
                    approx_computer.build(base_directory=model_directory)
                else:
                    raise RuntimeError(
                        "Matrix comparisons require HessianEstimator, don't use exact Hessian as approximation method."
                    )

                # Evaluate metrics
                for metric in hessian_config.matrix_config.metrics:
                    results["matrix_comparisons"].setdefault(metric.value, {})
                    results["matrix_comparisons"][metric.value].setdefault(
                        reference_approx.value, {}
                    )
                    score = approx_computer._compare_full_hessian_estimates(
                        comparison_matrix=ref_hessian,
                        metric=metric,
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
                    reference_computer.compute_hvp(grads_1),
                    f"{reference_approx.value}_hvp",
                )
            elif isinstance(reference_computer, HessianEstimator):
                ref_hvp = block_tree(
                    reference_computer.estimate_hvp(grads_1),
                    f"{reference_approx.value}_hvp",
                )

            for approx in comp_config.approximators:
                if approx == reference_approx:
                    continue
                logger.info(
                    f"[HESSIAN] Comparing {reference_approx.value} vs {approx.value} (HVP)"
                )
                approx_data = HessianComputerRegistry.get_compute_context(
                    approximator=approx,
                    collector_data=collector_data,
                    model_ctx=model_ctx,
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data
                )

                assert isinstance(approx_computer, HessianEstimator), (
                    "HVP comparisons require HessianEstimator"
                )
                approx_computer.build(base_directory=model_directory)

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

        # IHVP comparisons
        if ComputationType.IHVP in comp_config.computation_types:
            logger.info(f"[HESSIAN] Computing {reference_approx.value} IHVP")

            if isinstance(reference_computer, HessianComputer):
                ref_ihvp = block_tree(
                    reference_computer.compute_ihvp(grads_1, damping=damping, pseudo_inverse_factor=pseudo_inverse_factor),
                    f"{reference_approx.value}_ihvp",
                )
            elif isinstance(reference_computer, HessianEstimator):
                ref_ihvp = block_tree(
                    reference_computer.estimate_ihvp(grads_1, damping=damping, pseudo_inverse_factor=pseudo_inverse_factor),
                    f"{reference_approx.value}_ihvp",
                )

            for approx in comp_config.approximators:
                if approx == reference_approx:
                    continue
                logger.info(
                    f"[HESSIAN] Comparing {reference_approx.value} vs {approx.value} (IHVP)"
                )
                approx_data = HessianComputerRegistry.get_compute_context(
                    approximator=approx,
                    collector_data=collector_data,
                    model_ctx=model_ctx,
                )
                approx_computer = HessianComputerRegistry.get_computer(
                    approx, approx_data
                )

                assert isinstance(approx_computer, HessianEstimator), (
                    "IHVP comparisons require HessianEstimator"
                )
                approx_computer.build(base_directory=model_directory)

                approx_ihvp = block_tree(
                    approx_computer.estimate_ihvp(grads_1, damping=damping, pseudo_inverse_factor=pseudo_inverse_factor),
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
                    # Compute round-trip approximation error
                    if isinstance(reference_computer, HessianComputer):
                        round_trip_V = reference_computer.compute_hvp(
                            approx_ihvp
                        )
                    else:
                        round_trip_V = reference_computer.estimate_hvp(
                            approx_ihvp
                        )
                    approx_error = VectorMetric.RELATIVE_ERROR.compute(
                        grads_1, round_trip_V, x=None, power=2.0
                    )
                    results.setdefault("ihvp_round_trip_approximation_errors", {})
                    results["ihvp_round_trip_approximation_errors"].setdefault(
                        reference_approx.value, {}
                    )
                    results["ihvp_round_trip_approximation_errors"][
                        reference_approx.value
                    ].setdefault(approx.value, {})
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
    """
    Run Hessian analysis on a single model at a specific epoch.

    Args:
        model_directory: Path to model checkpoint directory
        dataset: Training dataset to use for analysis
        hessian_config: Configuration for Hessian analysis
        seed: Random seed
        epoch: Optional specific epoch to analyze. If None, uses final checkpoint.

    Returns:
        Dictionary with analysis results including epoch information
    """
    # Load model and parameters
    params, model, model_config, metadata = load_model_checkpoint(
        model_directory, epoch=epoch
    )

    epoch_str = f"epoch_{epoch}" if epoch is not None else "final"
    logger.info(f"{'=' * 70}")
    logger.info(
        f"[HESSIAN] Analyzing: {model_config.get_model_display_name()} ({epoch_str})"
    )
    logger.info(f"Model directory: {model_config.directory}")
    logger.info(f"{'=' * 70}")

    # Log training metrics if available
    if metadata:
        val_loss = metadata.get("val_loss", "N/A")
        logger.info(f"Val loss: {val_loss}")
        if model_config.loss == LossType.CROSS_ENTROPY:
            val_acc = metadata.get("val_accuracy", "N/A")
            logger.info(f"Val accuracy: {val_acc}")

    # Normalize data for regression tasks
    if model_config.loss == LossType.MSE:
        dataset.inputs, _ = Dataset.normalize_data(dataset.inputs, None)
        dataset.targets, _ = Dataset.normalize_data(dataset.targets, None)

    # Prepare data
    assert model_config.directory is not None, (
        "directory must be set in model_config for Hessian analysis."
    )

    # Create epoch-specific collector directories
    collector_base = os.path.join(model_config.directory, "collector")
    if epoch is not None:
        collector_base = os.path.join(collector_base, f"epoch_{epoch}")

    grads_1, grads_2, collector_data, model_ctx = collect_data(
        model=model,
        params=params,
        model_config=model_config,
        dataset=Dataset(dataset.inputs, dataset.targets),
        collector_dirs=(
            os.path.join(collector_base, "run_1"),
            os.path.join(collector_base, "run_2"),
        ),
        hessian_config=hessian_config,
        seed=seed,
    )

    # Run comparisons
    hessian_results = compute_hessian_comparison_for_single_model(
        hessian_config,
        collector_data,
        model_ctx,
        grads_1,
        grads_2,
        model_directory,
    )

    result = {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_config.directory,
        "epoch": epoch,
        "model_config": asdict(model_config),
        "num_parameters": model.num_params,
        "metadata": metadata or {},
        "hessian_analysis": hessian_results,
    }

    return result


def expand_models_with_epochs(
    model_directories: List[str], epochs: Optional[List[int]]
) -> List[Tuple[str, Optional[int]]]:
    """
    Expand model list to include epoch-specific variants.

    Args:
        model_directories: List of model directory paths
        epochs: Optional list of epochs to analyze. If None, only analyze final checkpoints.

    Returns:
        List of (model_directory, epoch) tuples
    """
    if epochs is None:
        # No epochs specified, analyze final checkpoint only
        return [(model_dir, None) for model_dir in model_directories]

    # Expand each model to include all requested epochs
    expanded = []
    for model_dir in model_directories:
        for epoch in epochs:
            expanded.append((model_dir, epoch))

    return expanded


@hydra.main(
    version_base="1.3", config_name="hessian_experiment", config_path="../configs"
)
def main(cfg: DictConfig) -> Dict:
    OmegaConf.resolve(cfg)

    # Check if models should be loaded from file
    override_file = cfg.get("override_config", None)

    if override_file:
        logger.info(f"[CONFIG] Overriding config data from: {override_file}")
        model_directories, dataset, seed, epochs = load_experiment_override_from_yaml(
            override_file
        )

        # Update config with loaded models
        cfg.models = model_directories
        if dataset is not None:
            cfg.dataset = asdict(dataset)
        if seed is not None:
            cfg.seed = seed
        if epochs is not None:
            cfg.epochs = epochs

    config: ExperimentConfig = to_dataclass(ExperimentConfig, cfg)  # type: ignore

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info(f"{'=' * 70}")
    logger.info(f"Starting Hessian Analysis: {config.experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Dataset: {config.dataset.name.value}")
    logger.info(f"Models to analyze: {len(config.models)}")
    if config.epochs is not None:
        logger.info(f"Epochs to investigate: {config.epochs}")
        logger.info(f"Total analyses: {len(config.models) * len(config.epochs)}")
    else:
        logger.info("Analyzing final checkpoints only")
    logger.info(f"{'=' * 70}")

    # Load dataset
    dataset = DownloadableDataset.load(
        dataset=config.dataset.name,
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )
    dataset, _ = dataset.train_test_split(
        test_size=config.dataset.test_size, seed=config.seed
    )

    logger.info(f"Loaded dataset: {config.dataset.name.value}")

    # Expand models with epochs
    model_epoch_pairs = expand_models_with_epochs(config.models, config.epochs)

    # Validate that all required checkpoints exist
    if config.epochs is not None:
        for model_dir in config.models:
            if not check_saved_model(model_dir, config.epochs):
                logger.error(
                    f"Model at {model_dir} does not have all specified epochs saved. "
                    f"Required epochs: {config.epochs}"
                )
                raise FileNotFoundError(f"Missing epoch checkpoints in {model_dir}")

    # Run Hessian analysis
    logger.info(f"{'#' * 70}")
    logger.info("HESSIAN ANALYSIS")
    logger.info(f"{'#' * 70}")

    hessian_results = []
    for i, (model_dir, epoch) in enumerate(model_epoch_pairs, 1):
        epoch_str = f"epoch_{epoch}" if epoch is not None else "final"
        logger.info(
            f"[ANALYSIS {i}/{len(model_epoch_pairs)}] "
            f"{os.path.basename(model_dir)} ({epoch_str})"
        )

        result = analyze_single_model(
            model_directory=model_dir,
            dataset=dataset,
            hessian_config=config.hessian_analysis,
            seed=config.seed,
            epoch=epoch,
        )
        hessian_results.append(result)

        cleanup_memory(f"model_{i}")

    # Save results
    results_dir = config.hessian_analysis.results_output_dir
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f"{timestamp}.json")

    full_results = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "epochs_analyzed": config.epochs,
        "hessian_config": asdict(config.hessian_analysis),
        "results": hessian_results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2, default=json_safe)

    logger.info(f"{'=' * 70}")
    logger.info("Hessian Analysis Complete!")
    logger.info(f"Total analyses performed: {len(hessian_results)}")
    logger.info(f"Unique models: {len(config.models)}")
    if config.epochs:
        logger.info(f"Epochs per model: {len(config.epochs)}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'=' * 70}")

    return full_results


if __name__ == "__main__":
    main()
