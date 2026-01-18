"""
Standalone Hessian analysis script with Top K aggregation.

Loads pre-trained models and performs Hessian approximation comparisons.
Groups models by architecture and computes statistics (mean, std) across top K models.

Usage:
    # Basic run with config specified in ../configs/hessian_analysis.yaml 
    # (must include the model directories to analyze)
    python -m experiments.hessian_analysis \
        --config-name=hessian_analysis \
        --config-path=../configs
    
    # Run with specific models from training output (e.g., top 5 models)
    python -m experiments.hessian_analysis \
        --config-name=hessian_analysis \
        --config-path=../configs \
        +override_config=path/to/top5_models.yaml
    
    # Override individual config parameters
    python -m experiments.hessian_analysis \
        --config-name=hessian_analysis \
        --config-path=../configs \
        hessian_analysis.vector_config.num_samples=500 \
        +override_config=experiments/outputs/models/top5_models_20240115.yaml

Notes:
    - The override_config parameter expects a YAML file with model paths, dataset config, and seed
    - Models are automatically grouped by architecture (architecture + hidden_dim)
    - For each group, statistics (mean, std) are computed across all models in that group
    - Individual model results are saved alongside aggregated statistics
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Tuple

import hydra
import jax.numpy as jnp
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
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_vectors
from src.utils.data.data import Dataset, DownloadableDataset
from src.utils.loss import get_loss
from src.utils.train import load_model_checkpoint

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="hessian_experiment", node=ExperimentConfig)


def get_model_group_key(model_config: ModelConfig) -> str:
    """Generate a key to group models by architecture and structure."""
    return f"{model_config.architecture.value}_{str(model_config.hidden_dim)}"


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
) -> Dict:
    """Compute all Hessian comparisons specified in the config."""
    results = {
        "matrix_comparisons": {},
        "hvp_comparisons": {},
        "ihvp_comparisons": {},
    }

    # Use EKFAC as base for damping selection
    ekfac_computer = EKFACComputer(compute_context=collector_data)
    ekfac_computer.build(base_directory=model_directory)
    damping = EKFACComputer.get_damping(
        ekfac_data=ekfac_computer.precomputed_data,
        damping_strategy=hessian_config.computation_config.damping_strategy,
        factor=hessian_config.computation_config.damping,
    )
    logger.info(f"[HESSIAN] Using damping: {damping:.6f}")
    results.setdefault("damping", damping)

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
                    reference_computer.compute_ihvp(grads_1, damping=damping),
                    f"{reference_approx.value}_ihvp",
                )
            elif isinstance(reference_computer, HessianEstimator):
                ref_ihvp = block_tree(
                    reference_computer.estimate_ihvp(grads_1, damping=damping),
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
                    approx_computer.estimate_ihvp(grads_1, damping=damping),
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
    assert model_directory is not None, "Model directory must be set"

    params, model, model_config, metadata = load_model_checkpoint(model_directory)

    logger.info(f"{'=' * 70}")
    logger.info(f"[HESSIAN] Analyzing: {model_config.get_model_display_name()}")
    logger.info(f"Model directory: {model_directory}")
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

    grads_1, grads_2, collector_data, model_ctx = collect_data(
        model=model,
        params=params,
        model_config=model_config,
        dataset=dataset,
        collector_dirs=(
            os.path.join(model_config.directory, "collector", "run_1"),
            os.path.join(model_config.directory, "collector", "run_2"),
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
        model_config.directory,
    )

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_config.directory,
        "model_config": asdict(model_config),
        "num_parameters": model.num_params,
        "metadata": metadata or {},
        "hessian_analysis": hessian_results,
        "group_key": get_model_group_key(model_config),
    }


def aggregate_results_by_group(individual_results: List[Dict]) -> Dict:
    """
    Aggregate Hessian analysis results by architecture group.

    Computes mean and std for each metric across models in the same group.

    Returns:
        Dictionary with aggregated statistics per group
    """
    groups = defaultdict(list)

    # Group results by architecture
    for result in individual_results:
        group_key = result["group_key"]
        groups[group_key].append(result)

    aggregated = {}

    for group_key, group_results in groups.items():
        logger.info(
            f"[AGGREGATION] Processing group: {group_key} ({len(group_results)} models)"
        )

        # Initialize structure for aggregated results
        agg_result = {
            "group_key": group_key,
            "num_models": len(group_results),
            "models": [r["model_name"] for r in group_results],
            "matrix_comparisons": {},
            "hvp_comparisons": {},
            "ihvp_comparisons": {},
        }

        # Collect all metric values across models
        metric_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for result in group_results:
            hessian = result["hessian_analysis"]

            # Matrix comparisons
            for metric_name, ref_dict in hessian.get("matrix_comparisons", {}).items():
                for ref_method, approx_dict in ref_dict.items():
                    for approx_method, value in approx_dict.items():
                        metric_values["matrix_comparisons"][metric_name][
                            f"{ref_method}_vs_{approx_method}"
                        ].append(value)

            # HVP comparisons
            for metric_name, ref_dict in hessian.get("hvp_comparisons", {}).items():
                for ref_method, approx_dict in ref_dict.items():
                    for approx_method, value in approx_dict.items():
                        metric_values["hvp_comparisons"][metric_name][
                            f"{ref_method}_vs_{approx_method}"
                        ].append(value)

            # IHVP comparisons
            for metric_name, ref_dict in hessian.get("ihvp_comparisons", {}).items():
                for ref_method, approx_dict in ref_dict.items():
                    for approx_method, value in approx_dict.items():
                        metric_values["ihvp_comparisons"][metric_name][
                            f"{ref_method}_vs_{approx_method}"
                        ].append(value)

        # Compute statistics
        for comp_type in ["matrix_comparisons", "hvp_comparisons", "ihvp_comparisons"]:
            for metric_name, comparisons in metric_values[comp_type].items():
                agg_result[comp_type][metric_name] = {}

                for comparison_key, values in comparisons.items():
                    values_array = jnp.array(values)
                    agg_result[comp_type][metric_name][comparison_key] = {
                        "mean": float(jnp.mean(values_array)),
                        "std": float(jnp.std(values_array)),
                        "min": float(jnp.min(values_array)),
                        "max": float(jnp.max(values_array)),
                        "count": len(values),
                        "values": [
                            float(v) for v in values
                        ],  # Keep individual values for reference
                    }

        aggregated[group_key] = agg_result

    return aggregated


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
        model_directories, dataset_config, seed = load_experiment_override_from_yaml(
            override_file
        )

        # Update config with loaded models
        cfg.models = model_directories
        if dataset_config is not None:
            cfg.dataset = asdict(dataset_config)
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
    dataset, _ = dataset.train_test_split(
        test_size=config.dataset.test_size, seed=config.seed
    )

    logger.info(f"Loaded dataset: {config.dataset.name.value}")

    # Run Hessian analysis
    logger.info(f"{'#' * 70}")
    logger.info("HESSIAN ANALYSIS - INDIVIDUAL MODELS")
    logger.info(f"{'#' * 70}")

    individual_results = []
    for i, model_config in enumerate(config.models, 1):
        logger.info(f"[MODEL {i}/{len(config.models)}]")

        result = analyze_single_model(
            model_config,
            dataset,
            config.hessian_analysis,
            config.seed,
        )
        individual_results.append(result)

        cleanup_memory(f"model_{i}")

    # Aggregate results by group
    logger.info(f"{'#' * 70}")
    logger.info("AGGREGATING RESULTS BY GROUP")
    logger.info(f"{'#' * 70}")

    aggregated_results = aggregate_results_by_group(individual_results)

    # Log summary statistics
    logger.info(f"{'=' * 70}")
    logger.info("AGGREGATION SUMMARY")
    logger.info(f"{'=' * 70}")
    for group_key, agg_data in aggregated_results.items():
        logger.info(f"Group: {group_key}")
        logger.info(f"  Models: {agg_data['num_models']}")
        logger.info(f"  Names: {', '.join(agg_data['models'])}")

    # Save results
    results_dir = config.hessian_analysis.results_output_dir
    os.makedirs(results_dir, exist_ok=True)

    # Save individual results
    individual_output_file = os.path.join(
        results_dir,
        f"{timestamp}_individual.json",
    )

    individual_full_results = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "hessian_config": asdict(config.hessian_analysis),
        "results": individual_results,
    }

    with open(individual_output_file, "w") as f:
        json.dump(individual_full_results, f, indent=2, default=json_safe)

    logger.info(f"Individual results saved to: {individual_output_file}")

    # Save aggregated results
    aggregated_output_file = os.path.join(
        results_dir,
        f"{timestamp}_aggregated.json",
    )

    aggregated_full_results = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "hessian_config": asdict(config.hessian_analysis),
        "num_groups": len(aggregated_results),
        "total_models_analyzed": len(individual_results),
        "aggregated_results": aggregated_results,
    }

    with open(aggregated_output_file, "w") as f:
        json.dump(aggregated_full_results, f, indent=2, default=json_safe)

    logger.info(f"Aggregated results saved to: {aggregated_output_file}")

    logger.info(f"{'=' * 70}")
    logger.info("Hessian Analysis Complete!")
    logger.info(f"Total models analyzed: {len(individual_results)}")
    logger.info(f"Architecture groups: {len(aggregated_results)}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  Individual: {individual_output_file}")
    logger.info(f"  Aggregated: {aggregated_output_file}")
    logger.info(f"{'=' * 70}")

    return {
        "individual": individual_full_results,
        "aggregated": aggregated_full_results,
    }


if __name__ == "__main__":
    main()
