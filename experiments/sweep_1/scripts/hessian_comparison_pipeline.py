"""
Hessian analysis pipeline for computing and comparing Hessian approximations.

This module handles:
- Loading trained models from manifest
- Collecting activations and gradients
- Computing EKFAC/KFAC data
- Running Hessian comparisons
- Saving analysis results
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional

import jax
from jax.random import PRNGKey

from src.config import HessianAnalysisConfig, HessianApproximator
from src.hessians.approximator.ekfac import EKFACApproximator
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import DataActivationsGradients, EKFACData, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_gradients
from src.utils.data.data import Dataset, DigitsDataset
from src.utils.loss import cross_entropy_loss
from src.utils.manifest import ModelEntry
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.train import load_model_checkpoint
from src.utils.utils import get_peak_bytes_in_use

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def block_tree(x, name: str):
    """Block until JAX operations complete."""
    from jax.tree_util import tree_map

    logger.info(f"[SYNC] Blocking on {name}")
    try:
        x = tree_map(lambda y: y.block_until_ready(), x)
    except Exception:
        logger.exception(f"[SYNC] Failure while blocking on {name}")
        raise
    logger.info(f"[SYNC] Completed {name}")
    return x


def cleanup_memory(stage: Optional[str] = None):
    """Clean up memory and report peak usage."""
    gc.collect()
    jax.clear_caches()
    if stage:
        logger.info(f"[MEMORY] after {stage}: peak_bytes={get_peak_bytes_in_use()}")
    else:
        logger.info(f"[MEMORY] peak_bytes={get_peak_bytes_in_use()}")


# -----------------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------------


def collect_gradients(
    model,
    params,
    train_inputs,
    train_targets,
    num_samples: int,
    seed: int,
) -> tuple:
    """Sample gradients for HVP/IHVP computations."""
    logger.info(f"[GRADIENTS] Sampling {num_samples} gradient samples")

    grads_1 = sample_gradients(
        model,
        params,
        train_inputs,
        train_targets,
        cross_entropy_loss,
        num_samples,
        PRNGKey(seed),
    )

    grads_2 = sample_gradients(
        model,
        params,
        train_inputs,
        train_targets,
        cross_entropy_loss,
        num_samples,
        PRNGKey(seed + 1),
    )

    cleanup_memory("gradient sampling")
    return grads_1, grads_2


def collect_activations_and_gradients(
    model,
    params,
    train_inputs,
    train_targets,
    collector_dir: str,
    seed: int,
    try_load: bool = True,
) -> DataActivationsGradients:
    """Collect activations and gradients with pseudo-targets."""
    logger.info(f"[COLLECTOR] Collecting activations/gradients: {collector_dir}")

    # Generate pseudo-targets
    pseudo_targets = generate_pseudo_targets(
        model=model,
        inputs=train_inputs,
        params=params,
        loss_fn=cross_entropy_loss,
        rng_key=PRNGKey(seed),
    )
    cleanup_memory("pseudo-target generation")

    # Create dataset with pseudo-targets
    collector_data = Dataset(train_inputs, train_targets).replace_targets(
        pseudo_targets
    )

    # Collect
    collector = CollectorActivationsGradients(
        model=model, params=params, loss_fn=cross_entropy_loss
    )

    collected_data = collector.collect(
        inputs=collector_data.inputs,
        targets=collector_data.targets,
        save_directory=collector_dir,
        try_load=try_load,
    )

    cleanup_memory("activation/gradient collection")
    return collected_data


def build_ekfac_data(
    collector_dir_1: str,
    collector_dir_2: str,
    ekfac_dir: str,
    model_name: str,
    model_dir: str,
    config: HessianAnalysisConfig,
) -> tuple[EKFACData, float]:
    """Build or load EKFAC data."""

    ekfac_path = Path(ekfac_dir)

    # Try to load existing EKFAC data
    if ekfac_path.exists():
        try:
            logger.info(f"[EKFAC] Loading existing data: {ekfac_dir}")
            ekfac_data, _ = EKFACApproximator.load_data(ekfac_dir)
            assert isinstance(ekfac_data, EKFACData)
            damping = ekfac_data.mean_eigenvalues_aggregated * 0.1
            logger.info(f"[EKFAC] Loaded (damping={damping:.6e})")
            return ekfac_data, damping
        except Exception as e:
            logger.warning(f"[EKFAC] Failed to load, will rebuild: {e}")

    # Build EKFAC data
    logger.info(f"[EKFAC] Building: {ekfac_dir}")
    ekfac = EKFACApproximator(collector_dir_1, collector_dir_2)

    ekfac.build(config=config, save_directory=ekfac_dir)

    # Load the built data
    ekfac_data, _ = EKFACApproximator.load_data(ekfac_dir)
    assert isinstance(ekfac_data, EKFACData)
    damping = ekfac_data.mean_eigenvalues_aggregated * 0.1

    logger.info(f"[EKFAC] Built (damping={damping:.6e})")
    cleanup_memory("EKFAC build")

    return ekfac_data, damping


# -----------------------------------------------------------------------------
# Hessian computations
# -----------------------------------------------------------------------------


def create_computer(
    approximator: HessianApproximator,
    ekfac_data: EKFACData,
    model_ctx: ModelContext,
    collected_data: DataActivationsGradients,
) -> HessianEstimator | HessianComputer:
    """Create a Hessian computer for the given approximation method."""
    if approximator == HessianApproximator.EXACT:
        return HessianComputer(model_ctx)
    elif approximator == HessianApproximator.KFAC:
        return KFACComputer(ekfac_data)
    elif approximator == HessianApproximator.EKFAC:
        return EKFACComputer(ekfac_data)
    elif approximator == HessianApproximator.GNH:
        return GNHComputer(model_ctx)
    elif approximator == HessianApproximator.FIM:
        return FIMComputer(collected_data)
    elif approximator == HessianApproximator.BLOCK_FIM:
        return FIMBlockComputer(collected_data)
    elif approximator == HessianApproximator.BLOCK_HESSIAN:
        return BlockHessianComputer(model_ctx)
    else:
        raise ValueError(f"Unknown approximator: {approximator}")


def compute_matrix_comparisons(
    reference_computer: HessianEstimator | HessianComputer,
    comparison_computers: dict[str, HessianEstimator],
    damping: float,
    reference_name: str,
) -> dict:
    """Compute matrix-based Hessian comparisons."""
    logger.info(f"[HESSIAN] Computing {reference_name} Hessian matrix")

    if reference_name == "exact":
        assert isinstance(reference_computer, HessianComputer), (
            "Exact reference must be HessianComputer"
        )
        ref_hessian = block_tree(
            reference_computer.compute_hessian(damping=damping),
            f"{reference_name} Hessian",
        )
    else:
        assert isinstance(reference_computer, HessianEstimator), (
            "Approximate reference must be instance of HessianEstimator"
        )
        ref_hessian = block_tree(
            reference_computer.estimate_hessian(damping=damping),
            f"{reference_name} Hessian",
        )

    results = {}
    for approx_name, computer in comparison_computers.items():
        logger.info(f"[HESSIAN] {reference_name} vs {approx_name} (matrix)")

        for metric in FullMatrixMetric:
            metric_value = computer.compare_full_hessian_estimates(
                ref_hessian, damping, metric
            )
            results.setdefault(metric.value, {})[approx_name] = float(metric_value)

    del ref_hessian
    cleanup_memory(f"{reference_name} matrix comparisons")

    return results


def compute_vector_comparisons(
    reference_computer: HessianEstimator,
    comparison_computers: dict[str, HessianEstimator],
    grads_1,
    grads_2,
    damping: float,
    reference_name: str,
) -> tuple[dict, dict]:
    """Compute vector-based (HVP/IHVP) Hessian comparisons."""
    logger.info(f"[HESSIAN] Computing {reference_name} HVP/IHVP")

    ref_hvp = block_tree(
        reference_computer.estimate_hvp(grads_1, damping=damping),
        f"{reference_name} HVP",
    )
    ref_ihvp = block_tree(
        reference_computer.estimate_ihvp(grads_1, damping=damping),
        f"{reference_name} IHVP",
    )

    hvp_results = {}
    ihvp_results = {}

    for approx_name, computer in comparison_computers.items():
        logger.info(f"[HESSIAN] {reference_name} vs {approx_name} (vector)")

        hvp = block_tree(computer.estimate_hvp(grads_1, damping), f"{approx_name} HVP")
        ihvp = block_tree(
            computer.estimate_ihvp(grads_1, damping), f"{approx_name} IHVP"
        )

        for metric in VectorMetric.all_metrics():
            hvp_value = metric.compute(ref_hvp, hvp, grads_2)
            ihvp_value = metric.compute(ref_ihvp, ihvp, grads_2)

            hvp_results.setdefault(metric.name, {})[approx_name] = float(hvp_value)
            ihvp_results.setdefault(metric.name, {})[approx_name] = float(ihvp_value)

    del ref_hvp, ref_ihvp
    cleanup_memory(f"{reference_name} vector comparisons")

    return hvp_results, ihvp_results


def analyze_single_model(
    model_entry: ModelEntry,
    dataset: Dataset,
    hessian_config: HessianAnalysisConfig,
) -> dict:
    """Run complete Hessian analysis for a single model."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"[HESSIAN] Analyzing: {model_entry.model_name}")
    logger.info(f"{'=' * 80}")

    # Load model
    from src.utils.models.mlp import MLP
    from src.utils.models.mlp_swiglu import MLPSwiGLU

    if model_entry.model_type == "mlp":
        model = MLP(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=model_entry.hidden_layers,
            seed=model_entry.seed,
        )
    elif model_entry.model_type == "mlpswiglu":
        model = MLPSwiGLU(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=model_entry.hidden_layers,
            activation="swiglu",
            seed=model_entry.seed,
        )
    else:
        raise ValueError(f"Unknown model type: {model_entry.model_type}")

    params, _, _ = load_model_checkpoint(model_entry.model_dir, model=model)

    # Collect gradients
    grads_1, grads_2 = collect_gradients(
        model,
        params,
        dataset.inputs,
        dataset.targets,
        num_samples=hessian_config.collector_config.num_gradient_samples,
        seed=hessian_config.analysis_seed,
    )

    # Collect activations/gradients (run 1)
    collector_dir_1 = f"{hessian_config.collector_config.collector_output_dir}/{model_entry.model_name}/run1/"
    collected_data_1 = collect_activations_and_gradients(
        model,
        params,
        dataset.inputs,
        dataset.targets,
        collector_dir_1,
        seed=hessian_config.analysis_seed,
        try_load=hessian_config.collector_config.try_load_cached,
    )

    # Collect activations/gradients (run 2)
    collector_dir_2 = f"{hessian_config.collector_config.collector_output_dir}/{model_entry.model_name}/run2/"
    collect_activations_and_gradients(
        model,
        params,
        dataset.inputs,
        dataset.targets,
        collector_dir_2,
        seed=hessian_config.analysis_seed + 1,
        try_load=hessian_config.collector_config.try_load_cached,
    )

    # Build EKFAC data
    ekfac_dir = f"{hessian_config.hessian_output_dir}/{model_entry.model_name}/"
    ekfac_data, auto_damping = build_ekfac_data(
        collector_dir_1,
        collector_dir_2,
        ekfac_dir,
        model_entry.model_name,
        model_entry.model_dir,
    )

    # Get damping value
    damping = hessian_config.damping_config.get_damping_value()
    logger.info(f"[DAMPING] Using damping={damping:.6e}")

    # Create model context
    model_ctx = ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=cross_entropy_loss,
    )

    # Prepare results
    results = {
        "model_name": model_entry.model_name,
        "damping": damping,
        "matrix_comparisons": {},
        "hvp_comparisons": {},
        "ihvp_comparisons": {},
    }

    comp_config = hessian_config.computation_config

    # Create all computers needed
    all_computers = {}
    for approx in comp_config.approximators:
        if approx == HessianApproximator.EXACT:
            continue  # Handled separately
        all_computers[approx.value] = create_computer(
            approx, ekfac_data, model_ctx, collected_data_1
        )

    # Run comparisons based on configuration
    for reference in comp_config.comparison_references:
        ref_name = reference.value

        # Skip exact if model too large
        if reference == ComparisonReference.EXACT:
            if not comp_config.should_compute_exact(model_entry.num_params):
                logger.info(
                    f"[HESSIAN] Skipping exact Hessian (model too large: {model_entry.num_params} params)"
                )
                continue
            ref_computer = HessianComputer(model_ctx)
        else:
            ref_computer = GNHComputer(model_ctx)

        # Determine which approximators to compare against this reference
        if reference == ComparisonReference.EXACT:
            # Compare all approximators against exact
            comparison_computers = all_computers
        else:
            # Compare only K-FAC variants against GNH
            comparison_computers = {
                k: v
                for k, v in all_computers.items()
                if k in ["kfac", "ekfac", "fim", "block_fim"]
            }

        # Matrix comparisons
        if ComputationType.MATRIX in comp_config.computation_types:
            matrix_results = compute_matrix_comparisons(
                ref_computer, comparison_computers, damping, ref_name
            )
            for metric, values in matrix_results.items():
                results["matrix_comparisons"].setdefault(metric, {})[ref_name] = values

        # Vector comparisons
        if (
            ComputationType.HVP in comp_config.computation_types
            or ComputationType.IHVP in comp_config.computation_types
        ):
            hvp_results, ihvp_results = compute_vector_comparisons(
                ref_computer, comparison_computers, grads_1, grads_2, damping, ref_name
            )

            if ComputationType.HVP in comp_config.computation_types:
                for metric, values in hvp_results.items():
                    results["hvp_comparisons"].setdefault(metric, {})[ref_name] = values

            if ComputationType.IHVP in comp_config.computation_types:
                for metric, values in ihvp_results.items():
                    results["ihvp_comparisons"].setdefault(metric, {})[ref_name] = (
                        values
                    )

        del ref_computer

    logger.info(f"[HESSIAN] Completed analysis for {model_entry.model_name}")
    return results


# -----------------------------------------------------------------------------
# Main Hessian pipeline
# -----------------------------------------------------------------------------


def run_hessian_pipeline(config: ExperimentConfig) -> dict:
    """
    Execute the full Hessian analysis pipeline.

    Returns:
        Dictionary containing all analysis results
    """
    hessian_config = config.get_hessian_config()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    logger.info(f"Starting Hessian analysis pipeline: {config.experiment_name}")

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    logger.info(f"Loading dataset from {config.dataset.path}")
    dataset: DigitsDataset = DigitsDataset.load(
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )

    train_ds, _ = dataset.train_test_split(
        test_size=config.dataset.test_size,
        seed=config.dataset.split_seed,
    )

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    if hessian_config.input_manifest_path:
        logger.info(f"Loading manifest: {hessian_config.input_manifest_path}")
        manifest = load_manifest(hessian_config.input_manifest_path)

        # Filter models
        models_to_analyze = manifest.filter_models(
            pattern=hessian_config.model_filter,
            min_accuracy=config.get_training_config().min_accuracy_threshold
            if config.training
            else None,
        )

        logger.info(
            f"Selected {len(models_to_analyze)} models for analysis "
            f"(from {len(manifest.models)} total)"
        )
    else:
        # Single model analysis
        raise NotImplementedError("Single model analysis not yet implemented")

    # -------------------------------------------------------------------------
    # Analyze models
    # -------------------------------------------------------------------------
    all_results = {
        "metadata": {
            "experiment_name": config.experiment_name,
            "timestamp": timestamp,
            "dataset_path": config.dataset.path,
            "analysis_seed": hessian_config.analysis_seed,
            "damping_config": {
                "strategy": hessian_config.damping_config.strategy.value,
                "fixed_value": hessian_config.damping_config.fixed_value,
                "auto_multiplier": hessian_config.damping_config.auto_multiplier,
            },
            "computation_config": {
                "approximators": [
                    a.value for a in hessian_config.computation_config.approximators
                ],
                "references": [
                    r.value
                    for r in hessian_config.computation_config.comparison_references
                ],
                "computation_types": [
                    t.value for t in hessian_config.computation_config.computation_types
                ],
            },
        },
        "results": [],
    }

    for model_entry in models_to_analyze:
        try:
            result = analyze_single_model(model_entry, train_ds, hessian_config)
            all_results["results"].append(result)
        except Exception as e:
            logger.exception(f"Failed to analyze {model_entry.model_name}: {e}")
            continue

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    results_file = (
        f"{hessian_config.results_output_dir}/"
        f"hessian_analysis_{config.experiment_name}_{timestamp}.json"
    )
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Saved Hessian analysis results to {results_file}")

    return all_results
