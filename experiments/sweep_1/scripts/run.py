import gc
import json
import logging
import os
import time
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Dict, List, Union, get_args, get_origin, get_type_hints

import hydra
import jax
from hydra.core.config_store import ConfigStore
from jax.random import PRNGKey
from jax.tree_util import tree_map
from omegaconf import DictConfig, OmegaConf

from src.config import (
    ComputationType,
    ExperimentConfig,
    HessianAnalysisConfig,
    HessianApproximator,
    ModelConfig,
    VectorAnalysisConfig,
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
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import (
    check_saved_model,
    evaluate_loss_and_classification_accuracy,
    load_model_checkpoint,
    save_model_checkpoint,
    train_model,
)
from src.utils.utils import get_peak_bytes_in_use, hash_data

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)


def to_dataclass(cls, config):
    """
    Recursively convert OmegaConf DictConfig to dataclass instances.
    Handles nested dataclasses, lists, enums, and optional types.
    """
    # If it's already the right type, return it
    if isinstance(config, cls):
        return config

    # Convert DictConfig to dict
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Handle None
    if config is None:
        return None

    # If not a dataclass, return as-is
    if not is_dataclass(cls):
        return config

    # Must be a dict at this point
    if not isinstance(config, dict):
        return config

    kwargs = {}

    # Use get_type_hints to properly resolve forward references and string annotations
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        # Fallback to field.type if get_type_hints fails
        type_hints = {f.name: f.type for f in fields(cls)}

    for field_info in fields(cls):
        field_name = field_info.name

        # Skip if field not in config
        if field_name not in config:
            continue

        value = config[field_name]

        # Handle None values
        if value is None:
            kwargs[field_name] = None
            continue

        # Get type hint for this field
        field_type = type_hints.get(field_name, field_info.type)

        # Parse the type
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Union types (including Optional[T] which is Union[T, None])
        if origin is Union:
            # Filter out NoneType to get the actual type
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                field_type = non_none_types[0]
                origin = get_origin(field_type)
                args = get_args(field_type)

        # Handle List[T]
        if origin is list:
            if args:
                item_type = args[0]
                if is_dataclass(item_type):
                    kwargs[field_name] = [
                        to_dataclass(item_type, item) for item in value
                    ]
                else:
                    # Try to convert items (handles enums in lists)
                    kwargs[field_name] = [
                        _convert_value(item_type, item) for item in value
                    ]
            else:
                kwargs[field_name] = value

        # Handle nested dataclass
        elif is_dataclass(field_type):
            kwargs[field_name] = to_dataclass(field_type, value)

        # Handle everything else (enums, primitives, etc.)
        else:
            kwargs[field_name] = _convert_value(field_type, value)

    return cls(**kwargs)  # type: ignore


def _convert_value(target_type, value):
    """Convert a single value to the target type."""
    # Already correct type
    if isinstance(value, target_type):
        return value

    # Handle Enum conversion
    try:
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            if isinstance(value, str):
                return target_type(value)
            return value
    except TypeError:
        pass

    # Return as-is for primitives and other types
    return value


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def block_tree(x, name: str):
    """Block until all arrays in a tree are ready."""
    logger.info(f"[SYNC] Blocking on {name}")
    try:
        x = tree_map(lambda y: y.block_until_ready(), x)
    except Exception:
        logger.exception(f"[SYNC] Failure while blocking on {name}")
        raise
    logger.info(f"[SYNC] Completed {name}")
    return x


def cleanup_memory(stage: str | None = None):
    """Force garbage collection and clear JAX caches."""
    gc.collect()
    jax.clear_caches()
    msg = f"[MEMORY] peak_bytes={get_peak_bytes_in_use()}"
    if stage:
        msg = f"[MEMORY] after {stage}: {msg}"
    logger.info(msg)


def generate_model_directory(
    model_config: ModelConfig,
    base_dir: str,
) -> str:
    """Generate a unique directory path for a model based on its configuration."""
    # Create a hash of the full configuration
    config_hash = hash_data(
        {
            "architecture": model_config.architecture.value,
            "hidden_dim": model_config.hidden_dim,
            "loss": model_config.loss.value,
            "init_seed": model_config.init_seed,
            "learning_rate": model_config.training.learning_rate,
            "weight_decay": model_config.training.weight_decay,
            "optimizer": model_config.training.optimizer.value,
            "epochs": model_config.training.epochs,
            "batch_size": model_config.training.batch_size,
        },
        length=12,
    )

    model_display_name = model_config.get_model_display_name()
    return os.path.join(base_dir, f"{model_display_name}_{config_hash}")


# -----------------------------------------------------------------------------
# Training Phase
# -----------------------------------------------------------------------------


def train_single_model(
    model_config: ModelConfig,
    dataset: Dataset,
    seed: int,
) -> Dict:
    """
    Train a single model and return its results.

    Returns:
        Dict with keys: model_config, training_config, model_directory, val_loss, val_accuracy
    """
    train_ds, val_ds = dataset.train_test_split(test_size=0.1, seed=seed)
    train_inputs, train_targets = train_ds.inputs, train_ds.targets
    val_inputs, val_targets = val_ds.inputs, val_ds.targets

    # Create model
    model = ModelRegistry.get_model(
        model_config=model_config,
        input_dim=dataset.input_dim(),
        output_dim=dataset.output_dim(),
        seed=seed,
    )
    logger.info(f"Model has {model.num_params} parameters")

    model_dir = model_config.directory
    assert model_dir is not None, "Model directory must be set"

    # Load or train
    if model_config.skip_existing and check_saved_model(model_dir, model=model):
        params, _, _ = load_model_checkpoint(model_dir, model=model)
        logger.info(f"[LOAD] {model_config.get_model_display_name()} from {model_dir}")
    else:
        logger.info(f"[TRAIN] {model_config.get_model_display_name()}")
        model, params, _ = train_model(
            model=model,
            dataloader=Dataset(train_inputs, train_targets).get_dataloader(
                batch_size=model_config.training.batch_size, seed=seed
            ),
            loss_fn=get_loss(model_config.loss),
            optimizer=optimizer(
                optimizer_enum=model_config.training.optimizer,
                lr=model_config.training.learning_rate,
                weight_decay=model_config.training.weight_decay,
            ),
            epochs=model_config.training.epochs,
        )

    # Evaluate
    val_loss, val_acc = evaluate_loss_and_classification_accuracy(
        model, params, val_inputs, val_targets, get_loss(model_config.loss)
    )

    logger.info(
        f"[EVAL] {model_config.get_model_display_name()}: "
        f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}"
    )

    # Save checkpoint with full metadata
    save_model_checkpoint(
        model=model,
        params=params,
        directory=model_dir,
        metadata={
            "model_config": asdict(model_config),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "num_parameters": model.num_params,
        },
    )

    cleanup_memory(f"train_{model_config.get_model_display_name()}")

    return {
        "model_config": model_config,
        "model_directory": model_dir,
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "num_parameters": model.num_params,
    }


def train_all_models(
    model_configs: List[ModelConfig],
    dataset: Dataset,
    seed: int,
) -> List[Dict]:
    """Train all specified models."""
    logger.info(f"Training {len(model_configs)} models")

    results = []
    for i, model_config in enumerate(model_configs, 1):
        logger.info(f"{'=' * 70}")
        logger.info(
            f"Model {i}/{len(model_configs)}: {model_config.get_model_display_name()}"
        )
        logger.info(
            f"  lr={model_config.training.learning_rate:.2e}, "
            f"wd={model_config.training.weight_decay:.2e}"
        )
        logger.info(f"{'=' * 70}")

        result = train_single_model(model_config, dataset, seed)
        results.append(result)

    return results


# -----------------------------------------------------------------------------
# Hessian Analysis Phase
# -----------------------------------------------------------------------------


def setup_hessian_paths(
    model_config: ModelConfig,
    dataset_name: str,
    hessian_config: HessianAnalysisConfig,
) -> tuple:
    """Setup directory paths for Hessian analysis."""
    model_base_name = model_config.get_model_base_name()

    collector_dirs_ekfac = [
        os.path.join(
            hessian_config.base_collector_dir, dataset_name, model_base_name, "run1"
        ),
        os.path.join(
            hessian_config.base_collector_dir, dataset_name, model_base_name, "run2"
        ),
    ]

    collector_dir_single = os.path.join(
        hessian_config.base_collector_dir, dataset_name, model_base_name, "run1"
    )

    ekfac_dir = os.path.join(
        hessian_config.base_ekfac_dir, dataset_name, model_base_name
    )

    return collector_dirs_ekfac, collector_dir_single, ekfac_dir


def prepare_hessian_data(
    model,
    params,
    model_config: ModelConfig,
    dataset: Dataset,
    collector_dirs_ekfac: List[str],
    collector_dir_single: str,
    ekfac_dir: str,
    vector_config: VectorAnalysisConfig,
    seed: int,
):
    """Prepare all data needed for Hessian analysis."""
    train_ds, _ = dataset.train_test_split(test_size=0.1, seed=seed)
    train_inputs, train_targets = train_ds.inputs, train_ds.targets
    loss_fn = get_loss(model_config.loss)

    # Sample gradient vectors
    logger.info("[HESSIAN] Sampling gradient vectors")
    grads = sample_vectors(
        vector_config=vector_config,
        model=model,
        params=params,
        inputs=train_inputs,
        targets=train_targets,
        loss_fn=loss_fn,
    )

    # Split gradients for HVP/IHVP
    half = len(grads) // 2
    grads_1, grads_2 = grads[:half], grads[half:]
    del grads
    cleanup_memory("gradient_sampling")

    # Collect for EKFAC (2 runs)
    logger.info("[HESSIAN] Collecting for EKFAC")
    for run_idx, collector_dir in enumerate(collector_dirs_ekfac):
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

    # Collect for single run (FIM/BLOCK_FIM)
    logger.info("[HESSIAN] Collecting for single-run approximators")
    pseudo_targets_single = generate_pseudo_targets(
        model=model,
        inputs=train_inputs,
        params=params,
        loss_fn=loss_fn,
        rng_key=PRNGKey(seed),
    )
    collector_data_single = Dataset(train_inputs, train_targets).replace_targets(
        pseudo_targets_single
    )
    collector_single = CollectorActivationsGradients(
        model=model, params=params, loss_fn=loss_fn
    )
    collector_single.collect(
        inputs=collector_data_single.inputs,
        targets=collector_data_single.targets,
        save_directory=collector_dir_single,
        try_load=True,
    )
    cleanup_memory("single_collection")

    # Build/Load EKFAC
    logger.info("[HESSIAN] Building EKFAC approximation")
    ekfac = EKFACApproximator(collector_dirs_ekfac[0], collector_dirs_ekfac[1])

    if not ekfac.data_exists(ekfac_dir):
        ekfac.build(config=model_config, save_directory=ekfac_dir)

    ekfac_data, _ = EKFACApproximator.load_data(ekfac_dir)
    assert isinstance(ekfac_data, EKFACData)
    cleanup_memory("ekfac_build")

    # Load collector data
    collector_single_data = CollectorActivationsGradients.load(collector_dir_single)

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
                        "Matrix comparisons can only be done on an estimator as calling computer."
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
                    "HVP comparisons can only be done between estimators."
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
                    "IHVP comparisons can only be done between estimators."
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
    model_info: Dict,
    dataset: Dataset,
    hessian_config: HessianAnalysisConfig,
    dataset_name: str,
    seed: int,
) -> Dict:
    """Run Hessian analysis on a single model."""
    model_config = model_info["model_config"]

    logger.info(f"{'=' * 70}")
    logger.info(f"[HESSIAN] Analyzing: {model_config.get_model_display_name()}")
    # Log some model information
    logger.info(f" Num parameters: {model_info.get('num_parameters', 'N/A')}")
    logger.info(f"  Val loss: {model_info['val_loss']:.6f}")
    logger.info(f"  Val accuracy: {model_info['val_accuracy']:.4f}")
    logger.info(f"{'=' * 70}")

    # Load model
    model = ModelRegistry.get_model(
        model_config=model_config,
        input_dim=dataset.input_dim(),
        output_dim=dataset.output_dim(),
        seed=seed,
    )
    params, _, _ = load_model_checkpoint(model_config.directory, model=model)

    # Setup paths
    collector_dirs_ekfac, collector_dir_single, ekfac_dir = setup_hessian_paths(
        model_config, dataset_name, hessian_config
    )

    # Prepare data
    grads_1, grads_2, ekfac_data, collector_data, model_ctx = prepare_hessian_data(
        model,
        params,
        model_config,
        dataset,
        collector_dirs_ekfac,
        collector_dir_single,
        ekfac_dir,
        hessian_config.vector_config,
        seed,
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
        "val_loss": model_info["val_loss"],
        "val_accuracy": model_info["val_accuracy"],
        "model_config": asdict(model_config),
        "hessian_analysis": hessian_results,
    }


def analyze_all_models(
    training_results: List[Dict],
    dataset: Dataset,
    hessian_config: HessianAnalysisConfig,
    dataset_name: str,
    seed: int,
) -> List[Dict]:
    """Run Hessian analysis on all models that meet the accuracy threshold."""
    # Filter by accuracy threshold
    qualified_models = [
        result
        for result in training_results
        if result["val_accuracy"] >= hessian_config.accuracy_threshold
    ]

    logger.info(
        f"[HESSIAN] Analyzing {len(qualified_models)}/{len(training_results)} models "
        f"(accuracy >= {hessian_config.accuracy_threshold})"
    )

    results = []
    for model_info in qualified_models:
        result = analyze_single_model(
            model_info, dataset, hessian_config, dataset_name, seed
        )
        results.append(result)

    return results


# -----------------------------------------------------------------------------
# Main Experiment
# -----------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_name="experiment", config_path="../configs")
def main(cfg: DictConfig) -> Dict:
    OmegaConf.resolve(cfg)
    config: ExperimentConfig = to_dataclass(ExperimentConfig, cfg)  # type: ignore

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info(f"{'=' * 70}")
    logger.info(f"Starting Experiment: {config.experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Dataset: {config.dataset.name.value}")
    logger.info(f"Models to train: {len(config.models)}")
    logger.info(f"{'=' * 70}")

    # Ensure model directories are set
    models_base_dir = config.get_models_base_dir()
    for model_config in config.models:
        if model_config.directory is None:
            model_config.directory = generate_model_directory(
                model_config, models_base_dir
            )

    # Load dataset
    dataset = DownloadableDataset.load(
        dataset=config.dataset.name,
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )
    logger.info(f"Loaded dataset: {config.dataset.name.value}")

    # Phase 1: Training
    logger.info(f"{'#' * 70}")
    logger.info("PHASE 1: MODEL TRAINING")
    logger.info(f"{'#' * 70}")

    training_results = train_all_models(config.models, dataset, config.seed)

    logger.info(f"[SUMMARY] Trained {len(training_results)} models")
    for result in training_results:
        logger.info(
            f"  {result['model_config'].get_model_display_name()}: "
            f"val_loss={result['val_loss']:.6f}, val_acc={result['val_accuracy']:.4f}"
        )

    # Phase 2: Hessian Analysis (optional)
    all_results = training_results

    if config.run_hessian_analysis:
        logger.info(f"{'#' * 70}")
        logger.info("PHASE 2: HESSIAN ANALYSIS")
        logger.info(f"{'#' * 70}")

        hessian_results = analyze_all_models(
            training_results,
            dataset,
            config.hessian_analysis,
            config.dataset.name.value,
            config.seed,
        )
        all_results = hessian_results

    # Save results
    results_dir = config.get_results_dir()
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(
        results_dir,
        f"{timestamp}.json",
    )

    full_results = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "config": asdict(config),
        "experiments": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"{'=' * 70}")
    logger.info("Experiment Complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'=' * 70}")

    return full_results


if __name__ == "__main__":
    main()
