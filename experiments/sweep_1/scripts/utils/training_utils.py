"""
Training pipeline for model training and hyperparameter search.

This module handles:
- Model instantiation
- Hyperparameter grid search
- Model training and evaluation
- Checkpoint saving and loading
- Manifest management
"""

import logging
import time
from typing import Any, Callable, List, Tuple, cast

from src.config import (
    ExperimentConfig,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
)
from src.utils.data.data import Dataset, DigitsDataset
from src.utils.loss import cross_entropy_loss
from src.utils.manifest import (
    ModelEntry,
    TrainingManifest,
    create_empty_manifest,
    save_manifest,
)
from src.utils.models.mlp import MLP
from src.utils.models.mlp_swiglu import MLPSwiGLU
from src.utils.optimizers import optimizer
from src.utils.train import (
    check_saved_model,
    evaluate_loss_and_classification_accuracy,
    load_model_checkpoint,
    save_model_checkpoint,
    train_model,
)
from src.utils.utils import hash_data

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Model instantiation
# -----------------------------------------------------------------------------


def split_dim_for_swiglu(x: int) -> tuple[int, int, int]:
    """Split dimension for SwiGLU architecture."""
    base = x // 3
    return (2 * base, 2 * base, 2 * base)


def create_model(
    architecture: ModelArchitecture,
    layer_config: ModelConfig,
    input_dim: int,
    output_dim: int,
    seed: int,
):
    """Create a model instance based on architecture and layer configuration."""
    if architecture == ModelArchitecture.MLP:
        assert isinstance(layer_config.hidden_dims, list) and all(
            isinstance(x, int) for x in layer_config.hidden_dims
        ), "Hidden dims must be a list of integers for MLP architecture"
        hidden_dims = cast(List[int], layer_config.hidden_dims)
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dims,
            seed=seed,
        )
    elif architecture == ModelArchitecture.MLP_SWIGLU:
        is_list_of_tuples = isinstance(layer_config.hidden_dims, list) and all(
            isinstance(x, tuple) and len(x) == 3 and all(isinstance(i, int) for i in x)
            for x in layer_config.hidden_dims
        )
        is_list_of_ints = isinstance(layer_config.hidden_dims, list) and all(
            isinstance(x, int) for x in layer_config.hidden_dims
        )
        if not (is_list_of_tuples or is_list_of_ints):
            raise ValueError(
                "Hidden dims must be a list of integers or list of tuples for MLP_SWIGLU architecture"
            )
        if is_list_of_tuples:
            swiglu_dims = cast(List[Tuple[int, int, int]], layer_config.hidden_dims)
        else:
            swiglu_dims = [
                split_dim_for_swiglu(d)
                for d in cast(List[int], layer_config.hidden_dims)
            ]
        return MLPSwiGLU(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=swiglu_dims,
            activation="swiglu",
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# -----------------------------------------------------------------------------
# Training workflow
# -----------------------------------------------------------------------------


def train_single_model(
    model,
    model_name: str,
    model_dir: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    loss_fn: Callable,
    optimizer_type: OptimizerType,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    skip_if_exists: bool = True,
) -> tuple[Any, float, float, float]:
    """
    Train a single model or load from checkpoint if exists.

    Returns:
        tuple of (params, train_loss, val_loss, val_accuracy)
    """
    # Check if model already exists
    if skip_if_exists and check_saved_model(model_dir, model=model):
        logger.info(f"[LOAD] Loading existing model: {model_name}")
        params, _, meta = load_model_checkpoint(model_dir, model=model)
        train_loss = meta["train_loss"]
        val_loss = meta["val_loss"]
        val_accuracy = meta["val_accuracy"]
    else:
        logger.info(f"[TRAIN] Training model: {model_name}")
        start_time = time.time()

        # Train model
        model, params, loss_hist = train_model(
            model,
            train_dataset.get_dataloader(batch_size=batch_size, seed=seed),
            loss_fn,
            optimizer(optimizer_type, lr=learning_rate, weight_decay=weight_decay),
            epochs=epochs,
        )

        train_time = time.time() - start_time
        train_loss = loss_hist[-1]

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate_loss_and_classification_accuracy(
            model,
            params,
            val_dataset.inputs,
            val_dataset.targets,
            loss_fn,
        )

        logger.info(
            f"[TRAIN] Completed {model_name} in {train_time:.1f}s: "
            f"val_loss={val_loss:.6f}, val_acc={val_accuracy:.4f}"
        )

        # Save checkpoint
        save_model_checkpoint(
            model=model,
            params=params,
            directory=model_dir,
            metadata={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "training_time_seconds": train_time,
            },
        )

    return params, float(train_loss), float(val_loss), float(val_accuracy)


def run_hyperparameter_search(
    model,
    model_type: ModelArchitecture,
    experiment_hidden_layers: List[int] | List[Tuple[int, int, int]],
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainingConfig,
    model_output_dir: str,
    seed: int,
) -> list[ModelEntry]:
    """
    Run hyperparameter search for a single model architecture.

    Returns:
        List of ModelEntry objects for all trained models.
    """
    logger.info(f"[HP SEARCH] Starting for {model_type}")

    model_entries = []

    for lr in config.hyperparam_grid.learning_rates:
        assert config.hyperparam_grid.weight_decays is not None, (
            "Weight decays must be specified"
        )
        for wd in config.hyperparam_grid.weight_decays:
            # Generate unique model name
            cfg_hash = hash_data(
                {
                    "hidden_layers": experiment_hidden_layers,
                    "optimizer": config.optimizer.value,
                    "lr": lr,
                    "wd": wd,
                    "seed": seed,
                }
            )
            model_name = f"{model_type.lower()}_{cfg_hash}_lr{lr}_wd{wd}_seed{seed}"
            model_dir = f"{model_output_dir}/{model_name}/"

            # Train model
            _, train_loss, val_loss, val_accuracy = train_single_model(
                model=model,
                model_name=model_name,
                model_dir=model_dir,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                loss_fn=cross_entropy_loss,
                optimizer_type=config.optimizer,
                learning_rate=lr,
                weight_decay=wd,
                epochs=config.epochs,
                batch_size=config.batch_size,
                seed=seed,
                skip_if_exists=config.skip_existing,
            )

            # Create model entry
            entry = ModelEntry(
                model_name=model_name,
                model_type=model_type,
                model_hash=cfg_hash,
                hidden_layers=experiment_hidden_layers,  # type: ignore
                num_params=model.num_params,
                learning_rate=lr,
                weight_decay=wd,
                optimizer=config.optimizer,
                epochs=config.epochs,
                seed=seed,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                model_dir=model_dir,
            )

            model_entries.append(entry)

            logger.info(
                f"[HP SEARCH] {model_name}: "
                f"val_loss={val_loss:.6f}, val_acc={val_accuracy:.4f}"
            )

    # Find best model
    best_entry = min(model_entries, key=lambda e: e.val_loss)
    logger.info(
        f"[HP SEARCH] Best model for {model_type}: {best_entry.model_name} "
        f"(val_loss={best_entry.val_loss:.6f}, val_acc={best_entry.val_accuracy:.4f})"
    )

    return model_entries


# -----------------------------------------------------------------------------
# Main training pipeline
# -----------------------------------------------------------------------------


def run_training_pipeline(config: ExperimentConfig) -> TrainingManifest:
    """
    Execute the full training pipeline.

    This includes:
    1. Loading dataset
    2. Training models for each architecture and layer configuration
    3. Hyperparameter search
    4. Saving manifest

    Returns:
        TrainingManifest with all trained models
    """
    training_config = config.get_training_config()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    logger.info(f"Starting training pipeline: {config.experiment_name}")
    logger.info(f"Seed: {config.global_seed}")

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    logger.info(f"Loading dataset from {config.dataset.path}")
    dataset: DigitsDataset = DigitsDataset.load(
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )

    train_ds, val_ds = dataset.train_test_split(
        test_size=config.dataset.test_size,
        seed=config.dataset.split_seed,
    )

    logger.info(
        f"Dataset split: {len(train_ds.inputs)} train, {len(val_ds.inputs)} val"
    )

    # -------------------------------------------------------------------------
    # Create manifest
    # -------------------------------------------------------------------------
    manifest = create_empty_manifest(
        experiment_name=config.experiment_name,
        timestamp=timestamp,
        seed=config.global_seed,
        dataset_path=config.dataset.path,
        hyperparameter_grid={
            "learning_rates": training_config.hyperparam_grid.learning_rates,
            "weight_decays": training_config.hyperparam_grid.weight_decays,
            "optimizer": training_config.optimizer.value,
        },
    )

    # -------------------------------------------------------------------------
    # Train models
    # -------------------------------------------------------------------------
    for layer_config in training_config.model_config.layer_configs:
        logger.info(f"Training models with hidden_layers={layer_config.hidden_dims}")

        for architecture in training_config.model_config.architectures:
            logger.info(f"Architecture: {architecture.value}")

            # Create model
            model = create_model(
                architecture=architecture,
                layer_config=layer_config,
                input_dim=dataset.input_dim(),
                output_dim=dataset.output_dim(),
                seed=training_config.model_config.init_seed,
            )

            experiment_hidden_layers = layer_config.hidden_dims

            logger.info(
                f"Created {architecture.value} model "
                f"(params={model.num_params}, hidden={experiment_hidden_layers})"
            )

            # Run hyperparameter search
            assert experiment_hidden_layers is not None, (
                "Hidden layers must be specified for training"
            )
            model_entries = run_hyperparameter_search(
                model=model,
                model_type=architecture,
                experiment_hidden_layers=experiment_hidden_layers,
                train_dataset=train_ds,
                val_dataset=val_ds,
                config=training_config,
                model_output_dir=training_config.model_output_dir,
                seed=config.global_seed,
            )

            # Add to manifest
            for entry in model_entries:
                manifest.add_model(entry)

    # -------------------------------------------------------------------------
    # Save manifest
    # -------------------------------------------------------------------------
    save_manifest(manifest, training_config.manifest_output_path)

    logger.info(
        f"Training pipeline completed. "
        f"Trained {len(manifest.models)} models across "
        f"{len(manifest.best_models)} architectures."
    )

    return manifest
