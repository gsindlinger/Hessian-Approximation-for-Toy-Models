"""
Training-only script that trains models and selects best performers.

This script trains multiple model configurations with different hyperparameters,
evaluates their performance, and selects the best model for each architecture
based on validation metrics. It outputs results in both JSON (full training log)
and YAML (best models only) formats for downstream analysis.

Uses TrainingExperimentConfig instead of full ExperimentConfig.

Usage:
    # Basic run with default config (must include the configs of the models to train)
    python -m experiments.train_models \
        --config-name=training_experiment \
        --config-path=../configs
    
    # Run specific training sweep (e.g., a predefined sweep for the CONCRETE dataset)
    python -m experiments.train_models \
        --config-name=concrete_sweep \
        --config-path=../configs
    
    # Override individual parts of the config from command line, e.g., dataset & seed
    python -m experiments.train_models \
        --config-name=training_experiment \
        --config-path=../configs \
        dataset.name=CONCRETE \
        dataset.path=experiments/data/datasets/concrete \
        seed=42
    
    # Capture output in shell pipeline (for automation)
    BEST_MODELS=$(python -m experiments.train_models \
        --config-name=concrete_sweep \
        --config-path=../configs | \
        sed -n 's/^BEST_MODELS_YAML=//p')
    
Model Selection:
    Models are grouped by architecture and structure (architecture + hidden_dim).
    Within each group, the top 5 models with the best validation metric are selected.
    
    For classification (CROSS_ENTROPY loss):
        - Default metric: val_accuracy (higher is better)
        - Alternative: val_loss (lower is better)
    
    For regression (MSE loss):
        - Default metric: val_loss (lower is better)

Output Files:
    1. Full training results (JSON):
       experiments/results/<experiment_name>/training/<timestamp>.json
       Contains all trained models with their metrics and hyperparameters
    
    2. Top 5 models (YAML):
       experiments/results/<experiment_name>/top5_models/<timestamp>.yaml
       Contains the top 5 model paths for each architecture group
       Format compatible with hessian_analysis.py's override_config parameter
    
    3. Model checkpoints:
       experiments/results/<experiment_name>/models/<model_name>_<hash>/
       Contains checkpoint.msgpack, model.json for each model

Shell Output:
    The script prints "TOP5_MODELS_YAML=<path>" to stdout for pipeline capture.
    This path can be parsed and passed to subsequent analysis scripts.

Notes:
    - Input/output data is normalized for regression tasks (MSE loss)
    - Model directories are generated using config hash for reproducibility
    - Models can be skipped if already trained (set skip_existing=true)
    - Hydra manages logging; use hydra.run.dir to specify log location
    - Each model is assigned a unique directory based on its config hash
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List

import hydra
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from experiments.config_builder import (
    normalize_for_yaml,
    register_enum_representers,
)
from experiments.utils import cleanup_memory, to_dataclass
from src.config import LossType, ModelConfig, TrainingExperimentConfig
from src.utils.data.data import Dataset, DownloadableDataset
from src.utils.loss import get_loss
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import (
    check_saved_model,
    evaluate_loss,
    evaluate_loss_and_classification_accuracy,
    load_model_checkpoint,
    save_model_checkpoint,
    train_model,
)
from src.utils.utils import hash_data

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="training_experiment", node=TrainingExperimentConfig)


def generate_model_directory(
    model_config: ModelConfig, base_dir: str, seed: int
) -> str:
    """Generate a unique directory path for a model based on its configuration."""
    config_hash = hash_data(
        {
            "architecture": model_config.architecture.value,
            "hidden_dim": model_config.hidden_dim,
            "loss": model_config.loss.value,
            "learning_rate": model_config.training.learning_rate,
            "weight_decay": model_config.training.weight_decay,
            "optimizer": model_config.training.optimizer.value,
            "epochs": model_config.training.epochs,
            "batch_size": model_config.training.batch_size,
            "seed": seed,
        },
        length=12,
    )

    model_display_name = model_config.get_model_display_name()
    return os.path.join(base_dir, f"{model_display_name}_{config_hash}")


def train_single_model(
    model_config: ModelConfig,
    dataset: Dataset,
    seed: int,
    test_size: float,
) -> Dict:
    """Train a single model and return its results."""
    train_ds, val_ds = dataset.train_test_split(test_size=test_size, seed=seed)
    train_inputs, train_targets = train_ds.inputs, train_ds.targets
    val_inputs, val_targets = val_ds.inputs, val_ds.targets

    # Normalize data for regression tasks
    if model_config.loss == LossType.MSE:
        train_inputs, val_inputs = Dataset.normalize_data(train_inputs, val_inputs)
        train_targets, val_targets = Dataset.normalize_data(train_targets, val_targets)

    assert val_targets is not None, "Validation targets cannot be None"
    assert train_targets is not None, "Training targets cannot be None"
    assert val_inputs is not None, "Validation inputs cannot be None"
    assert train_inputs is not None, "Training inputs cannot be None"

    if model_config.input_dim != dataset.input_dim():
        model_config.input_dim = dataset.input_dim()
    if model_config.output_dim != dataset.output_dim():
        model_config.output_dim = dataset.output_dim()

    model = ModelRegistry.get_model(model_config=model_config, seed=seed)
    logger.info(f"Model has {model.num_params} parameters")

    model_dir = model_config.directory
    assert model_dir is not None, "Model directory must be set"

    if model_config.skip_existing and check_saved_model(model_dir):
        params, _, _, _ = load_model_checkpoint(model_dir)
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

    if model_config.loss == LossType.CROSS_ENTROPY:
        val_loss, val_acc = evaluate_loss_and_classification_accuracy(
            model, params, val_inputs, val_targets, get_loss(model_config.loss)
        )
        logger.info(
            f"[EVAL] {model_config.get_model_display_name()}: "
            f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}"
        )
        metadata = {
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
        }
    else:
        val_loss = evaluate_loss(
            model, params, val_inputs, val_targets, get_loss(model_config.loss)
        )
        train_loss = evaluate_loss(
            model, params, train_inputs, train_targets, get_loss(model_config.loss)
        )
        logger.info(
            f"[EVAL] {model_config.get_model_display_name()}: val_loss={val_loss:.6f}"
        )
        logger.info(
            f"[EVAL] {model_config.get_model_display_name()}: "
            f"train_loss={train_loss:.6f}"
        )
        metadata = {
            "val_loss": float(val_loss),
        }
    metadata.update({"num_parameters": model.num_params})

    save_model_checkpoint(
        model_config=model_config,
        params=params,
        metadata=metadata,
    )

    cleanup_memory(f"train_{model_config.get_model_display_name()}")

    if model_config.loss == LossType.CROSS_ENTROPY:
        return {
            "model_config": model_config,
            "model_directory": model_dir,
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "num_parameters": model.num_params,
        }
    else:
        return {
            "model_config": model_config,
            "model_directory": model_dir,
            "val_loss": float(val_loss),
            "num_parameters": model.num_params,
        }


def get_model_group_key(model_config: ModelConfig) -> str:
    """
    Generate a key to group models by architecture and structure.

    Models with the same architecture and hidden_dim structure but different
    hyperparameters (lr, wd) will have the same key.
    """
    return f"{model_config.architecture.value}_{str(model_config.hidden_dim)}"


def select_top_k_models(
    training_results: List[Dict],
    selection_metric: str = "val_accuracy",
    minimize: bool = False,
    k: int = 5,
) -> List[Dict]:
    """
    Select the top k models for each architecture/structure group.

    Args:
        training_results: List of training result dictionaries
        selection_metric: Metric to use for selection ('val_accuracy' or 'val_loss')
        minimize: If True, select models with minimum metric value
        k: Number of top models to select per group

    Returns:
        List of top k models per group
    """
    groups = defaultdict(list)

    for result in training_results:
        key = get_model_group_key(result["model_config"])
        groups[key].append(result)

    top_models = []

    for group_key, group_results in groups.items():
        sorted_results = sorted(
            group_results,
            key=lambda x: x[selection_metric],
            reverse=not minimize,
        )

        # Select top k models (or all if fewer than k available)
        top_k = sorted_results[:k]

        logger.info(f"[SELECTION] Top {len(top_k)} models for {group_key}:")

        for i, model in enumerate(top_k, 1):
            logger.info(
                f"  #{i}: {model['model_config'].get_model_display_name()} "
                f"(lr={model['model_config'].training.learning_rate:.2e}, "
                f"wd={model['model_config'].training.weight_decay:.2e}) "
                f"with {selection_metric}={model[selection_metric]:.6f}"
            )

        top_models.extend(top_k)

    return top_models


@hydra.main(
    version_base="1.3", config_name="training_experiment", config_path="../configs"
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    config: TrainingExperimentConfig = to_dataclass(TrainingExperimentConfig, cfg)  # type: ignore

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info(f"{'=' * 70}")
    logger.info(f"Starting Training: {config.experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Dataset: {config.dataset.name.value}")
    logger.info(f"Models to train: {len(config.models)}")
    logger.info(f"Selection metric: {config.selection_metric}")
    logger.info(f"{'=' * 70}")

    # Set model directories
    models_base_dir = config.get_models_base_dir()
    for model_config in config.models:
        if model_config.directory is None:
            model_config.directory = generate_model_directory(
                model_config, models_base_dir, seed=config.seed
            )

    # Load dataset
    dataset = DownloadableDataset.load(
        dataset=config.dataset.name,
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )
    logger.info(f"Loaded dataset: {config.dataset.name.value}")

    # Train all models
    logger.info(f"{'#' * 70}")
    logger.info("TRAINING ALL MODELS")
    logger.info(f"{'#' * 70}")

    training_results = []
    for i, model_config in enumerate(config.models, 1):
        logger.info(f"{'=' * 70}")
        logger.info(
            f"Model {i}/{len(config.models)}: {model_config.get_model_display_name()}"
        )
        logger.info(
            f"  lr={model_config.training.learning_rate:.2e}, "
            f"wd={model_config.training.weight_decay:.2e}"
        )
        logger.info(f"{'=' * 70}")

        result = train_single_model(
            model_config=model_config,
            dataset=dataset,
            seed=config.seed,
            test_size=config.dataset.test_size,
        )
        training_results.append(result)

    # Summary of all results
    logger.info(f"{'#' * 70}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'#' * 70}")
    for result in training_results:
        if model_config.loss == LossType.CROSS_ENTROPY:
            logger.info(
                f"{result['model_config'].get_model_display_name()}: "
                f"val_loss={result['val_loss']:.6f}, val_acc={result['val_accuracy']:.4f}"
            )
        else:
            logger.info(
                f"{result['model_config'].get_model_display_name()}: "
                f"val_loss={result['val_loss']:.6f}"
            )

    # Select top 5 models per group
    logger.info(f"{'#' * 70}")
    logger.info("SELECTING TOP 5 MODELS PER GROUP")
    logger.info(f"{'#' * 70}")

    top_models = select_top_k_models(
        training_results,
        selection_metric=config.selection_metric,
        minimize=config.selection_minimize,
        k=5,
    )

    logger.info(
        f"Selected {len(top_models)} top models from {len(training_results)} total"
    )

    # Save results
    results_dir = config.get_results_dir()
    os.makedirs(results_dir, exist_ok=True)

    # Save full training results
    full_results_file = os.path.join(results_dir, "training", f"{timestamp}.json")
    os.makedirs(os.path.dirname(full_results_file), exist_ok=True)
    with open(full_results_file, "w") as f:
        json.dump(
            {
                "experiment_name": config.experiment_name,
                "timestamp": timestamp,
                "total_models": len(training_results),
                "results": [
                    {
                        "model_name": r["model_config"].get_model_display_name(),
                        "model_directory": r["model_directory"],
                        "val_loss": r["val_loss"],
                        "num_parameters": r["num_parameters"],
                        "learning_rate": r["model_config"].training.learning_rate,
                        "weight_decay": r["model_config"].training.weight_decay,
                    }
                    if r["model_config"].loss != LossType.CROSS_ENTROPY.value
                    else {
                        "model_name": r["model_config"].get_model_display_name(),
                        "model_directory": r["model_directory"],
                        "val_loss": r["val_loss"],
                        "val_accuracy": r["val_accuracy"],
                        "num_parameters": r["num_parameters"],
                        "learning_rate": r["model_config"].training.learning_rate,
                        "weight_decay": r["model_config"].training.weight_decay,
                    }
                    for r in training_results
                ],
            },
            f,
            indent=2,
        )

    logger.info(f"Full training results saved to: {full_results_file}")

    # Prepare top models with proper directory paths for Hessian analysis
    top_models_for_hessian = []

    for r in top_models:
        model_config: ModelConfig = r["model_config"]

        # Create a copy of the model config dict and add Hessian paths
        model_directory = asdict(model_config)
        if model_config.loss == LossType.CROSS_ENTROPY:
            top_models_for_hessian.append(
                {
                    "model_config": model_directory,
                    "val_loss": r["val_loss"],
                    "val_accuracy": r["val_accuracy"],
                    "num_parameters": r["num_parameters"],
                }
            )
        else:
            top_models_for_hessian.append(
                {
                    "model_config": model_directory,
                    "val_loss": r["val_loss"],
                    "num_parameters": r["num_parameters"],
                }
            )

    # Save top 5 models as YAML for direct use in ExperimentConfig
    top_models_yaml = os.path.join(results_dir, "top5_models", f"{timestamp}.yaml")

    # Prepare model configs for YAML (just the model_config part)
    models_for_yaml = []
    for r in top_models:
        model_directory = r["model_directory"]
        models_for_yaml.append(model_directory)

    os.makedirs(os.path.dirname(top_models_yaml), exist_ok=True)
    register_enum_representers()
    with open(top_models_yaml, "w") as f:
        yaml.dump(
            {
                "dataset": asdict(config.dataset),
                "seed": config.seed,
                "models": normalize_for_yaml(models_for_yaml),
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    logger.info(f"Top 5 models YAML saved to: {top_models_yaml}")

    logger.info(f"{'=' * 70}")
    logger.info("Training Complete!")
    logger.info(f"Total models trained: {len(training_results)}")
    logger.info(f"Top models selected: {len(top_models)}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"Full results (JSON): {full_results_file}")
    logger.info(f"Top 5 models (YAML): {top_models_yaml}")

    print(f"TOP5_MODELS_YAML={top_models_yaml}")


if __name__ == "__main__":
    main()
