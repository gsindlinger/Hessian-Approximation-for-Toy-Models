#!/usr/bin/env python3
"""
Training script - Stage 1 of the experiment pipeline.

This script trains models across hyperparameter grids and saves a manifest
of trained models for later Hessian analysis.

Usage:
    python scripts/train.py --help
    
Examples:
    # Basic training with default config
    python scripts/train.py --dataset.path=data/digits
    
    # Full hyperparameter search
    python scripts/train.py \
        --dataset.path=data/digits \
        --training.hyperparam_grid.learning_rates=[1e-4,5e-4,1e-3] \
        --training.hyperparam_grid.weight_decays=[0.0,1e-4,1e-3] \
        --training.model_config.architectures=[mlp,mlpswiglu]
    
    # Single model training
    python scripts/train.py \
        --dataset.path=data/digits \
        --training.hyperparam_grid.learning_rates=[1e-3] \
        --training.hyperparam_grid.weight_decays=[0.0]
"""

import logging
import sys
from pathlib import Path

from simple_parsing import ArgumentParser

from experiments.sweep_1.scripts.training_pipeline import run_training_pipeline
from src.config import ExperimentConfig, Stage, TrainingConfig

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description="Train models with hyperparameter search")

    # Add config arguments
    parser.add_arguments(ExperimentConfig, dest="config")

    args = parser.parse_args()
    config: ExperimentConfig = args.config

    # Override stage to ensure it's TRAIN
    config.stage = Stage.TRAIN

    # Ensure training config exists
    if config.training is None:
        config.training = TrainingConfig()

    # Log configuration
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Experiment name: {config.experiment_name}")
    logger.info(f"Dataset: {config.dataset.path}")
    logger.info(f"Global seed: {config.global_seed}")
    logger.info(f"Output directory: {config.training.model_output_dir}")
    logger.info(f"Manifest output: {config.training.manifest_output_path}")
    logger.info(
        f"Architectures: {[a.value for a in config.training.model_config.architectures]}"
    )
    logger.info(
        f"Layer configs: {[lc.hidden_dims for lc in config.training.model_config.layer_configs]}"
    )
    logger.info(f"Learning rates: {config.training.hyperparam_grid.learning_rates}")
    logger.info(f"Weight decays: {config.training.hyperparam_grid.weight_decays}")
    logger.info(f"Optimizer: {config.training.optimizer.value}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info("=" * 80)

    # Validate
    if not Path(config.dataset.path).exists():
        logger.error(f"Dataset path does not exist: {config.dataset.path}")
        sys.exit(1)

    # Run training
    try:
        run_training_pipeline(config)
        logger.info("✓ Training completed successfully")
        logger.info(f"✓ Manifest saved to: {config.training.manifest_output_path}")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
