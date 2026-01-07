#!/usr/bin/env python3
"""
Full pipeline script - Run both training and Hessian analysis.

This script orchestrates the complete experiment pipeline:
1. Train models with hyperparameter search
2. Analyze best models with Hessian approximations

Usage:
    python scripts.run_full_pipeline --help
    
Examples:
    # Run full pipeline with defaults
    python scripts.run_full_pipeline \
        --dataset.path=data/digits \
        --experiment_name=my_experiment
    
    # Custom hyperparameter search + selective Hessian analysis
    python scripts.run_full_pipeline \
        --dataset.path=data/digits \
        --experiment_name=sweep_custom \
        --training.hyperparam_grid.learning_rates=[1e-4,1e-3,1e-2] \
        --training.hyperparam_grid.weight_decays=[0.0,1e-3] \
        --hessian.computation_config.comparison_references=[gnh] \
        --hessian.computation_config.skip_exact_above_params=10000
    
    # Quick test run (single config, minimal analysis)
    python scripts.run_full_pipeline \
        --dataset.path=data/digits \
        --experiment_name=quick_test \
        --training.hyperparam_grid.learning_rates=[1e-3] \
        --training.hyperparam_grid.weight_decays=[0.0] \
        --training.model_config.layer_configs='[{"hidden_dims": [64]}]' \
        --hessian.computation_config.computation_types=[hvp]
"""

import logging
import sys
import time
from pathlib import Path

from simple_parsing import ArgumentParser

from experiments.sweep_1.scripts.hessian_comparison_pipeline import run_hessian_pipeline
from experiments.sweep_1.scripts.training_pipeline import run_training_pipeline
from src.config import ExperimentConfig, Stage
from src.utils.manifest import list_models_table

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description="Run full training + Hessian analysis pipeline")

    # Add config arguments
    parser.add_arguments(ExperimentConfig, dest="config")
    args = parser.parse_args()
    config: ExperimentConfig = args.config

    # Set stage to BOTH
    config.stage = Stage.BOTH

    # Validate
    if not Path(config.dataset.path).exists():
        logger.error(f"Dataset path does not exist: {config.dataset.path}")
        sys.exit(1)

    if config.training is None or config.hessian is None:
        logger.error("Both training and hessian configs must be provided")
        sys.exit(1)

    # Set hessian input to training output if not specified
    if config.hessian.input_manifest_path is None:
        config.hessian.input_manifest_path = config.training.manifest_output_path

    start_time = time.time()

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Dataset: {config.dataset.path}")
    logger.info(f"Root output: {config.root_output_dir}")

    # =========================================================================
    # STAGE 1: TRAINING
    # =========================================================================

    logger.info("STAGE 1: TRAINING")
    logger.info("Configuration:")
    logger.info(
        f"  Architectures: {[a.value for a in config.training.model_config.architectures]}"
    )
    logger.info(
        f"  Layer configs: {[lc.hidden_dims for lc in config.training.model_config.layer_configs]}"
    )
    logger.info(f"  Learning rates: {config.training.hyperparam_grid.learning_rates}")
    logger.info(f"  Weight decays: {config.training.hyperparam_grid.weight_decays}")
    logger.info(
        f"  Total configs: {config.training.hyperparam_grid.num_configurations()}"
    )

    try:
        training_start = time.time()
        manifest = run_training_pipeline(config)
        training_time = time.time() - training_start

        logger.info(f"✓ Training completed in {training_time:.1f}s")
        logger.info(f"✓ Trained {len(manifest.models)} models")
        logger.info(f"✓ Manifest: {config.training.manifest_output_path}")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)

    # =========================================================================
    # STAGE 2: HESSIAN ANALYSIS
    # =========================================================================

    logger.info("STAGE 2: HESSIAN ANALYSIS")

    # Filter models for analysis
    models_to_analyze = manifest.filter_models(
        pattern=config.hessian.model_filter,
        min_accuracy=config.training.min_accuracy_threshold,
    )

    logger.info(f"Selected {len(models_to_analyze)} models for Hessian analysis")
    logger.info(
        f"  (filtered from {len(manifest.models)} total, "
        f"min_accuracy={config.training.min_accuracy_threshold})"
    )

    if len(models_to_analyze) == 0:
        logger.warning("No models meet criteria for Hessian analysis")
        logger.info("Pipeline completed (training only)")
        sys.exit(0)

    list_models_table(models_to_analyze, max_rows=20)

    logger.info("Configuration:")
    logger.info(
        f"  Approximators: {[a.value for a in config.hessian.computation_config.approximators]}"
    )
    logger.info(
        f"  References: {[r.value for r in config.hessian.computation_config.comparison_references]}"
    )
    logger.info(
        f"  Computation types: {[t.value for t in config.hessian.computation_config.computation_types]}"
    )
    logger.info(f"  Damping: {config.hessian.damping_config.strategy.value}")

    try:
        analysis_start = time.time()
        results = run_hessian_pipeline(config)
        analysis_time = time.time() - analysis_start

        logger.info("HESSIAN ANALYSIS RESULTS")
        logger.info(f"✓ Analysis completed in {analysis_time:.1f}s")
        logger.info(f"✓ Analyzed {len(results['results'])} models")
        logger.info(f"✓ Results: {config.hessian.results_output_dir}")

    except Exception as e:
        logger.exception(f"Hessian analysis failed: {e}")
        sys.exit(1)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    total_time = time.time() - start_time

    logger.info("PIPELINE SUMMARY")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info(f"  Training: {training_time:.1f}s ({training_time / 60:.1f} min)")
    logger.info(f"  Analysis: {analysis_time:.1f}s ({analysis_time / 60:.1f} min)")
    logger.info(f"\nModels trained: {len(manifest.models)}")
    logger.info(f"Models analyzed: {len(results['results'])}")
    logger.info("\nOutputs:")
    logger.info(f"  Training manifest: {config.training.manifest_output_path}")
    logger.info(f"  Hessian results: {config.hessian.results_output_dir}")
    logger.info("✓ Full pipeline completed successfully")


if __name__ == "__main__":
    main()
