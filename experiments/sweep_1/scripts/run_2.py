#!/usr/bin/env python3
"""
Unified experiment runner - Single entry point for all pipeline stages.

This script automatically determines which pipeline to run based on the
configuration provided. It supports:
- Training only (stage=train)
- Hessian analysis only (stage=hessian)
- Full pipeline (stage=both)

Usage:
    python run.py --config_path=<path_to_config.json>
    python run.py --stage=<train|hessian|both> [other flags...]
    
Examples:
    # From config file
    python run.py --config_path=configs/my_experiment.json
    
    # Training only
    python run.py \
        --stage=train \
        --dataset.path=data/digits \
        --experiment_name=training_run
    
    # Hessian analysis only
    python run.py \
        --stage=hessian \
        --dataset.path=data/digits \
        --hessian.input_manifest_path=experiments/manifest.json
    
    # Full pipeline
    python run.py \
        --stage=both \
        --dataset.path=data/digits \
        --experiment_name=full_experiment
    
    # Auto-detect from config (if stage not specified, infers from available configs)
    python run.py \
        --dataset.path=data/digits \
        --training.hyperparam_grid.learning_rates=[1e-3] \
        --training.manifest_output_path=experiments/manifest.json
        # Will automatically run training pipeline
"""

import logging
import sys
import time
from pathlib import Path

from simple_parsing import ArgumentParser

from experiments.sweep_1.scripts.utils.hessian_utils import run_hessian_pipeline
from experiments.sweep_1.scripts.utils.training_utils import run_training_pipeline
from src.config import ExperimentConfig, Stage
from src.utils.manifest import TrainingManifest, load_manifest

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def infer_stage_from_config(config: ExperimentConfig) -> Stage:
    """
    Infer the execution stage from the configuration if not explicitly set.

    Logic:
    - If both training and hessian configs exist -> BOTH
    - If only training config exists -> TRAIN
    - If only hessian config exists -> HESSIAN
    - Otherwise -> raise error
    """
    has_training = config.training is not None
    has_hessian = config.hessian is not None

    if has_training and has_hessian:
        logger.info(
            "Auto-detected stage: BOTH (both training and hessian configs present)"
        )
        return Stage.BOTH
    elif has_training and not has_hessian:
        logger.info("Auto-detected stage: TRAIN (only training config present)")
        return Stage.TRAIN
    elif has_hessian and not has_training:
        logger.info("Auto-detected stage: HESSIAN (only hessian config present)")
        return Stage.HESSIAN
    else:
        raise ValueError(
            "Cannot infer stage: no training or hessian config provided. "
            "Please specify --stage=<train|hessian|both> or provide appropriate configs."
        )


def validate_config(config: ExperimentConfig):
    """Validate configuration before running pipeline."""
    # Validate dataset path
    if not Path(config.dataset.path).exists():
        raise ValueError(f"Dataset path does not exist: {config.dataset.path}")

    # Stage-specific validation
    if config.stage in [Stage.TRAIN, Stage.BOTH]:
        if config.training is None:
            raise ValueError("Training config required for TRAIN or BOTH stages")

    if config.stage in [Stage.HESSIAN, Stage.BOTH]:
        if config.hessian is None:
            raise ValueError("Hessian config required for HESSIAN or BOTH stages")

        # For HESSIAN-only stage, manifest must exist
        if config.stage == Stage.HESSIAN:
            if config.hessian.input_manifest_path is None:
                raise ValueError(
                    "hessian.input_manifest_path required for HESSIAN stage"
                )
            if not Path(config.hessian.input_manifest_path).exists():
                raise ValueError(
                    f"Manifest not found: {config.hessian.input_manifest_path}"
                )

    # For BOTH stage, auto-connect training output to hessian input
    if config.stage == Stage.BOTH:
        if config.hessian is None:
            raise ValueError("Hessian config required for BOTH stage")
        if config.training is None:
            raise ValueError("Training config required for BOTH stage")

        if config.hessian.input_manifest_path is None:
            config.hessian.input_manifest_path = config.training.manifest_output_path
            logger.info(
                f"Auto-connected: hessian.input_manifest_path = "
                f"{config.training.manifest_output_path}"
            )
        else:
            assert (
                config.hessian.input_manifest_path
                == config.training.manifest_output_path
            ), (
                "Mismatch: hessian.input_manifest_path must match "
                "training.manifest_output_path in BOTH stage"
            )


# -----------------------------------------------------------------------------
# Pipeline runners
# -----------------------------------------------------------------------------


def run_training_stage(config: ExperimentConfig) -> dict:
    """Run training pipeline and return results."""

    start_time = time.time()
    manifest = run_training_pipeline(config)
    elapsed_time = time.time() - start_time

    assert config.training is not None, "Training config must be provided"

    logger.info(
        f"✓ Training completed in {elapsed_time:.1f}s ({elapsed_time / 60:.1f} min)"
    )
    logger.info(f"✓ Trained {len(manifest.models)} models")
    logger.info(f"✓ Best models: {list(manifest.best_models.keys())}")
    logger.info(f"✓ Manifest saved: {config.training.manifest_output_path}")

    return {
        "manifest": manifest,
        "elapsed_time": elapsed_time,
        "num_models": len(manifest.models),
    }


def run_hessian_stage(config: ExperimentConfig) -> dict:
    """Run Hessian analysis pipeline and return results."""

    assert config.hessian is not None, "Hessian config must be provided"
    assert config.hessian.input_manifest_path is not None, (
        "hessian.input_manifest_path must be specified for Hessian analysis"
    )

    # Load manifest to show what we're analyzing
    manifest = load_manifest(config.hessian.input_manifest_path)

    logger.info(f"Experiment: {manifest.experiment_name}")
    logger.info(f"Total models in manifest: {len(manifest.models)}")
    logger.info(f"Best models: {list(manifest.best_models.keys())}")

    # Filter models
    min_accuracy = None
    if config.training is not None:
        min_accuracy = config.training.min_accuracy_threshold

    models_to_analyze = manifest.filter_models(
        pattern=config.hessian.model_filter,
        min_accuracy=min_accuracy,
    )

    logger.info(f"Selected {len(models_to_analyze)} models")
    if min_accuracy:
        logger.info(f"(filtered by min_accuracy={min_accuracy})")

    if len(models_to_analyze) == 0:
        logger.error("No models meet criteria for Hessian analysis")
        sys.exit(1)

    # Confirm
    if not config.hessian.computation_config.approximators:
        logger.error("No approximators specified in config")
        sys.exit(1)

    # Run analysis
    start_time = time.time()
    results = run_hessian_pipeline(config)
    elapsed_time = time.time() - start_time

    logger.info(
        f"✓ Analysis completed in {elapsed_time:.1f}s ({elapsed_time / 60:.1f} min)"
    )
    logger.info(f"✓ Analyzed {len(results['results'])} models")
    logger.info(f"✓ Results saved: {config.hessian.results_output_dir}")

    return {
        "results": results,
        "elapsed_time": elapsed_time,
        "num_models_analyzed": len(results["results"]),
    }


def run_full_pipeline(config: ExperimentConfig) -> dict:
    """Run both training and Hessian analysis pipelines."""
    assert config.training is not None, "Training config must be provided"
    assert config.hessian is not None, "Hessian config must be provided"

    total_start = time.time()

    # Stage 1: Training
    training_results = run_training_stage(config)

    # Check if we should proceed to analysis
    manifest: TrainingManifest = training_results["manifest"]
    min_accuracy = config.training.min_accuracy_threshold

    models_eligible = manifest.filter_models(
        pattern=config.hessian.model_filter,
        min_accuracy=min_accuracy,
    )

    if len(models_eligible) == 0:
        logger.warning(
            f"No models meet criteria for Hessian analysis "
            f"(min_accuracy={min_accuracy}, filter={config.hessian.model_filter})"
        )
        logger.info("Pipeline completed (training only)")

        return {"training": training_results, "hessian": None}

    # Stage 2: Hessian Analysis
    hessian_results = run_hessian_stage(config)

    # Summary
    total_time = time.time() - total_start

    return {
        "training": training_results,
        "hessian": hessian_results,
        "total_time": total_time,
    }


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main():
    parser = ArgumentParser(
        description="Unified experiment runner for training and Hessian analysis"
    )

    # Add config arguments
    parser.add_arguments(ExperimentConfig, dest="config")

    args = parser.parse_args()
    config: ExperimentConfig = args.config

    try:
        # Auto-detect stage if not explicitly set or if set to default
        if config.stage == Stage.BOTH and (
            config.training is None or config.hessian is None
        ):
            # Stage was not explicitly set or is ambiguous, infer from config
            config.stage = infer_stage_from_config(config)

        # Validate configuration
        validate_config(config)

        # Run appropriate pipeline based on stage
        if config.stage == Stage.TRAIN:
            results = run_training_stage(config)
            logger.info("=" * 80)
            logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

        elif config.stage == Stage.HESSIAN:
            results = run_hessian_stage(config)
            logger.info("=" * 80)
            logger.info("✓ HESSIAN ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

        elif config.stage == Stage.BOTH:
            results = run_full_pipeline(config)
            logger.info("=" * 80)
            logger.info("✓ FULL PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

        else:
            logger.error(f"Unknown stage: {config.stage}")
            sys.exit(1)

        return results

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
