#!/usr/bin/env python3
"""
Hessian analysis script - Stage 2 of the experiment pipeline.

This script loads trained models from a manifest and performs Hessian
approximation analysis with configurable comparisons.

Usage:
    python scripts/analyze_hessian.py --help
    
Examples:
    # Analyze all best models with default comparisons
    python scripts/analyze_hessian.py \
        --dataset.path=data/digits \
        --hessian.input_manifest_path=experiments/training_manifest.json
    
    # Analyze specific models with exact Hessian
    python scripts/analyze_hessian.py \
        --dataset.path=data/digits \
        --hessian.input_manifest_path=experiments/training_manifest.json \
        --hessian.model_filter="mlp.*seed42" \
        --hessian.computation_config.comparison_references=[exact,gnh]
    
    # Skip expensive computations
    python scripts/analyze_hessian.py \
        --dataset.path=data/digits \
        --hessian.input_manifest_path=experiments/training_manifest.json \
        --hessian.computation_config.comparison_references=[gnh] \
        --hessian.computation_config.computation_types=[hvp,ihvp]
    
    # Custom damping
    python scripts/analyze_hessian.py \
        --dataset.path=data/digits \
        --hessian.input_manifest_path=experiments/training_manifest.json \
        --hessian.damping_config.strategy=fixed \
        --hessian.damping_config.fixed_value=0.01
"""

import logging
import sys
from pathlib import Path

from simple_parsing import ArgumentParser

from src.config import ExperimentConfig, Stage
from src.utils.manifest import load_manifest

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(
        description="Analyze Hessian approximations for trained models"
    )

    # Add config arguments
    parser.add_arguments(ExperimentConfig, dest="config")

    args = parser.parse_args()
    config: ExperimentConfig = args.config

    # Override stage to ensure it's HESSIAN
    config.stage = Stage.HESSIAN

    # Ensure hessian config exists
    if config.hessian is None:
        logger.error("Hessian analysis config must be provided")
        sys.exit(1)

    # Validate inputs
    if config.hessian.input_manifest_path is None:
        logger.error("input_manifest_path must be specified for Hessian analysis")
        sys.exit(1)

    if not Path(config.hessian.input_manifest_path).exists():
        logger.error(
            f"Manifest path does not exist: {config.hessian.input_manifest_path}"
        )
        sys.exit(1)

    if not Path(config.dataset.path).exists():
        logger.error(f"Dataset path does not exist: {config.dataset.path}")
        sys.exit(1)

    # Load and display manifest info
    manifest = load_manifest(config.hessian.input_manifest_path)
    logger.info("=" * 80)
    logger.info("Loaded Training Manifest")
    logger.info("=" * 80)
    logger.info(f"Experiment: {manifest.experiment_name}")
    logger.info(f"Total models: {len(manifest.models)}")
    logger.info(f"Best models: {list(manifest.best_models.keys())}")

    # Filter models
    models_to_analyze = manifest.filter_models(
        pattern=config.hessian.model_filter,
    )

    logger.info("=" * 80)
    logger.info("Models Selected for Analysis")
    logger.info("=" * 80)
    list_models_table(models_to_analyze, max_rows=50)

    # Log configuration
    logger.info("=" * 80)
    logger.info("Hessian Analysis Configuration")
    logger.info("=" * 80)
    logger.info(f"Experiment name: {config.experiment_name}")
    logger.info(f"Dataset: {config.dataset.path}")
    logger.info(f"Analysis seed: {config.hessian.analysis_seed}")
    logger.info(
        f"Approximators: {[a.value for a in config.hessian.computation_config.approximators]}"
    )
    logger.info(
        f"Reference methods: {[r.value for r in config.hessian.computation_config.comparison_references]}"
    )
    logger.info(
        f"Computation types: {[t.value for t in config.hessian.computation_config.computation_types]}"
    )
    logger.info(f"Damping strategy: {config.hessian.damping_config.strategy.value}")
    if config.hessian.damping_config.strategy.value == "fixed":
        logger.info(f"Fixed damping value: {config.hessian.damping_config.fixed_value}")
    else:
        logger.info(
            f"Auto damping multiplier: {config.hessian.damping_config.auto_multiplier}"
        )
    logger.info(f"Output directory: {config.hessian.hessian_output_dir}")
    logger.info(f"Results directory: {config.hessian.results_output_dir}")
    logger.info("=" * 80)

    # Confirm with user
    if len(models_to_analyze) == 0:
        logger.error("No models selected for analysis. Check your filter settings.")
        sys.exit(1)

    response = input(
        f"\nProceed with analysis of {len(models_to_analyze)} models? [y/N]: "
    )
    if response.lower() not in ["y", "yes"]:
        logger.info("Analysis cancelled by user")
        sys.exit(0)

    # Run analysis
    try:
        results = run_hessian_pipeline(config)

        logger.info("=" * 80)
        logger.info("✓ Hessian analysis completed successfully")
        logger.info(f"✓ Analyzed {len(results['results'])} models")
        logger.info(f"✓ Results saved to: {config.hessian.results_output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"Hessian analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
