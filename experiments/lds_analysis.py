"""
LDS (Linear Datamodeling Score) via ELSO (Expected Leave-Some-Out) retraining.

Evaluates data attribution methods by measuring their ability to predict the
counterfactual effect of *removing* groups of training examples from D.

ELSO formulation (Bae et al. 2022):
  Ground truth:  Δm_j(z_q) = E_ξ[m(z_q, θ(D\\S_j))] - m(z_q, θ(D))
  Predicted:     g_τ(z_q, S_j) = Σ_{z_i ∈ S_j} τ(z_q, z_i, D)
  LDS:           Spearman({Δm_j}, {g_τ_j}) averaged over query points

Influence scores use the standard Newton approximation:
  τ(z_q, z_i) = -(H^{-1} ∇_θ L(z_q, θ)) · ∇_θ L(z_i, θ)

Where H is the (approximate) Hessian of the average training loss.

Usage:
    # Basic run with predefined config
    uv run python -m experiments.lds_analysis \\
        --config-name=lds_experiment \\
        --config-path=../configs

    # Use models from a training run
    uv run python -m experiments.lds_analysis \\
        --config-name=lds_experiment \\
        --config-path=../configs \\
        +override_config=path/to/best_models.yaml

    # Override LDS parameters from CLI
    uv run python -m experiments.lds_analysis \\
        --config-name=lds_experiment \\
        --config-path=../configs \\
        num_subsets=200 \\
        reps_per_model=5 \\
        subset_fraction=0.3

    # Combine override file with CLI parameter changes
    uv run python -m experiments.lds_analysis \\
        --config-name=lds_experiment \\
        --config-path=../configs \\
        +override_config=path/to/best_models.yaml \\
        num_test_examples=50

Notes:
    - The override_config YAML file specifies model paths, dataset config, and seed
    - Results are saved as timestamped JSON files in lds_analysis.results_output_dir
    - Core LDS/ELSO logic lives in src/utils/lds.py; influence utilities in src/utils/influence.py
"""

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Dict

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from experiments.utils import (
    cleanup_memory,
    json_safe,
    load_experiment_override_from_yaml,
    to_dataclass,
)
from src.config import LDSExperimentConfig
from src.utils.data.data import DownloadableDataset
from src.utils.lds import compute_lds_for_model

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="lds_experiment", node=LDSExperimentConfig)


@hydra.main(version_base="1.3", config_name="lds_experiment", config_path="../configs")
def main(cfg: DictConfig) -> Dict:
    OmegaConf.resolve(cfg)

    override_file = cfg.get("override_config", None)
    if override_file:
        logger.info("[CONFIG] Loading overrides from: %s", override_file)
        model_directories, dataset_config, seed, _ = load_experiment_override_from_yaml(
            override_file
        )
        cfg.models = model_directories
        if dataset_config is not None:
            cfg.dataset = asdict(dataset_config)
        if seed is not None:
            cfg.seed = seed

    config: LDSExperimentConfig = to_dataclass(LDSExperimentConfig, cfg)  # type: ignore

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info("=" * 70)
    logger.info("LDS Analysis: %s  [%s]", config.experiment_name, timestamp)
    logger.info(
        "Dataset: %s | Models: %d", config.dataset.name.value, len(config.models)
    )
    logger.info("=" * 70)

    full_dataset = DownloadableDataset.load(
        dataset=config.dataset.name,
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )
    train_dataset, test_dataset = full_dataset.train_test_split(
        test_size=config.dataset.test_size,
        seed=config.seed,
    )
    logger.info(
        "Split: %d train / %d test",
        len(train_dataset.inputs),
        len(test_dataset.inputs),
    )

    lds_results = []
    for i, model_dir in enumerate(config.models, 1):
        logger.info("[MODEL %d/%d] %s", i, len(config.models), model_dir)
        result = compute_lds_for_model(
            model_directory=model_dir,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=config,
        )
        lds_results.append(result)
        cleanup_memory(f"model_{i}")

    results_dir = config.results_output_dir
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{timestamp}.json")

    full_results = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "lds_config": json_safe(asdict(config)),
        "results": json_safe(lds_results),
    }
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info("Results saved to: %s", output_file)

    # Summary table
    logger.info("=" * 70)
    logger.info("LDS SUMMARY")
    logger.info("=" * 70)
    for result in lds_results:
        logger.info("%s:", result["model_name"])
        for method, s in result["lds_scores"].items():
            lo = float(np.mean(s["per_query_ci_low"]))
            hi = float(np.mean(s["per_query_ci_high"]))
            logger.info(
                "  %-12s %.4f ± %.4f  (95%% CI [%.4f, %.4f])",
                method,
                s["mean_lds"],
                s["std_lds"],
                lo,
                hi,
            )

    return full_results


if __name__ == "__main__":
    main()
