"""
LDS (Linear Datamodeling Score) via ELSO (Expected Leave-Some-Out) retraining.

Evaluates data attribution methods by measuring their ability to predict the
counterfactual effect of *removing* groups of training examples from D.

ELSO formulation (Bae et al. 2022):
  Ground truth:  Δm_j(z_q) = E_ξ[m(z_q, θ(D\\S_j))] - m(z_q, θ(D))
  Predicted:     g_τ(z_q, S_j) = Σ_{z_i ∈ S_j} τ(z_q, z_i, D)
  LDS:           Spearman({Δm_j}, {g_τ_j}) averaged over query points

Influence scores use the standard Newton approximation:
  τ(z_q, z_i) = (H^{-1} ∇_θ L(z_q, θ)) · ∇_θ L(z_i, θ)

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

    # Analyze specific epoch checkpoints
    uv run python -m experiments.lds_analysis \\
        --config-name=lds_experiment \\
        --config-path=../configs \\
        +override_config=path/to/best_models.yaml \\
        epochs=[10,100,1000]

    # Override LDS parameters from CLI
    uv run python -m experiments.lds_analysis \\
        --config-name=lds_experiment \\
        --config-path=../configs \\
        num_subsets=200 \\
        reps_per_model=5 \\
        subset_fraction=0.3

Notes:
    - epochs=None (default): analyzes only final checkpoints
    - epochs=[10,100,1000]: analyzes each model at those epoch checkpoints
    - All specified epoch checkpoints must exist (saved via train_models save_epochs)
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
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import jax

from experiments.utils import (
    cleanup_memory,
    json_safe,
    load_experiment_override_from_yaml,
    to_dataclass,
)
from src.config import LDSExperimentConfig
from src.utils.data.data import Dataset, DownloadableDataset
from src.utils.lds import compute_lds_for_model, compute_lds_for_model_multi_epoch
from src.utils.loss import get_loss
from src.utils.train import (
    evaluate_loss_and_classification_accuracy,
    load_model_checkpoint,
)

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="lds_experiment", node=LDSExperimentConfig)


def _evaluate_checkpoint_metrics(
    model_directory: str,
    epoch: "int | None",
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> Dict:
    """Load the checkpoint at ``epoch`` and evaluate loss/accuracy on train + test.

    Replaces the stale ``model.json`` metadata (which always reflects the
    final-training-epoch validation metrics regardless of which epoch
    checkpoint is actually loaded) with metrics computed from the real params
    at this specific checkpoint.
    """
    params, model, model_config, _ = load_model_checkpoint(
        model_directory, epoch=epoch
    )
    loss_fn = get_loss(model_config.loss)
    test_loss, test_acc = evaluate_loss_and_classification_accuracy(
        model, params, test_dataset.inputs, test_dataset.targets, loss_fn
    )
    train_loss, train_acc = evaluate_loss_and_classification_accuracy(
        model, params, train_dataset.inputs, train_dataset.targets, loss_fn
    )
    num_parameters = int(
        sum(x.size for x in jax.tree_util.tree_leaves(params))
    )
    return {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "num_parameters": num_parameters,
    }


@hydra.main(version_base="1.3", config_name="lds_experiment", config_path="../configs")
def main(cfg: DictConfig) -> Dict:
    OmegaConf.resolve(cfg)

    override_file = cfg.get("override_config", None)
    if override_file:
        logger.info("[CONFIG] Loading overrides from: %s", override_file)
        model_directories, dataset_config, seed, epochs = (
            load_experiment_override_from_yaml(override_file)
        )
        cfg.models = model_directories
        if dataset_config is not None:
            cfg.dataset = asdict(dataset_config)
        if seed is not None:
            cfg.seed = seed
        if epochs is not None:
            cfg.epochs = epochs

    config: LDSExperimentConfig = to_dataclass(LDSExperimentConfig, cfg)  # type: ignore

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info("=" * 70)
    logger.info("LDS Analysis: %s  [%s]", config.experiment_name, timestamp)
    logger.info(
        "Dataset: %s | Models: %d", config.dataset.name.value, len(config.models)
    )
    if config.epochs:
        logger.info(
            "Epochs: %s  (total analyses: %d)",
            config.epochs,
            len(config.models) * len(config.epochs),
        )
    else:
        logger.info("Analyzing final checkpoints only")
    logger.info("=" * 70)

    # Validate epoch checkpoints exist before doing any expensive retraining
    if config.epochs:
        from src.utils.train import check_saved_model

        for model_dir in config.models:
            if not check_saved_model(model_dir, config.epochs):
                raise FileNotFoundError(
                    f"Missing epoch checkpoints {config.epochs} in {model_dir}"
                )

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
    if config.epochs:
        # Multi-epoch path: one ELSO retraining per model covers all epoch
        # checkpoints, reducing total retraining by len(epochs)×.
        for i, model_dir in enumerate(config.models, 1):
            logger.info(
                "[%d/%d] %s (epochs=%s, multi-epoch ELSO)",
                i,
                len(config.models),
                model_dir,
                config.epochs,
            )

            # Evaluate each checkpoint on train/test before LDS computation so
            # the reported metadata reflects the actual params at that epoch
            # rather than the stale metrics cached in model.json.
            checkpoint_metrics: Dict[int, Dict] = {}
            for ep in config.epochs:
                metrics = _evaluate_checkpoint_metrics(
                    model_dir, ep, train_dataset, test_dataset
                )
                checkpoint_metrics[ep] = metrics
                logger.info(
                    "[%d/%d] epoch=%d  train_loss=%.4f train_acc=%.4f | "
                    "test_loss=%.4f test_acc=%.4f",
                    i,
                    len(config.models),
                    ep,
                    metrics["train_loss"],
                    metrics["train_accuracy"],
                    metrics["test_loss"],
                    metrics["test_accuracy"],
                )

            epoch_results = compute_lds_for_model_multi_epoch(
                model_directory=model_dir,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                config=config,
                epochs=config.epochs,
            )
            for r in epoch_results:
                r["metadata"] = checkpoint_metrics[r["epoch"]]
            lds_results.extend(epoch_results)
            cleanup_memory(f"model_{i}")
    else:
        # Single final-checkpoint path (original behaviour).
        for i, model_dir in enumerate(config.models, 1):
            logger.info("[%d/%d] %s (final)", i, len(config.models), model_dir)
            metrics = _evaluate_checkpoint_metrics(
                model_dir, None, train_dataset, test_dataset
            )
            logger.info(
                "[%d/%d] final  train_loss=%.4f train_acc=%.4f | "
                "test_loss=%.4f test_acc=%.4f",
                i,
                len(config.models),
                metrics["train_loss"],
                metrics["train_accuracy"],
                metrics["test_loss"],
                metrics["test_accuracy"],
            )
            result = compute_lds_for_model(
                model_directory=model_dir,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                config=config,
                epoch=None,
            )
            result["metadata"] = metrics
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
        epoch_label = f" epoch={result['epoch']}" if result.get("epoch") else ""
        logger.info("%s%s:", result["model_name"], epoch_label)
        for method, s in result["lds_scores"].items():
            logger.info(
                "  %-12s %.4f ± %.4f  (95%% CI [%.4f, %.4f])",
                method,
                s["mean_lds"],
                s["std_lds"],
                s["ci_low"],
                s["ci_high"],
            )

    return full_results


if __name__ == "__main__":
    main()
