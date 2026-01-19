"""
LDS (Linear Datamodeling Score) analysis script.

Evaluates data attribution methods by their ability to make counterfactual
predictions about model outputs when trained on different subsets of data.

The LDS metric measures the Spearman rank correlation between:
1. Predicted model outputs (sum of attribution scores for subset)
2. Actual model outputs (from retraining on subset)

Usage:
    # Basic run with config
    python -m experiments.lds_analysis \\
        --config-name=lds_analysis \\
        --config-path=../configs
    
    # With specific models from training output
    python -m experiments.lds_analysis \\
        --config-name=lds_analysis \\
        --config-path=../configs \\
        +override_config=path/to/best_models.yaml
    
    # Override LDS parameters
    python -m experiments.lds_analysis \\
        --config-name=lds_analysis \\
        --config-path=../configs \\
        lds_analysis.num_subsets=500 \\
        lds_analysis.subset_fraction=0.5 \\
        +override_config=experiments/outputs/models/best_models.yaml

Notes:
    - Trains multiple models on random subsets (can be expensive)
    - For each test example, computes LDS across all attribution methods
    - Results saved as timestamped JSON with per-example and aggregate scores
"""

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import hydra
import jax.numpy as jnp
import numpy as np
from hydra.core.config_store import ConfigStore
from jax.random import PRNGKey
from jaxtyping import Array, Float
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

from experiments.utils import (
    cleanup_memory,
    load_experiment_override_from_yaml,
    to_dataclass,
)
from src.config import (
    LDSAnalysisConfig,
    LDSExperimentConfig,
    LossType,
    ModelConfig,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets
from src.utils.data.data import Dataset, DownloadableDataset
from src.utils.loss import get_loss
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import (
    evaluate_loss,
    load_model_checkpoint,
    train_model,
)

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="lds_experiment", node=LDSExperimentConfig)


def generate_random_subsets(
    dataset_size: int,
    num_subsets: int,
    subset_fraction: float,
    seed: int,
) -> List[np.ndarray]:
    """
    Generate random subsets of training data indices.

    Args:
        dataset_size: Total number of training examples
        num_subsets: Number of random subsets to generate
        subset_fraction: Fraction of dataset to include in each subset (α)
        seed: Random seed for reproducibility

    Returns:
        List of boolean arrays indicating which examples are in each subset
    """
    rng = np.random.RandomState(seed)
    subset_size = int(dataset_size * subset_fraction)

    subsets = []
    for _ in range(num_subsets):
        # Sample indices without replacement
        indices = rng.choice(dataset_size, size=subset_size, replace=False)
        # Create boolean mask
        mask = np.zeros(dataset_size, dtype=bool)
        mask[indices] = True
        subsets.append(mask)

    logger.info(
        f"Generated {num_subsets} random subsets "
        f"(each with {subset_size}/{dataset_size} examples)"
    )

    return subsets


def train_model_on_subset(
    model_config: ModelConfig,
    full_dataset: Dataset,
    subset_mask: np.ndarray,
    seed: int,
) -> Tuple[Any, Any, Any]:
    """
    Train a model on a subset of the training data.

    Args:
        model_config: Configuration for model architecture and training
        full_dataset: Complete training dataset
        subset_mask: Boolean mask indicating which examples to use
        seed: Random seed for training

    Returns:
        Tuple of (model, params, final_loss)
    """
    # Extract subset
    subset_inputs = full_dataset.inputs[subset_mask]
    subset_targets = full_dataset.targets[subset_mask]
    subset_dataset = full_dataset.__class__(subset_inputs, subset_targets)

    # Normalize for regression
    if model_config.loss == LossType.MSE:
        subset_inputs, _ = full_dataset.__class__.normalize_data(
            subset_inputs, subset_inputs
        )
        subset_targets, _ = full_dataset.__class__.normalize_data(
            subset_targets, subset_targets
        )
        subset_dataset = full_dataset.__class__(subset_inputs, subset_targets)

    # Create model
    model = ModelRegistry.get_model(model_config=model_config)

    # Train
    model, params, history = train_model(
        model=model,
        dataloader=subset_dataset.get_dataloader(
            batch_size=model_config.training.batch_size,
            seed=seed,
        ),
        loss_fn=get_loss(model_config.loss),
        optimizer=optimizer(
            optimizer_enum=model_config.training.optimizer,
            lr=model_config.training.learning_rate,
            weight_decay=model_config.training.weight_decay,
        ),
        epochs=model_config.training.epochs,
    )

    return model, params, history[-1] if history else None


def compute_attribution_scores(
    model,
    params,
    model_config: ModelConfig,
    train_dataset: Dataset,
    test_examples: Dataset,
    approximator,
    damping: float,
    model_directory: str,
    seed: int,
) -> Float[Array, "n_test n_train"]:
    """
    Compute attribution scores using a Hessian approximation method.

    This computes influence scores via IHVP: -H^{-1} * grad_test

    Args:
        model: Trained model
        params: Model parameters
        model_config: Model configuration
        train_dataset: Full training dataset
        test_examples: Test examples to compute attributions for
        approximator: Hessian approximation method
        damping: Damping parameter
        model_directory: Directory for caching
        seed: Random seed

    Returns:
        Attribution matrix of shape (n_test, n_train)
    """
    loss_fn = get_loss(model_config.loss)

    # Collect data for Hessian computation
    logger.info(f"[LDS] Collecting data for {approximator.value}")
    collector_dir = os.path.join(model_directory, "lds_collector", approximator.value)

    # Generate pseudo targets
    pseudo_targets = generate_pseudo_targets(
        model=model,
        inputs=train_dataset.inputs,
        params=params,
        loss_fn=loss_fn,
        rng_key=PRNGKey(seed),
    )
    collector_data_ds = train_dataset.replace_targets(pseudo_targets)

    # Collect activations and gradients
    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss_fn,
    )
    collected_data = collector.collect(
        inputs=collector_data_ds.inputs,
        targets=collector_data_ds.targets,
        save_directory=collector_dir,
        try_load=True,
    )

    # Create model context
    model_ctx = ModelContext.create(
        dataset=train_dataset,
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    # Get computer
    compute_context = HessianComputerRegistry.get_compute_context(
        approximator=approximator,
        collector_data=(collected_data, collected_data),
        model_ctx=model_ctx,
    )
    computer = HessianComputerRegistry.get_computer(approximator, compute_context)

    if isinstance(computer, HessianEstimator):
        computer.build(base_directory=model_directory)

    # Compute test gradients
    logger.info("[LDS] Computing test gradients")

    def loss_fn_single(p, x, y):
        pred = model.apply(p, x)
        return loss_fn(pred, y)

    from jax import grad, vmap

    grad_fn = vmap(lambda x, y: grad(lambda p: loss_fn_single(p, x, y))(params))
    test_grads = grad_fn(test_examples.inputs, test_examples.targets)

    # Flatten test gradients
    from jax.tree_util import tree_flatten, tree_unflatten

    test_grads_flat, tree_def = tree_flatten(test_grads)
    test_grads_flat = jnp.stack(
        [g.reshape(g.shape[0], -1) for g in test_grads_flat], axis=-1
    )
    test_grads_flat = test_grads_flat.reshape(test_grads_flat.shape[0], -1)

    # Compute IHVP for each test example
    logger.info(f"[LDS] Computing IHVPs for {len(test_examples.inputs)} test examples")
    ihvps = []

    for i in range(len(test_examples.inputs)):
        test_grad = test_grads_flat[i]

        # Unflatten to match parameter structure
        test_grad_tree = tree_unflatten(
            tree_def, [test_grad[..., j] for j in range(len(tree_def.num_leaves))]
        )

        # Compute IHVP
        if isinstance(computer, HessianEstimator):
            ihvp = computer.estimate_ihvp(test_grad_tree, damping)
        else:
            raise ValueError("Only HessianEstimator supported for LDS")

        ihvps.append(ihvp)

    # Compute attributions as dot product with training gradients
    logger.info("[LDS] Computing training gradients")
    train_grads = grad_fn(train_dataset.inputs, train_dataset.targets)
    train_grads_flat, _ = tree_flatten(train_grads)
    train_grads_flat = jnp.stack(
        [g.reshape(g.shape[0], -1) for g in train_grads_flat], axis=-1
    )
    train_grads_flat = train_grads_flat.reshape(train_grads_flat.shape[0], -1)

    # Compute attribution scores: -IHVP^T * train_grad
    attributions = np.zeros((len(test_examples.inputs), len(train_dataset.inputs)))

    for i, ihvp in enumerate(ihvps):
        ihvp_flat, _ = tree_flatten(ihvp)
        ihvp_flat = jnp.concatenate([g.flatten() for g in ihvp_flat])

        # Dot product with each training gradient
        for j in range(len(train_dataset.inputs)):
            attributions[i, j] = -float(jnp.dot(ihvp_flat, train_grads_flat[j]))

    cleanup_memory(f"attribution_{approximator.value}")

    return attributions


def compute_lds_for_model(
    model_directory: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    lds_config: LDSAnalysisConfig,
    seed: int,
) -> Dict:
    """
    Compute LDS scores for a single model across all approximation methods.

    Args:
        model_directory: Path to saved model
        train_dataset: Training data
        test_dataset: Test data for computing attributions
        lds_config: LDS configuration
        seed: Random seed

    Returns:
        Dictionary with LDS results
    """
    # Load base model
    params, model, model_config, metadata = load_model_checkpoint(model_directory)

    logger.info(f"{'=' * 70}")
    logger.info(f"[LDS] Analyzing: {model_config.get_model_display_name()}")
    logger.info(f"Model directory: {model_directory}")
    logger.info(f"{'=' * 70}")

    # Generate random subsets
    subsets = generate_random_subsets(
        dataset_size=len(train_dataset.inputs),
        num_subsets=lds_config.num_subsets,
        subset_fraction=lds_config.subset_fraction,
        seed=seed,
    )

    # Train models on each subset and evaluate on test examples
    logger.info(f"[LDS] Training {len(subsets)} models on subsets")
    actual_outputs = np.zeros((lds_config.num_test_examples, len(subsets)))

    for subset_idx, subset_mask in enumerate(subsets):
        if (subset_idx + 1) % 10 == 0:
            logger.info(f"[LDS] Training subset model {subset_idx + 1}/{len(subsets)}")

        subset_model, subset_params, _ = train_model_on_subset(
            model_config=model_config,
            full_dataset=train_dataset,
            subset_mask=subset_mask,
            seed=seed + subset_idx,
        )

        # Evaluate on test examples
        test_sample = Dataset(
            test_dataset.inputs[: lds_config.num_test_examples],
            test_dataset.targets[: lds_config.num_test_examples],
        )

        for test_idx in range(lds_config.num_test_examples):
            actual_outputs[test_idx, subset_idx] = evaluate_loss(
                model=subset_model,
                params=subset_params,
                inputs=test_sample.inputs[test_idx],
                targets=test_sample.targets[test_idx],
                loss_fn=get_loss(model_config.loss),
            )

        cleanup_memory(f"subset_model_{subset_idx}")

    # Compute damping
    logger.info("[LDS] Computing damping parameter")
    collector_dir = os.path.join(model_directory, "lds_collector", "ekfac_damping")
    pseudo_targets = generate_pseudo_targets(
        model=model,
        inputs=train_dataset.inputs,
        params=params,
        loss_fn=get_loss(model_config.loss),
        rng_key=PRNGKey(seed),
    )
    collector_data_ds = train_dataset.replace_targets(pseudo_targets)

    collector = CollectorActivationsGradients(
        model=model, params=params, loss_fn=get_loss(model_config.loss)
    )
    collected_data = collector.collect(
        inputs=collector_data_ds.inputs,
        targets=collector_data_ds.targets,
        save_directory=collector_dir,
        try_load=True,
    )

    ekfac_computer = EKFACComputer(compute_context=(collected_data, collected_data))
    ekfac_computer.build(base_directory=model_directory)
    damping = EKFACComputer.get_damping(
        ekfac_data=ekfac_computer.precomputed_data,
        damping_strategy=lds_config.damping_strategy,
        factor=lds_config.damping,
    )
    logger.info(f"[LDS] Using damping: {damping:.6f}")

    # Compute attribution scores for each method
    test_sample = Dataset(
        test_dataset.inputs[: lds_config.num_test_examples],
        test_dataset.targets[: lds_config.num_test_examples],
    )

    lds_scores = {}

    for approx in lds_config.approximators:
        logger.info(f"[LDS] Computing attributions with {approx.value}")

        attributions = compute_attribution_scores(
            model=model,
            params=params,
            model_config=model_config,
            train_dataset=train_dataset,
            test_examples=test_sample,
            approximator=approx,
            damping=damping,
            model_directory=model_directory,
            seed=seed,
        )

        # Compute predicted outputs using attribution scores
        # g_τ(z, S'; S) = Σ τ(z, S)_i for i in S'
        predicted_outputs = np.zeros((lds_config.num_test_examples, len(subsets)))

        for test_idx in range(lds_config.num_test_examples):
            for subset_idx, subset_mask in enumerate(subsets):
                # Sum attribution scores for examples in subset
                predicted_outputs[test_idx, subset_idx] = np.sum(
                    attributions[test_idx, subset_mask]
                )

        # Compute LDS (Spearman correlation) for each test example
        per_example_lds = []
        for test_idx in range(lds_config.num_test_examples):
            corr, _ = spearmanr(
                actual_outputs[test_idx, :],
                predicted_outputs[test_idx, :],
            )
            per_example_lds.append(float(corr) if not np.isnan(corr) else 0.0)

        # Average LDS across test examples
        mean_lds = float(np.mean(per_example_lds))

        logger.info(f"[LDS] {approx.value}: mean LDS = {mean_lds:.4f}")

        lds_scores[approx.value] = {
            "mean_lds": mean_lds,
            "per_example_lds": per_example_lds,
            "std_lds": float(np.std(per_example_lds)),
        }

        cleanup_memory(f"lds_{approx.value}")

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_directory,
        "model_config": asdict(model_config),
        "damping": float(damping),
        "num_subsets": len(subsets),
        "subset_fraction": lds_config.subset_fraction,
        "num_test_examples": lds_config.num_test_examples,
        "lds_scores": lds_scores,
        "metadata": metadata or {},
    }


@hydra.main(version_base="1.3", config_name="lds_experiment", config_path="../configs")
def main(cfg: DictConfig) -> Dict:
    OmegaConf.resolve(cfg)

    # Check if models should be loaded from file
    override_file = cfg.get("override_config", None)

    if override_file:
        logger.info(f"[CONFIG] Overriding config data from: {override_file}")
        model_directories, dataset_config, seed = load_experiment_override_from_yaml(
            override_file
        )

        cfg.models = model_directories
        if dataset_config is not None:
            cfg.dataset = asdict(dataset_config)
        if seed is not None:
            cfg.seed = seed

    config: LDSExperimentConfig = to_dataclass(LDSExperimentConfig, cfg)  # type: ignore

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info(f"{'=' * 70}")
    logger.info(f"Starting LDS Analysis: {config.experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Dataset: {config.dataset.name.value}")
    logger.info(f"Models to analyze: {len(config.models)}")
    logger.info(f"{'=' * 70}")

    # Load dataset
    full_dataset = DownloadableDataset.load(
        dataset=config.dataset.name,
        directory=config.dataset.path,
        store_on_disk=config.dataset.store_on_disk,
    )

    # Split into train and test
    train_dataset, test_dataset = full_dataset.train_test_split(
        test_size=config.dataset.test_size,
        seed=config.seed,
    )

    logger.info(
        f"Dataset split: {len(train_dataset.inputs)} train, "
        f"{len(test_dataset.inputs)} test"
    )

    # Run LDS analysis
    logger.info(f"{'#' * 70}")
    logger.info("LDS ANALYSIS")
    logger.info(f"{'#' * 70}")

    lds_results = []
    for i, model_dir in enumerate(config.models, 1):
        logger.info(f"[MODEL {i}/{len(config.models)}]")

        result = compute_lds_for_model(
            model_directory=model_dir,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            lds_config=config.lds_analysis,
            seed=config.seed,
        )
        lds_results.append(result)

        cleanup_memory(f"model_{i}")

    # Save results
    results_dir = config.lds_analysis.results_output_dir
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f"{timestamp}.json")

    full_results = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "lds_config": asdict(config.lds_analysis),
        "results": lds_results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"{'=' * 70}")
    logger.info("LDS Analysis Complete!")
    logger.info(f"Models analyzed: {len(lds_results)}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'=' * 70}")

    # Print summary
    logger.info(f"{'#' * 70}")
    logger.info("LDS SUMMARY")
    logger.info(f"{'#' * 70}")
    for result in lds_results:
        logger.info(f"\n{result['model_name']}:")
        for method, scores in result["lds_scores"].items():
            logger.info(
                f"  {method}: {scores['mean_lds']:.4f} ± {scores['std_lds']:.4f}"
            )

    return full_results


if __name__ == "__main__":
    main()
