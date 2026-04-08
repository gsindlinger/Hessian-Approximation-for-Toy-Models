from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from scipy.stats import spearmanr

from src.config import LDSExperimentConfig, ModelConfig
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import EKFACData, ModelContext
from src.utils.data.data import Dataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.influence import compute_influence_matrix, compute_per_example_flat_grads
from src.utils.loss import get_loss
from src.utils.optimizers import optimizer as create_optimizer
from src.utils.train import (
    evaluate_per_example_losses,
    load_model_checkpoint,
    train_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subset generation
# ---------------------------------------------------------------------------


def generate_random_subsets(
    dataset_size: int,
    num_subsets: int,
    subset_fraction: float,
    seed: int,
) -> List[np.ndarray]:
    r"""Generate K random subsets of training data as boolean masks.

    Each mask S_j selects ⌊\alpha·N⌋ examples without replacement. For ELSO
    retraining the *complement* D\\S_j is used as the training set.

    Returns:
        List of K boolean arrays of shape (dataset_size,). True = member of S_j.
    """
    rng = np.random.RandomState(seed)
    subset_size = int(dataset_size * subset_fraction)
    subsets: List[np.ndarray] = []
    for _ in range(num_subsets):
        indices = rng.choice(dataset_size, size=subset_size, replace=False)
        mask = np.zeros(dataset_size, dtype=bool)
        mask[indices] = True
        subsets.append(mask)
    logger.info(
        "Generated %d subsets (|S_j|=%d / N=%d, \\alpha=%.2f)",
        num_subsets,
        subset_size,
        dataset_size,
        subset_fraction,
    )
    return subsets


# ---------------------------------------------------------------------------
# ELSO ground-truth Δm computation
# ---------------------------------------------------------------------------


def compute_elso_ground_truth(
    model_config: ModelConfig,
    full_train_inputs: jnp.ndarray,
    full_train_targets: jnp.ndarray,
    query_inputs: jnp.ndarray,
    query_targets: jnp.ndarray,
    subsets: List[np.ndarray],
    reps_per_subset: int,
    baseline_losses: np.ndarray,
    base_seed: int,
) -> np.ndarray:
    """Compute ELSO ground-truth Δm_j(z_q) for all (query, subset) pairs.

    For each subset S_j, retrains `reps_per_subset` models on D\\S_j and
    averages their per-query losses to estimate:

        Δm_j(z_q) = E_ξ[m(z_q, θ(D\\S_j))] - m(z_q, θ(D))

    Returns:
        Δm array of shape (n_queries, K).
    """
    loss_fn = get_loss(model_config.loss)
    opt = create_optimizer(
        optimizer_enum=model_config.training.optimizer,
        lr=model_config.training.learning_rate,
        weight_decay=model_config.training.weight_decay,
    )

    n_queries = len(query_inputs)
    K = len(subsets)
    delta_m = np.zeros((n_queries, K))

    for j, mask in enumerate(subsets):
        complement_mask = ~mask
        comp_inputs = full_train_inputs[complement_mask]
        comp_targets = full_train_targets[complement_mask]

        if len(comp_inputs) == 0:
            logger.warning("Subset %d: complement is empty — skipping.", j)
            continue

        rep_losses = np.zeros((reps_per_subset, n_queries))

        for r in range(reps_per_subset):
            seed = base_seed + j * reps_per_subset + r
            dataloader = JAXDataLoader(
                comp_inputs,
                comp_targets,
                batch_size=model_config.training.batch_size,
                shuffle=True,
                rng_key=jax.random.PRNGKey(seed),
            )
            subset_model, subset_params, _ = train_model(
                model_config=model_config,
                dataloader=dataloader,
                loss_fn=loss_fn,
                optimizer=opt,
                epochs=model_config.training.epochs,
                seed=seed,
                save_checkpoints=False,
                verbose=False,
            )
            rep_losses[r] = np.array(
                evaluate_per_example_losses(
                    model=subset_model,
                    params=subset_params,
                    inputs=query_inputs,
                    targets=query_targets,
                    loss_fn=loss_fn,
                )
            )

        delta_m[:, j] = rep_losses.mean(axis=0) - baseline_losses

        if (j + 1) % max(1, K // 10) == 0 or j == 0:
            logger.info("[ELSO] Subset %d / %d done.", j + 1, K)

    return delta_m  # (n_queries, K)


# ---------------------------------------------------------------------------
# Group attribution prediction
# ---------------------------------------------------------------------------


def compute_group_attributions(
    attributions: np.ndarray,
    subsets: List[np.ndarray],
) -> np.ndarray:
    """Sum attribution scores over subset members for each (query, subset) pair.

    g_τ(z_q, S_j) = Σ_{z_i ∈ S_j} τ(z_q, z_i)

    Returns:
        Predicted group effects of shape (n_test, K).
    """
    n_test = attributions.shape[0]
    K = len(subsets)
    predicted = np.zeros((n_test, K))
    for j, mask in enumerate(subsets):
        predicted[:, j] = attributions[:, mask].sum(axis=1)
    return predicted


# ---------------------------------------------------------------------------
# LDS statistics with bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_spearman_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Spearman correlation with percentile bootstrap confidence interval.

    Returns:
        (spearman_r, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n = len(actual)

    point_est, _ = spearmanr(actual, predicted)
    point_est = float(point_est) if not np.isnan(point_est) else 0.0  # type: ignore

    bootstrap_rs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(actual[idx], predicted[idx])
        bootstrap_rs.append(float(r) if not np.isnan(r) else 0.0)  # type: ignore

    ci_low = float(np.percentile(bootstrap_rs, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_rs, 100 * (1 - alpha / 2)))
    return point_est, ci_low, ci_high


def aggregate_lds_scores(
    delta_m: np.ndarray,
    predicted: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Dict:
    """Compute per-query LDS with bootstrap CI, then aggregate across queries.

    Returns:
        Dict with keys: mean_lds, std_lds, per_query_lds,
        per_query_ci_low, per_query_ci_high.
    """
    n_queries = delta_m.shape[0]
    per_query_lds, per_query_ci_low, per_query_ci_high = [], [], []

    for i in range(n_queries):
        r, lo, hi = bootstrap_spearman_ci(
            delta_m[i], predicted[i], n_bootstrap=n_bootstrap, seed=seed
        )
        per_query_lds.append(r)
        per_query_ci_low.append(lo)
        per_query_ci_high.append(hi)

    return {
        "mean_lds": float(np.mean(per_query_lds)),
        "std_lds": float(np.std(per_query_lds)),
        "per_query_lds": per_query_lds,
        "per_query_ci_low": per_query_ci_low,
        "per_query_ci_high": per_query_ci_high,
    }


# ---------------------------------------------------------------------------
# Main pipeline per model
# ---------------------------------------------------------------------------


def compute_lds_for_model(
    model_directory: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: LDSExperimentConfig,
) -> Dict:
    """Compute ELSO LDS scores for a single pre-trained model.

    Steps:
      1. Load the base model (trained on full D).
      2. Generate K random subsets S_j.
      3. Evaluate baseline per-query losses from the full model.
      4. ELSO: for each S_j retrain R models on D\\S_j → Δm_j.
      5. Compute per-example gradients (train + query) using the full model.
      6. Build Hessian approximations; compute influence matrix τ.
      7. Predicted group effects g_τ(S_j) = Σ_{z∈S_j} τ(z_q, z).
      8. LDS = Spearman(Δm_j, g_τ_j) with 95% bootstrap CI, averaged over queries.
    """
    from experiments.utils import cleanup_memory

    params, model, model_config, metadata = load_model_checkpoint(model_directory)
    loss_fn = get_loss(model_config.loss)

    logger.info("=" * 70)
    logger.info("[LDS] %s", model_config.get_model_display_name())
    logger.info("[LDS] dir: %s", model_directory)
    logger.info("=" * 70)

    n_query = min(config.num_test_examples, len(test_dataset.inputs))
    query_inputs = test_dataset.inputs[:n_query]
    query_targets = test_dataset.targets[:n_query]

    # ── 1. Subsets ──────────────────────────────────────────────────────────
    subsets = generate_random_subsets(
        dataset_size=len(train_dataset.inputs),
        num_subsets=config.num_subsets,
        subset_fraction=config.subset_fraction,
        seed=config.seed,
    )

    # ── 2. Baseline losses from full model ──────────────────────────────────
    logger.info("[LDS] Computing baseline losses from full model.")
    baseline_losses = np.array(
        evaluate_per_example_losses(
            model=model,
            params=params,
            inputs=query_inputs,
            targets=query_targets,
            loss_fn=loss_fn,
        )
    )

    # ── 3. ELSO retraining ──────────────────────────────────────────────────
    logger.info(
        "[LDS] ELSO retraining: K=%d subsets × R=%d reps on D\\S_j.",
        len(subsets),
        config.reps_per_model,
    )
    delta_m = compute_elso_ground_truth(
        model_config=model_config,
        full_train_inputs=train_dataset.inputs,
        full_train_targets=train_dataset.targets,
        query_inputs=query_inputs,
        query_targets=query_targets,
        subsets=subsets,
        reps_per_subset=config.reps_per_model,
        baseline_losses=baseline_losses,
        base_seed=config.seed,
    )
    cleanup_memory("elso_retraining")

    # ── 4. Hessian / damping setup ──────────────────────────────────────────
    logger.info("[LDS] Building EKFAC for damping estimation.")
    collector_dir_base = os.path.join(model_directory, "lds_collector")

    # MCMC with 2 repetitions: EKFACComputer splits them internally (first half
    # for covariances, second half for eigenvalue corrections).
    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss_fn,
        pseudo_target_repetitions=config.hessian_estimators.pseudo_target_generation_repetitions,
        pseudo_target_strategy=config.hessian_estimators.pseudo_target_generation_strategy,
    )

    collector_data = collector.collect(
        dataset=train_dataset,
        rng_key=PRNGKey(config.seed),
        save_directory=os.path.join(collector_dir_base, "run1"),
        try_load=True,
    )

    ekfac_computer = EKFACComputer(compute_context=collector_data).build(
        base_directory=model_directory
    )
    assert isinstance(ekfac_computer.precomputed_data, EKFACData), (
        "EKFAC precomputation failed."
    )
    damping = EKFACComputer.get_damping(
        ekfac_data=ekfac_computer.precomputed_data,
        damping_strategy=config.hessian_estimators.regularization_strategy,
        factor=config.hessian_estimators.regularization_value,
    )
    logger.info("[LDS] Damping: %.6f", damping)

    model_ctx = ModelContext.create(
        dataset=train_dataset,
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    # ── 5. Per-example gradients ────────────────────────────────────────────
    logger.info(
        "[LDS] Computing per-example gradients (%d train, %d queries).",
        len(train_dataset.inputs),
        n_query,
    )
    train_flat_grads = compute_per_example_flat_grads(
        model=model,
        params=params,
        inputs=train_dataset.inputs,
        targets=train_dataset.targets,
        loss_fn=loss_fn,
    )
    test_flat_grads = compute_per_example_flat_grads(
        model=model,
        params=params,
        inputs=query_inputs,
        targets=query_targets,
        loss_fn=loss_fn,
    )
    cleanup_memory("gradient_computation")

    # ── 6. LDS per approximation method ────────────────────────────────────
    lds_scores: Dict = {}

    for approx in config.hessian_estimators.approximators:
        logger.info("[LDS] Attribution method: %s", approx.value)

        compute_ctx = HessianComputerRegistry.get_compute_context(
            approximator=approx,
            collector_data=collector_data,
            model_ctx=model_ctx,
        )
        computer = HessianComputerRegistry.get_computer(approx, compute_ctx)

        if not isinstance(computer, (HessianEstimator, HessianComputer)):
            logger.warning("Skipping %s: no IHVP support.", approx.value)
            continue

        if isinstance(computer, HessianEstimator):
            computer.build(base_directory=model_directory)

        attributions = compute_influence_matrix(
            test_flat_grads=test_flat_grads,
            train_flat_grads=train_flat_grads,
            computer=computer,
            damping=damping,
        )

        predicted = compute_group_attributions(attributions, subsets)

        scores = aggregate_lds_scores(
            delta_m=delta_m,
            predicted=predicted,
            n_bootstrap=1000,
            seed=config.seed,
        )

        mean_ci_lo = float(np.mean(scores["per_query_ci_low"]))
        mean_ci_hi = float(np.mean(scores["per_query_ci_high"]))
        logger.info(
            "[LDS] %s: mean_LDS=%.4f ± %.4f  (95%% CI: [%.4f, %.4f])",
            approx.value,
            scores["mean_lds"],
            scores["std_lds"],
            mean_ci_lo,
            mean_ci_hi,
        )

        lds_scores[approx.value] = scores
        cleanup_memory(f"lds_{approx.value}")

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_directory,
        "model_config": asdict(model_config),
        "damping": float(damping),
        "num_subsets": len(subsets),
        "subset_fraction": config.subset_fraction,
        "reps_per_model": config.reps_per_model,
        "num_queries": n_query,
        "lds_scores": lds_scores,
        "metadata": metadata or {},
    }
