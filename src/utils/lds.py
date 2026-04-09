from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import PRNGKey
from scipy.stats import spearmanr

from src.config import LDSExperimentConfig, ModelConfig
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import DataActivationsGradients, EKFACData, ModelContext
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
from src.utils.utils import collector_cache_dir

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


def _build_vmapped_elso_fn(
    model,
    optimizer,
    loss_fn,
    epoch_checkpoints: List[int],
    batch_size: int,
):
    """Return a JIT+vmap-compiled function:

        fn(keys, comp_x, comp_y, query_x, query_y)
            -> rep_losses  shape (R, n_checkpoints, n_queries)

    Trains R models in parallel (one per key) on the **same** complement
    dataset.  Training runs to ``max(epoch_checkpoints)`` epochs total;
    params are snapshotted and evaluated on the query set at each checkpoint
    epoch in a single pass.  This means that a sweep over epoch checkpoints
    [10, 100, 1000] requires only one ELSO retraining instead of three.

    ``jax.lax.scan`` replaces the Python epoch/batch loops, so there is no
    per-step Python dispatch overhead.  On GPU with a small model this gives
    ~R× speedup because Python overhead is paid once per subset instead of
    once per rep.

    The function is compiled on the first call and reused for all subsequent
    subsets (shapes are fixed: all complements have the same size).
    """
    checkpoints = sorted(epoch_checkpoints)
    max_epochs = checkpoints[-1]

    # Lengths of each training segment between consecutive checkpoints.
    # Python loop at trace time → unrolled into separate lax.scan calls.
    seg_lengths = [checkpoints[0]] + [
        checkpoints[i] - checkpoints[i - 1] for i in range(1, len(checkpoints))
    ]

    def _eval_queries(p, query_x, query_y):
        """Evaluate p on all query examples; returns (n_queries,)."""

        def single(x, y):
            return loss_fn(model.apply(p, x[None]), jnp.atleast_1d(y))

        return jax.vmap(single)(query_x, query_y)

    def train_and_eval_one(key, comp_x, comp_y, query_x, query_y):
        n = comp_x.shape[0]
        n_batches = n // batch_size
        n_keep = n_batches * batch_size

        params = model.init(key, comp_x[:1])
        opt_state = optimizer.init(params)

        # Pre-generate all epoch keys up-front so the training trajectory is
        # identical to a single max_epochs run (consistent with a sequential run).
        all_epoch_keys = jax.random.split(key, max_epochs)

        def epoch_step(carry, epoch_key):
            params, opt_state = carry
            perm = jax.random.permutation(epoch_key, n_keep)
            x_s = comp_x[perm].reshape(n_batches, batch_size, -1)
            y_s = comp_y[perm].reshape(n_batches, batch_size)

            def batch_step(carry, batch):
                params, opt_state = carry
                x, y = batch

                def _loss(p):
                    return loss_fn(model.apply(p, x), y)

                grads = jax.grad(_loss)(params)
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), None

            (params, opt_state), _ = jax.lax.scan(
                batch_step, (params, opt_state), (x_s, y_s)
            )
            return (params, opt_state), None

        # Train through each segment and evaluate at the checkpoint boundary.
        # Pass params explicitly to _eval_queries to avoid Python closure issues.
        checkpoint_losses = []
        start = 0
        for seg_len in seg_lengths:
            seg_keys = all_epoch_keys[start : start + seg_len]
            (params, opt_state), _ = jax.lax.scan(
                epoch_step, (params, opt_state), seg_keys
            )
            checkpoint_losses.append(_eval_queries(params, query_x, query_y))
            start += seg_len

        return jnp.stack(checkpoint_losses)  # (n_checkpoints, n_queries)

    # vmap over keys; complement data and queries are broadcast (shared) across reps
    return jax.jit(jax.vmap(train_and_eval_one, in_axes=(0, None, None, None, None)))


def compute_elso_ground_truth(
    model_config: ModelConfig,
    full_train_inputs: jnp.ndarray,
    full_train_targets: jnp.ndarray,
    query_inputs: jnp.ndarray,
    query_targets: jnp.ndarray,
    subsets: List[np.ndarray],
    reps_per_subset: int,
    baseline_losses: "np.ndarray | Dict[int, np.ndarray]",
    base_seed: int,
    epoch_checkpoints: Optional[List[int]] = None,
    use_vmap: bool = True,
) -> "np.ndarray | Dict[int, np.ndarray]":
    """Compute ELSO ground-truth Δm_j(z_q) for all (query, subset) pairs.

    For each subset S_j, retrains `reps_per_subset` models on D\\S_j and
    averages their per-query losses to estimate:

        Δm_j(z_q) = E_ξ[m(z_q, θ(D\\S_j))] - m(z_q, θ(D))

    Args:
        baseline_losses: For a single checkpoint, a ``(n_queries,)`` array.
            For multiple checkpoints, a dict ``{epoch: (n_queries,) array}``.
        epoch_checkpoints: Epoch numbers at which to evaluate query losses.
            Defaults to ``[model_config.training.epochs]``.  Providing
            multiple values (e.g. ``[10, 100, 1000]``) trains each complement
            model once to the maximum epoch and snapshots params at each
            checkpoint, reducing ELSO cost by ``len(epoch_checkpoints)×``.
        use_vmap: When True (default), uses jax.lax.scan + jax.vmap to train
            all R reps in a single JIT call.  Recommended on GPU.

    Returns:
        Single checkpoint: Δm array of shape ``(n_queries, K)``.
        Multiple checkpoints: dict ``{epoch: (n_queries, K) array}``.
    """
    checkpoints = sorted(
        epoch_checkpoints if epoch_checkpoints else [model_config.training.epochs]
    )
    multi_epoch = len(checkpoints) > 1

    loss_fn = get_loss(model_config.loss)
    opt = create_optimizer(
        optimizer_enum=model_config.training.optimizer,
        lr=model_config.training.learning_rate,
        weight_decay=model_config.training.weight_decay,
    )

    n_queries = len(query_inputs)
    K = len(subsets)
    # (n_checkpoints, n_queries, K) — accumulates R-averaged losses
    rep_mean = np.zeros((len(checkpoints), n_queries, K))

    if use_vmap:
        from src.utils.models.registry import ModelRegistry

        model = ModelRegistry.get_model(model_config=model_config)
        vmapped_fn = _build_vmapped_elso_fn(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            epoch_checkpoints=checkpoints,
            batch_size=model_config.training.batch_size,
        )
        logger.info(
            "[ELSO] vmap R=%d reps, checkpoints=%s (lax.scan, compiling on first subset).",
            reps_per_subset,
            checkpoints,
        )

        for j, mask in enumerate(subsets):
            comp_x = jnp.asarray(full_train_inputs[~mask])
            comp_y = jnp.asarray(full_train_targets[~mask])

            if len(comp_x) == 0:
                logger.warning("Subset %d: complement is empty — skipping.", j)
                continue

            keys = jax.random.split(
                PRNGKey(base_seed + j * reps_per_subset), reps_per_subset
            )
            # rep_losses: (R, n_checkpoints, n_queries)
            rep_losses = vmapped_fn(keys, comp_x, comp_y, query_inputs, query_targets)
            rep_mean[:, :, j] = np.array(jnp.mean(rep_losses, axis=0))

            if (j + 1) % max(1, K // 10) == 0 or j == 0:
                logger.info("[ELSO] Subset %d / %d done.", j + 1, K)

    else:
        # Sequential fallback: one train_model call per (checkpoint, rep, subset).
        for j, mask in enumerate(subsets):
            comp_inputs = full_train_inputs[~mask]
            comp_targets = full_train_targets[~mask]

            if len(comp_inputs) == 0:
                logger.warning("Subset %d: complement is empty — skipping.", j)
                continue

            for ci, ckpt in enumerate(checkpoints):
                ckpt_rep_losses = np.zeros((reps_per_subset, n_queries))
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
                        epochs=ckpt,
                        seed=seed,
                        save_checkpoints=False,
                        verbose=False,
                    )
                    ckpt_rep_losses[r] = np.array(
                        evaluate_per_example_losses(
                            model=subset_model,
                            params=subset_params,
                            inputs=query_inputs,
                            targets=query_targets,
                            loss_fn=loss_fn,
                        )
                    )
                rep_mean[ci, :, j] = ckpt_rep_losses.mean(axis=0)

            if (j + 1) % max(1, K // 10) == 0 or j == 0:
                logger.info("[ELSO] Subset %d / %d done.", j + 1, K)

    # Subtract per-epoch baselines and return
    if multi_epoch:
        assert isinstance(baseline_losses, dict), (
            "baseline_losses must be Dict[int, np.ndarray] when epoch_checkpoints has multiple entries."
        )
        return {
            ckpt: rep_mean[ci] - baseline_losses[ckpt][:, None]
            for ci, ckpt in enumerate(checkpoints)
        }
    else:
        assert isinstance(baseline_losses, np.ndarray)
        return rep_mean[0] - baseline_losses[:, None]  # (n_queries, K)


def compute_expected_baseline_losses(
    model_config: ModelConfig,
    full_train_inputs: jnp.ndarray,
    full_train_targets: jnp.ndarray,
    query_inputs: jnp.ndarray,
    query_targets: jnp.ndarray,
    baseline_reps: int,
    base_seed: int,
    epoch_checkpoints: Optional[List[int]] = None,
    use_vmap: bool = True,
) -> "np.ndarray | Dict[int, np.ndarray]":
    """Estimate E_ξ[m(z_q, θ(D; ξ))] by retraining on the full dataset."""
    checkpoints = sorted(
        epoch_checkpoints if epoch_checkpoints else [model_config.training.epochs]
    )
    multi_epoch = len(checkpoints) > 1
    n_queries = len(query_inputs)

    loss_fn = get_loss(model_config.loss)
    opt = create_optimizer(
        optimizer_enum=model_config.training.optimizer,
        lr=model_config.training.learning_rate,
        weight_decay=model_config.training.weight_decay,
    )

    if use_vmap:
        from src.utils.models.registry import ModelRegistry

        model = ModelRegistry.get_model(model_config=model_config)
        vmapped_fn = _build_vmapped_elso_fn(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            epoch_checkpoints=checkpoints,
            batch_size=model_config.training.batch_size,
        )
        keys = jax.random.split(PRNGKey(base_seed), baseline_reps)
        rep_losses = vmapped_fn(
            keys,
            jnp.asarray(full_train_inputs),
            jnp.asarray(full_train_targets),
            query_inputs,
            query_targets,
        )
        rep_mean = np.array(jnp.mean(rep_losses, axis=0))
    else:
        rep_mean = np.zeros((len(checkpoints), n_queries))
        for ci, ckpt in enumerate(checkpoints):
            ckpt_rep_losses = np.zeros((baseline_reps, n_queries))
            for r in range(baseline_reps):
                seed = base_seed + r
                dataloader = JAXDataLoader(
                    full_train_inputs,
                    full_train_targets,
                    batch_size=model_config.training.batch_size,
                    shuffle=True,
                    rng_key=jax.random.PRNGKey(seed),
                )
                base_model, base_params, _ = train_model(
                    model_config=model_config,
                    dataloader=dataloader,
                    loss_fn=loss_fn,
                    optimizer=opt,
                    epochs=ckpt,
                    seed=seed,
                    save_checkpoints=False,
                    verbose=False,
                )
                ckpt_rep_losses[r] = np.array(
                    evaluate_per_example_losses(
                        model=base_model,
                        params=base_params,
                        inputs=query_inputs,
                        targets=query_targets,
                        loss_fn=loss_fn,
                    )
                )
            rep_mean[ci] = ckpt_rep_losses.mean(axis=0)

    if multi_epoch:
        return {ckpt: rep_mean[ci] for ci, ckpt in enumerate(checkpoints)}
    return rep_mean[0]


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
    """Compute mean LDS across queries with a bootstrap CI over queries.

    Returns:
        Dict with keys: mean_lds, std_lds, ci_low, ci_high, per_query_lds.
    """
    rng = np.random.RandomState(seed)
    n_queries = delta_m.shape[0]
    per_query_lds = []

    for i in range(n_queries):
        r, _ = spearmanr(delta_m[i], predicted[i])
        r = float(r) if not np.isnan(r) else 0.0  # type: ignore
        per_query_lds.append(r)

    per_query_lds_arr = np.asarray(per_query_lds)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_queries, size=n_queries, replace=True)
        bootstrap_means.append(float(np.mean(per_query_lds_arr[idx])))

    ci_low = float(np.percentile(bootstrap_means, 2.5))
    ci_high = float(np.percentile(bootstrap_means, 97.5))

    return {
        "mean_lds": float(np.mean(per_query_lds_arr)),
        "std_lds": float(np.std(per_query_lds_arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "per_query_lds": per_query_lds,
    }


# ---------------------------------------------------------------------------
# Main pipeline per model
# ---------------------------------------------------------------------------


def compute_lds_for_model(
    model_directory: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: LDSExperimentConfig,
    epoch: Optional[int] = None,
) -> Dict:
    """Compute ELSO LDS scores for a single pre-trained model.

    Steps:
      1. Load the base model (trained on full D), optionally at a specific epoch.
      2. Generate K random subsets S_j.
      3. Evaluate baseline per-query losses from the full model.
      4. ELSO: for each S_j retrain R models on D\\S_j → Δm_j.
      5. Compute per-example gradients (train + query) using the full model.
      6. Build Hessian approximations; compute influence matrix τ.
      7. Predicted group effects g_τ(S_j) = Σ_{z∈S_j} τ(z_q, z).
      8. LDS = Spearman(Δm_j, g_τ_j) with 95% bootstrap CI, averaged over queries.
    """
    from experiments.utils import cleanup_memory

    params, model, model_config, metadata = load_model_checkpoint(
        model_directory, epoch=epoch
    )
    loss_fn = get_loss(model_config.loss)

    epoch_str = f"epoch_{epoch}" if epoch is not None else "final"
    logger.info("=" * 70)
    logger.info("[LDS] %s (%s)", model_config.get_model_display_name(), epoch_str)
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
    if config.baseline_reps is not None:
        logger.info(
            "[LDS] Computing expected baseline losses with %d retraining reps on full D.",
            config.baseline_reps,
        )
        baseline_losses = compute_expected_baseline_losses(
            model_config=model_config,
            full_train_inputs=train_dataset.inputs,
            full_train_targets=train_dataset.targets,
            query_inputs=query_inputs,
            query_targets=query_targets,
            baseline_reps=config.baseline_reps,
            base_seed=config.seed,
        )
        assert isinstance(baseline_losses, np.ndarray)
    else:
        logger.info(
            "[LDS] Computing baseline losses from loaded full-model checkpoint."
        )
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
    assert isinstance(delta_m, np.ndarray)
    cleanup_memory("elso_retraining")

    # ── 4. Hessian / damping setup ──────────────────────────────────────────
    logger.info("[LDS] Building EKFAC for damping estimation.")
    collector_dir_base = collector_cache_dir(
        model_directory=model_directory,
        pseudo_target_strategy=config.hessian_estimators.pseudo_target_generation_strategy.value,
        pseudo_target_repetitions=config.hessian_estimators.pseudo_target_generation_repetitions,
        epoch=epoch,
    )

    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss_fn,
        pseudo_target_repetitions=config.hessian_estimators.pseudo_target_generation_repetitions,
        pseudo_target_strategy=config.hessian_estimators.pseudo_target_generation_strategy,
    )

    collector_data: DataActivationsGradients = collector.collect(
        dataset=train_dataset,
        save_directory=collector_dir_base,
        try_load=True,
        rng_key=PRNGKey(config.seed),
    )

    ekfac_computer = EKFACComputer(compute_context=collector_data).build(
        base_directory=collector_dir_base
    )
    assert isinstance(ekfac_computer.precomputed_data, EKFACData), (
        "EKFAC precomputation failed."
    )
    from experiments.utils import resolve_regularization

    damping, pseudo_inverse_factor = resolve_regularization(
        strategy=config.hessian_estimators.regularization_strategy,
        factor=config.hessian_estimators.regularization_value,
        ekfac_data=ekfac_computer.precomputed_data,
    )
    logger.info(
        "[LDS] Regularization: damping=%.6f, pseudo_inverse_factor=%s",
        damping or 0.0,
        pseudo_inverse_factor,
    )

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
            computer.build(base_directory=collector_dir_base)

        attributions = compute_influence_matrix(
            test_flat_grads=test_flat_grads,
            train_flat_grads=train_flat_grads,
            computer=computer,
            damping=damping,
            pseudo_inverse_factor=pseudo_inverse_factor,
        )

        predicted = compute_group_attributions(attributions, subsets)  # type: ignore

        scores = aggregate_lds_scores(
            delta_m=delta_m,
            predicted=predicted,
            n_bootstrap=1000,
            seed=config.seed,
        )

        logger.info(
            "[LDS] %s: mean_LDS=%.4f ± %.4f  (95%% CI: [%.4f, %.4f])",
            approx.value,
            scores["mean_lds"],
            scores["std_lds"],
            scores["ci_low"],
            scores["ci_high"],
        )

        lds_scores[approx.value] = scores
        cleanup_memory(f"lds_{approx.value}")

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_directory,
        "epoch": epoch,
        "model_config": asdict(model_config),
        "damping": float(damping) if damping is not None else None,
        "num_subsets": len(subsets),
        "subset_fraction": config.subset_fraction,
        "reps_per_model": config.reps_per_model,
        "baseline_reps": config.baseline_reps,
        "num_queries": n_query,
        "lds_scores": lds_scores,
        "metadata": metadata or {},
    }


def compute_lds_for_model_multi_epoch(
    model_directory: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: LDSExperimentConfig,
    epochs: List[int],
) -> List[Dict]:
    """Compute ELSO LDS scores for one model across multiple epoch checkpoints
    in a single ELSO retraining pass.

    Compared to calling ``compute_lds_for_model`` once per epoch, this reduces
    ELSO retraining cost by ``len(epochs)×`` because complement models are
    trained once to ``max(epochs)`` and their params are snapshotted at each
    checkpoint rather than being retrained from scratch per epoch.

    Hessian approximations and gradient computations are still done separately
    per epoch (they depend on the base model checkpoint).

    Returns:
        A list of result dicts in the same format as ``compute_lds_for_model``,
        one entry per epoch in ``epochs``.
    """
    from experiments.utils import cleanup_memory

    sorted_epochs = sorted(epochs)

    # Load all epoch checkpoints and compute per-epoch baselines up-front.
    epoch_data: Dict[int, Dict] = {}
    for ep in sorted_epochs:
        params_ep, model_ep, model_config_ep, metadata_ep = load_model_checkpoint(
            model_directory, epoch=ep
        )
        loss_fn_ep = get_loss(model_config_ep.loss)
        n_query = min(config.num_test_examples, len(test_dataset.inputs))
        query_inputs = test_dataset.inputs[:n_query]
        query_targets = test_dataset.targets[:n_query]
        epoch_data[ep] = {
            "params": params_ep,
            "model": model_ep,
            "model_config": model_config_ep,
            "metadata": metadata_ep,
            "loss_fn": loss_fn_ep,
        }
        if config.baseline_reps is not None:
            epoch_data[ep]["baseline_losses"] = None
        else:
            epoch_data[ep]["baseline_losses"] = np.array(
                evaluate_per_example_losses(
                    model=model_ep,
                    params=params_ep,
                    inputs=query_inputs,
                    targets=query_targets,
                    loss_fn=loss_fn_ep,
                )
            )

    # Use model_config from the final epoch (determines ELSO training depth).
    model_config = epoch_data[sorted_epochs[-1]]["model_config"]
    n_query = min(config.num_test_examples, len(test_dataset.inputs))
    query_inputs = test_dataset.inputs[:n_query]
    query_targets = test_dataset.targets[:n_query]

    logger.info("=" * 70)
    logger.info(
        "[LDS-multi] %s  epochs=%s",
        model_config.get_model_display_name(),
        sorted_epochs,
    )
    logger.info("[LDS-multi] dir: %s", model_directory)
    logger.info("=" * 70)

    subsets = generate_random_subsets(
        dataset_size=len(train_dataset.inputs),
        num_subsets=config.num_subsets,
        subset_fraction=config.subset_fraction,
        seed=config.seed,
    )

    # Single ELSO pass — snapshots params at every epoch checkpoint.
    logger.info(
        "[LDS-multi] ELSO retraining: K=%d subsets × R=%d reps, checkpoints=%s.",
        len(subsets),
        config.reps_per_model,
        sorted_epochs,
    )
    if config.baseline_reps is not None:
        logger.info(
            "[LDS-multi] Computing expected baseline losses with %d retraining reps on full D.",
            config.baseline_reps,
        )
        baseline_losses_dict = compute_expected_baseline_losses(
            model_config=model_config,
            full_train_inputs=train_dataset.inputs,
            full_train_targets=train_dataset.targets,
            query_inputs=query_inputs,
            query_targets=query_targets,
            baseline_reps=config.baseline_reps,
            base_seed=config.seed,
            epoch_checkpoints=sorted_epochs,
        )
        assert isinstance(baseline_losses_dict, dict)
    else:
        baseline_losses_dict = {
            ep: epoch_data[ep]["baseline_losses"] for ep in sorted_epochs
        }
    delta_m_dict = compute_elso_ground_truth(
        model_config=model_config,
        full_train_inputs=train_dataset.inputs,
        full_train_targets=train_dataset.targets,
        query_inputs=query_inputs,
        query_targets=query_targets,
        subsets=subsets,
        reps_per_subset=config.reps_per_model,
        baseline_losses=baseline_losses_dict,
        base_seed=config.seed,
        epoch_checkpoints=sorted_epochs,
    )
    assert isinstance(delta_m_dict, dict)
    cleanup_memory("elso_retraining_multi")

    results = []
    for ep in sorted_epochs:
        ep_params = epoch_data[ep]["params"]
        ep_model = epoch_data[ep]["model"]
        ep_model_config = epoch_data[ep]["model_config"]
        ep_loss_fn = epoch_data[ep]["loss_fn"]
        ep_metadata = epoch_data[ep]["metadata"]
        delta_m = delta_m_dict[ep]

        collector_dir_base = collector_cache_dir(
            model_directory=model_directory,
            pseudo_target_strategy=config.hessian_estimators.pseudo_target_generation_strategy.value,
            pseudo_target_repetitions=config.hessian_estimators.pseudo_target_generation_repetitions,
            epoch=ep,
        )
        collector = CollectorActivationsGradients(
            model=ep_model,
            params=ep_params,
            loss_fn=ep_loss_fn,
            pseudo_target_repetitions=config.hessian_estimators.pseudo_target_generation_repetitions,
            pseudo_target_strategy=config.hessian_estimators.pseudo_target_generation_strategy,
        )
        collector_data: DataActivationsGradients = collector.collect(
            dataset=train_dataset,
            save_directory=collector_dir_base,
            try_load=True,
            rng_key=PRNGKey(config.seed),
        )
        ekfac_computer = EKFACComputer(compute_context=collector_data).build(
            base_directory=collector_dir_base
        )
        assert isinstance(ekfac_computer.precomputed_data, EKFACData)
        from experiments.utils import resolve_regularization

        damping, pseudo_inverse_factor = resolve_regularization(
            strategy=config.hessian_estimators.regularization_strategy,
            factor=config.hessian_estimators.regularization_value,
            ekfac_data=ekfac_computer.precomputed_data,
        )

        model_ctx = ModelContext.create(
            dataset=train_dataset,
            model=ep_model,
            params=ep_params,
            loss_fn=ep_loss_fn,
        )
        train_flat_grads = compute_per_example_flat_grads(
            model=ep_model,
            params=ep_params,
            inputs=train_dataset.inputs,
            targets=train_dataset.targets,
            loss_fn=ep_loss_fn,
        )
        test_flat_grads = compute_per_example_flat_grads(
            model=ep_model,
            params=ep_params,
            inputs=query_inputs,
            targets=query_targets,
            loss_fn=ep_loss_fn,
        )
        cleanup_memory(f"gradients_epoch_{ep}")

        lds_scores: Dict = {}
        for approx in config.hessian_estimators.approximators:
            compute_ctx = HessianComputerRegistry.get_compute_context(
                approximator=approx,
                collector_data=collector_data,
                model_ctx=model_ctx,
            )
            computer = HessianComputerRegistry.get_computer(approx, compute_ctx)
            if not isinstance(computer, (HessianEstimator, HessianComputer)):
                continue
            if isinstance(computer, HessianEstimator):
                computer.build(base_directory=collector_dir_base)

            attributions = compute_influence_matrix(
                test_flat_grads=test_flat_grads,
                train_flat_grads=train_flat_grads,
                computer=computer,
                damping=damping,
                pseudo_inverse_factor=pseudo_inverse_factor,
            )
            predicted = compute_group_attributions(attributions, subsets)  # type: ignore
            scores = aggregate_lds_scores(
                delta_m=delta_m,
                predicted=predicted,
                n_bootstrap=1000,
                seed=config.seed,
            )
            lds_scores[approx.value] = scores
            cleanup_memory(f"lds_{approx.value}_epoch_{ep}")

        results.append(
            {
                "model_name": ep_model_config.get_model_display_name(),
                "model_directory": model_directory,
                "epoch": ep,
                "model_config": asdict(ep_model_config),
                "damping": float(damping) if damping is not None else None,
                "num_subsets": len(subsets),
                "subset_fraction": config.subset_fraction,
                "reps_per_model": config.reps_per_model,
                "baseline_reps": config.baseline_reps,
                "num_queries": n_query,
                "lds_scores": lds_scores,
                "metadata": ep_metadata or {},
            }
        )
        cleanup_memory(f"epoch_{ep}")

    return results
