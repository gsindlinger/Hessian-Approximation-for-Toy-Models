"""LDS (Linear Datamodeling Score) via ELSO retraining.

Evaluates a precomputed data-attribution matrix against the counterfactual
effect of *removing* groups of training examples from D (Bae et al. 2022):

    Ground truth:  Δm_j(z_q) = E_ξ[m(z_q, θ(D\\S_j))] - m(z_q, θ(D))
    Predicted:     g_τ(z_q, S_j) = Σ_{z_i ∈ S_j} τ(z_q, z_i, D)
    LDS:           Spearman({Δm_j}, {g_τ_j}) averaged over query points

This module is purposely scoped to the *evaluation* half of the pipeline:
attribution scores are passed in as a `(n_query, n_train)` numpy array
(typically loaded from an .npy file produced by a separate attribution step).
The Hessian / collector / influence-matrix construction lives in the calling
script.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from math import ceil
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import PRNGKey
from scipy.stats import spearmanr

from src.config import LDSConfig, ModelConfig
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import get_loss
from src.utils.optimizers import optimizer as create_optimizer
from src.utils.train import (
    evaluate_per_example_losses,
    load_model_checkpoint,
    load_train_test_for_model,
    train_model,
)
from src.utils.utils import elso_cache_dir

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
        "Generated %d subsets (|S_j|=%d / N=%d, alpha=%.2f)",
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
    epochs: int,
    batch_size: int,
):
    """Return a JIT+vmap-compiled function:

        fn(keys, comp_x, comp_y, query_x, query_y) -> (R, n_queries) per-query losses

    Trains R models in parallel (one per key) on the **same** complement
    dataset for ``epochs`` epochs and evaluates each on the query set.
    """

    def _eval_queries(p, query_x, query_y):
        def single(x, y):
            return loss_fn(model.apply(p, x[None]), y[None, ...])

        return jax.vmap(single)(query_x, query_y)

    def train_and_eval_one(key, comp_x, comp_y, query_x, query_y):
        init_key, shuffle_key = jax.random.split(key, 2)
        n = comp_x.shape[0]
        # Use ceil so every example is seen each epoch. The trailing slots in
        # the final batch are filled with index 0 and zeroed out via `mask` so
        # they do not contribute to the gradient.
        n_batches = -(-n // batch_size)
        n_padded = n_batches * batch_size
        pad = n_padded - n

        params = model.init(init_key, comp_x[:1])
        opt_state = optimizer.init(params)

        all_epoch_keys = jax.random.split(shuffle_key, epochs)

        def epoch_step(carry, epoch_key):
            params, opt_state = carry
            perm = jax.random.permutation(epoch_key, n)
            perm_padded = jnp.concatenate(
                [perm, jnp.zeros(pad, dtype=perm.dtype)]
            )
            valid = jnp.concatenate(
                [jnp.ones(n, dtype=comp_x.dtype), jnp.zeros(pad, dtype=comp_x.dtype)]
            )

            x_s = comp_x[perm_padded].reshape(n_batches, batch_size, -1)
            y_s = comp_y[perm_padded].reshape(
                (n_batches, batch_size) + comp_y.shape[1:]
            )
            m_s = valid.reshape(n_batches, batch_size)

            def batch_step(carry, batch):
                params, opt_state = carry
                x, y, m = batch

                def _loss(p):
                    preds = model.apply(p, x)

                    def _single_example_loss(pred_i, y_i):
                        return loss_fn(
                            pred_i[None, ...],
                            y_i[None, ...],
                            reduction="mean",
                        )

                    per_ex = jax.vmap(_single_example_loss)(preds, y)
                    return jnp.sum(per_ex * m) / jnp.maximum(jnp.sum(m), 1.0)

                grads = jax.grad(_loss)(params)
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), None

            (params, opt_state), _ = jax.lax.scan(
                batch_step, (params, opt_state), (x_s, y_s, m_s)
            )
            return (params, opt_state), None

        (params, _), _ = jax.lax.scan(epoch_step, (params, opt_state), all_epoch_keys)
        return _eval_queries(params, query_x, query_y)

    return jax.jit(jax.vmap(train_and_eval_one, in_axes=(0, None, None, None, None)))


def _build_elso_cache_key(
    model_config: ModelConfig,
    num_subsets: int,
    subset_fraction: float,
    reps_per_subset: int,
    seed: int,
    n_queries: int,
    epochs: int,
) -> Dict:
    """Cache key capturing everything that influences ELSO rep_mean."""
    tr = model_config.training
    return {
        "subsets": {
            "num_subsets": num_subsets,
            "subset_fraction": subset_fraction,
            "seed": seed,
        },
        "reps_per_subset": reps_per_subset,
        "epochs": epochs,
        "num_queries": n_queries,
        "training": {
            "optimizer": str(getattr(tr.optimizer, "value", tr.optimizer)),
            "learning_rate": tr.learning_rate,
            "weight_decay": tr.weight_decay,
            "lr_schedule": (
                str(getattr(tr.lr_schedule, "value", tr.lr_schedule))
                if tr.lr_schedule is not None
                else None
            ),
            "batch_size": tr.batch_size,
        },
    }


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
    epochs: int,
    use_vmap: bool = True,
    cache_directory: Optional[str] = None,
) -> np.ndarray:
    """Compute Δm_j(z_q) = E_ξ[loss(z_q, θ(D\\S_j))] - loss(z_q, θ(D)).

    Complements are trained for ``epochs`` epochs so the complement model
    matches the training duration of the baseline checkpoint loaded by the
    caller. Returns an array of shape (n_queries, K).
    """
    n_queries = len(query_inputs)
    K = len(subsets)
    batch_size = model_config.training.batch_size

    if K == 0:
        raise ValueError("compute_elso_ground_truth requires at least one subset (K>0).")
    subset_size = int(subsets[0].sum())
    complement_size = len(full_train_inputs) - subset_size
    if complement_size <= 0:
        raise ValueError(
            f"Complement is empty (|S_j|={subset_size}, N={len(full_train_inputs)}); "
            "subset_fraction must be < 1.0 so each retraining set is non-empty."
        )

    cache_path = (
        os.path.join(cache_directory, "rep_mean.npz")
        if cache_directory is not None
        else None
    )
    if cache_path is not None and os.path.exists(cache_path):
        try:
            with np.load(cache_path) as data:
                cached_rep_mean = data["rep_mean"]
            if cached_rep_mean.shape == (n_queries, K):
                logger.info(
                    "[ELSO] Cache hit: loaded rep_mean from %s (shape=%s).",
                    cache_path,
                    cached_rep_mean.shape,
                )
                return cached_rep_mean - np.asarray(baseline_losses)[:, None]
            logger.warning(
                "[ELSO] Cache at %s has shape mismatch — recomputing.", cache_path
            )
        except (OSError, KeyError, ValueError) as e:
            logger.warning(
                "[ELSO] Failed to load cache %s (%s) — recomputing.", cache_path, e
            )

    loss_fn = get_loss(model_config.loss)
    # Both paths see every example each epoch (vmap pads + masks the partial
    # batch; the dataloader yields a partial last batch). Step counts match.
    total_steps = epochs * ceil(complement_size / batch_size)
    opt = create_optimizer(
        optimizer_enum=model_config.training.optimizer,
        lr=model_config.training.learning_rate,
        weight_decay=model_config.training.weight_decay,
        lr_schedule=model_config.training.lr_schedule,
        total_steps=total_steps,
    )

    rep_mean = np.zeros((n_queries, K))

    if use_vmap:
        from src.utils.models.registry import ModelRegistry

        model = ModelRegistry.get_model(model_config=model_config)
        vmapped_fn = _build_vmapped_elso_fn(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            epochs=epochs,
            batch_size=batch_size,
        )
        logger.info(
            "[ELSO] vmap R=%d reps over K=%d subsets (compiling on first subset).",
            reps_per_subset,
            K,
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
            rep_losses = vmapped_fn(keys, comp_x, comp_y, query_inputs, query_targets)
            rep_mean[:, j] = np.array(jnp.mean(rep_losses, axis=0))
            if (j + 1) % max(1, K // 10) == 0 or j == 0:
                logger.info("[ELSO] Subset %d / %d done.", j + 1, K)

    else:
        for j, mask in enumerate(subsets):
            comp_inputs = full_train_inputs[~mask]
            comp_targets = full_train_targets[~mask]
            if len(comp_inputs) == 0:
                logger.warning("Subset %d: complement is empty — skipping.", j)
                continue
            ckpt_rep_losses = np.zeros((reps_per_subset, n_queries))
            for r in range(reps_per_subset):
                seed = base_seed + j * reps_per_subset + r
                dataloader = JAXDataLoader(
                    comp_inputs,
                    comp_targets,
                    batch_size=batch_size,
                    shuffle=True,
                    rng_key=jax.random.PRNGKey(seed),
                )
                subset_model, subset_params, _ = train_model(
                    model_config=model_config,
                    dataloader=dataloader,
                    loss_fn=loss_fn,
                    optimizer=create_optimizer(
                        optimizer_enum=model_config.training.optimizer,
                        lr=model_config.training.learning_rate,
                        weight_decay=model_config.training.weight_decay,
                        lr_schedule=model_config.training.lr_schedule,
                        total_steps=ceil(len(comp_inputs) / batch_size) * epochs,
                    ),
                    epochs=epochs,
                    seed=seed,
                    save_checkpoints=False,
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
            rep_mean[:, j] = ckpt_rep_losses.mean(axis=0)
            if (j + 1) % max(1, K // 10) == 0 or j == 0:
                logger.info("[ELSO] Subset %d / %d done.", j + 1, K)

    if cache_path is not None:
        try:
            os.makedirs(cache_directory, exist_ok=True)  # type: ignore[arg-type]
            np.savez(cache_path, rep_mean=rep_mean)
            logger.info("[ELSO] Saved rep_mean cache to %s.", cache_path)
        except OSError as e:
            logger.warning("[ELSO] Failed to save cache %s (%s).", cache_path, e)

    return rep_mean - np.asarray(baseline_losses)[:, None]


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
        Predicted group effects of shape (n_query, K).
    """
    n_query = attributions.shape[0]
    K = len(subsets)
    predicted = np.zeros((n_query, K))
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
    """Compute mean LDS across queries with a bootstrap CI over subsets (K).

    Args:
        delta_m:   (n_queries, K) ground-truth Δm_j(z_q).
        predicted: (n_queries, K) predicted group attributions g_τ(z_q, S_j).

    Queries whose Spearman correlation is undefined (NaN — occurs when either
    row is constant across the selected subsets) are excluded from the mean.

    Returns:
        Dict with keys: mean_lds, std_lds, ci_low, ci_high, per_query_lds,
        num_undefined_queries, num_valid_queries.
    """
    rng = np.random.RandomState(seed)
    n_queries, K = delta_m.shape

    def _mean_lds_over_subsets(j_idx: np.ndarray) -> float:
        rs = []
        for i in range(n_queries):
            r, _ = spearmanr(delta_m[i, j_idx], predicted[i, j_idx])
            if not np.isnan(r):  # type: ignore
                rs.append(float(r))  # type: ignore
        return float(np.mean(rs)) if rs else float("nan")

    all_j = np.arange(K)
    mean_lds = _mean_lds_over_subsets(all_j)

    per_query_lds: List[float] = []
    num_undefined = 0
    for i in range(n_queries):
        r, _ = spearmanr(delta_m[i], predicted[i])
        if np.isnan(r):  # type: ignore
            num_undefined += 1
            per_query_lds.append(float("nan"))
        else:
            per_query_lds.append(float(r))  # type: ignore

    bootstrap_means = [
        _mean_lds_over_subsets(rng.choice(K, size=K, replace=True))
        for _ in range(n_bootstrap)
    ]
    finite_bootstrap = [b for b in bootstrap_means if not np.isnan(b)]
    if finite_bootstrap:
        ci_low = float(np.percentile(finite_bootstrap, 2.5))
        ci_high = float(np.percentile(finite_bootstrap, 97.5))
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    finite_pq = [v for v in per_query_lds if not np.isnan(v)]
    std_lds = float(np.std(finite_pq)) if finite_pq else float("nan")

    return {
        "mean_lds": mean_lds,
        "std_lds": std_lds,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "per_query_lds": per_query_lds,
        "num_undefined_queries": num_undefined,
        "num_valid_queries": n_queries - num_undefined,
    }


# ---------------------------------------------------------------------------
# Main entrypoint: LDS from a saved attribution file
# ---------------------------------------------------------------------------


def compute_lds(config: LDSConfig) -> Dict:
    """Run ELSO retraining and score a precomputed attribution matrix.

    Loads the saved checkpoint at ``config.epoch`` (final if None), the saved
    attribution scores at ``config.attribution_scores``, retrains the model on
    each complement D\\S_j to estimate Δm_j, and reports the Spearman
    correlation between predicted and observed group effects.

    The train/test split is derived from ``model.json`` metadata so the data
    matches what training (and any prior analysis) saw.

    Args:
        config: LDSConfig pointing at a model directory and an attribution
            .npy file of shape ``(n_query, n_train)``.
    """
    params, model, model_config, _ = load_model_checkpoint(
        config.model, epoch=config.epoch
    )
    loss_fn = get_loss(model_config.loss)
    train_dataset, test_dataset = load_train_test_for_model(config.model)

    # Complements must be trained to the same epoch as the baseline checkpoint;
    # otherwise Δm conflates "removed S_j" with "trained for more epochs".
    target_epochs = (
        config.epoch if config.epoch is not None else model_config.training.epochs
    )

    n_train = len(train_dataset.inputs)
    n_query = min(config.num_test_examples, len(test_dataset.inputs))

    attributions = np.load(config.attribution_scores)
    if attributions.ndim != 2 or attributions.shape[1] != n_train:
        raise ValueError(
            f"Attribution matrix at {config.attribution_scores} has shape "
            f"{attributions.shape}; expected (n_query, {n_train})."
        )
    if attributions.shape[0] < n_query:
        raise ValueError(
            f"Attribution matrix has only {attributions.shape[0]} query rows "
            f"but num_test_examples={n_query}."
        )
    attributions = attributions[:n_query, :]

    query_inputs = test_dataset.inputs[:n_query]
    query_targets = test_dataset.targets[:n_query]

    epoch_str = f"epoch_{config.epoch}" if config.epoch is not None else "final"
    logger.info("=" * 70)
    logger.info("[LDS] %s (%s)", model_config.get_model_display_name(), epoch_str)
    logger.info("[LDS] model dir:   %s", config.model)
    logger.info("[LDS] attributions: %s", config.attribution_scores)
    logger.info("=" * 70)

    subsets = generate_random_subsets(
        dataset_size=n_train,
        num_subsets=config.num_subsets,
        subset_fraction=config.subset_fraction,
        seed=config.lds_seed,
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

    cache_directory = (
        elso_cache_dir(
            config.model,
            _build_elso_cache_key(
                model_config=model_config,
                num_subsets=config.num_subsets,
                subset_fraction=config.subset_fraction,
                reps_per_subset=config.reps_per_model,
                seed=config.lds_seed,
                n_queries=n_query,
                epochs=target_epochs,
            ),
        )
        if config.cache_elso
        else None
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
        base_seed=config.lds_seed,
        epochs=target_epochs,
        cache_directory=cache_directory,
    )

    predicted = compute_group_attributions(attributions, subsets)
    scores = aggregate_lds_scores(
        delta_m=delta_m,
        predicted=predicted,
        n_bootstrap=1000,
        seed=config.lds_seed,
    )

    logger.info(
        "[LDS] mean_LDS=%.4f ± %.4f  (95%% CI [%.4f, %.4f])",
        scores["mean_lds"],
        scores["std_lds"],
        scores["ci_low"],
        scores["ci_high"],
    )

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": config.model,
        "epoch": config.epoch,
        "attribution_scores": config.attribution_scores,
        "model_config": asdict(model_config),
        "num_subsets": len(subsets),
        "subset_fraction": config.subset_fraction,
        "reps_per_model": config.reps_per_model,
        "num_queries": n_query,
        "lds_scores": scores,
    }
