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

import json
import logging
import os
from dataclasses import asdict
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import PRNGKey
from scipy.stats import rankdata, spearmanr
from tqdm.auto import tqdm

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


def _spearman_or_nan(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return Spearman correlation, or NaN when it is mathematically undefined."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.size < 2 or predicted.size < 2:
        return float("nan")
    if np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
        return float("nan")
    r, _ = spearmanr(actual, predicted)
    return float(r) if not np.isnan(r) else float("nan")


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
    logger.debug(
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
    n_train = len(full_train_inputs)

    if K == 0:
        raise ValueError("compute_elso_ground_truth requires at least one subset (K>0).")

    for subset_idx, subset_mask in enumerate(subsets):
        if not isinstance(subset_mask, np.ndarray):
            raise TypeError(
                f"Subset mask at index {subset_idx} must be a numpy.ndarray, "
                f"got {type(subset_mask).__name__}."
            )
        if subset_mask.dtype != np.bool_:
            raise ValueError(
                f"Subset mask at index {subset_idx} must have boolean dtype, "
                f"got {subset_mask.dtype}."
            )
        if subset_mask.ndim != 1:
            raise ValueError(
                f"Subset mask at index {subset_idx} must be 1D, got shape {subset_mask.shape}."
            )
        if len(subset_mask) != n_train:
            raise ValueError(
                f"Subset mask at index {subset_idx} has length {len(subset_mask)}, "
                f"expected {n_train}."
            )

        subset_size = int(subset_mask.sum())
        complement_size = n_train - subset_size
        if complement_size <= 0:
            raise ValueError(
                f"Complement is empty for subset index {subset_idx} "
                f"(|S_j|={subset_size}, N={n_train}); "
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
                logger.debug(
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
        logger.debug(
            "[ELSO] vmap R=%d reps over K=%d subsets (compiling on first subset).",
            reps_per_subset,
            K,
        )

        for j, mask in enumerate(subsets):
            comp_x = jnp.asarray(full_train_inputs[~mask])
            comp_y = jnp.asarray(full_train_targets[~mask])
            if len(comp_x) == 0:
                raise ValueError(
                    f"Subset {j}: complement is empty (|S_j|={int(mask.sum())}, "
                    f"N={len(full_train_inputs)}); cannot compute ELSO ground truth."
                )
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
                raise ValueError(
                    f"Subset {j}: complement is empty (|S_j|={int(mask.sum())}, "
                    f"N={len(full_train_inputs)}); cannot compute ELSO ground truth."
                )
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


def _spearman_along_axis(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Vectorized Spearman correlation along ``axis``.

    Equivalent to ``scipy.stats.spearmanr(x, y, axis=axis)[0]`` but accepts
    arbitrary leading dimensions and avoids per-pair Python overhead. Returns
    NaN where the correlation is mathematically undefined (constant row on
    either side, ``< 2`` elements along the axis), without emitting warnings.

    Uses ``method='average'`` rank ties to match scipy's default.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.shape[axis] < 2:
        out_shape = list(x.shape)
        del out_shape[axis if axis >= 0 else x.ndim + axis]
        return np.full(out_shape, np.nan)

    rx = rankdata(x, method="average", axis=axis)
    ry = rankdata(y, method="average", axis=axis)

    rx_c = rx - rx.mean(axis=axis, keepdims=True)
    ry_c = ry - ry.mean(axis=axis, keepdims=True)
    num = (rx_c * ry_c).sum(axis=axis)
    den = np.sqrt((rx_c ** 2).sum(axis=axis) * (ry_c ** 2).sum(axis=axis))

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)

    # Belt-and-braces: NaN any row that was constant on either side.
    const_x = (x.max(axis=axis) == x.min(axis=axis))
    const_y = (y.max(axis=axis) == y.min(axis=axis))
    out = np.where(const_x | const_y, np.nan, out)
    return out


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

    # Per-query point estimate over all K subsets (vectorized along subsets).
    per_query_arr = _spearman_along_axis(delta_m, predicted, axis=1)  # (n_queries,)
    valid_mask = ~np.isnan(per_query_arr)
    num_undefined = int((~valid_mask).sum())

    finite_pq = per_query_arr[valid_mask]
    mean_lds = float(finite_pq.mean()) if finite_pq.size else float("nan")
    std_lds = float(finite_pq.std()) if finite_pq.size else float("nan")

    # Bootstrap: resample subsets with replacement and recompute Spearman per
    # (query, bootstrap_iter). Done in one vectorized pass instead of
    # n_bootstrap × n_queries scipy.spearmanr calls.
    bootstrap_idx = rng.choice(K, size=(n_bootstrap, K), replace=True)  # (B, K)
    delta_m_boot = delta_m[:, bootstrap_idx]   # (n_queries, B, K)
    pred_boot = predicted[:, bootstrap_idx]    # (n_queries, B, K)
    sp = _spearman_along_axis(delta_m_boot, pred_boot, axis=2)  # (n_queries, B)

    # Mean over queries within each bootstrap sample, ignoring NaN queries.
    with np.errstate(invalid="ignore"):
        bootstrap_means = np.nanmean(sp, axis=0)  # (B,)
    finite_boot = bootstrap_means[~np.isnan(bootstrap_means)]
    if finite_boot.size:
        ci_low = float(np.percentile(finite_boot, 2.5))
        ci_high = float(np.percentile(finite_boot, 97.5))
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    per_query_lds = [
        float(v) if not np.isnan(v) else float("nan") for v in per_query_arr
    ]

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


def _compute_lds_for_attribution(
    *,
    model_directory: str,
    epoch: Optional[int],
    attribution_path: str,
    num_subsets: int,
    reps_per_model: int,
    subset_fraction: float,
    num_test_examples: int,
    lds_seed: int,
    cache_elso: bool,
) -> Dict:
    """Score a single attribution .npy via ELSO retraining.

    The ELSO cache is keyed on ``(model_directory, recipe)`` only, so distinct
    attribution files for the same checkpoint reuse the expensive retraining.
    """
    params, model, model_config, _ = load_model_checkpoint(
        model_directory, epoch=epoch
    )
    loss_fn = get_loss(model_config.loss)
    train_dataset, test_dataset = load_train_test_for_model(model_directory)

    # Complements must be trained to the same epoch as the baseline checkpoint;
    # otherwise Δm conflates "removed S_j" with "trained for more epochs".
    target_epochs = epoch if epoch is not None else model_config.training.epochs

    n_train = len(train_dataset.inputs)
    n_query = min(num_test_examples, len(test_dataset.inputs))

    attributions = np.load(attribution_path)
    if attributions.ndim != 2 or attributions.shape[1] != n_train:
        raise ValueError(
            f"Attribution matrix at {attribution_path} has shape "
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

    subsets = generate_random_subsets(
        dataset_size=n_train,
        num_subsets=num_subsets,
        subset_fraction=subset_fraction,
        seed=lds_seed,
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
            model_directory,
            _build_elso_cache_key(
                model_config=model_config,
                num_subsets=num_subsets,
                subset_fraction=subset_fraction,
                reps_per_subset=reps_per_model,
                seed=lds_seed,
                n_queries=n_query,
                epochs=target_epochs,
            ),
        )
        if cache_elso
        else None
    )

    delta_m = compute_elso_ground_truth(
        model_config=model_config,
        full_train_inputs=train_dataset.inputs,
        full_train_targets=train_dataset.targets,
        query_inputs=query_inputs,
        query_targets=query_targets,
        subsets=subsets,
        reps_per_subset=reps_per_model,
        baseline_losses=baseline_losses,
        base_seed=lds_seed,
        epochs=target_epochs,
        cache_directory=cache_directory,
    )

    predicted = compute_group_attributions(attributions, subsets)
    scores = aggregate_lds_scores(
        delta_m=delta_m,
        predicted=predicted,
        n_bootstrap=1000,
        seed=lds_seed,
    )

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_directory,
        "epoch": epoch,
        "attribution_scores": attribution_path,
        "model_config": asdict(model_config),
        "num_subsets": len(subsets),
        "subset_fraction": subset_fraction,
        "reps_per_model": reps_per_model,
        "num_queries": n_query,
        "lds_scores": scores,
    }


def _entry_passes_filter(entry: Dict[str, Any], method: str, config: LDSConfig) -> bool:
    f = config.filter
    if f.model_ids and entry.get("model_id") not in f.model_ids:
        return False
    if f.epochs and entry.get("epoch") not in f.epochs:
        return False
    if f.methods and method not in f.methods:
        return False
    return True


def compute_lds(*, results_json: str, config: LDSConfig) -> Dict:
    """Iterate every (entry, method) attribution recorded in a results.json
    and score it via ELSO retraining.

    Entries without an ``hessian_analysis.influence.paths`` block are skipped
    with a warning. Within a single ``(model_id, epoch)`` the ELSO retraining
    cache is shared across methods, so adding methods is cheap.
    """
    with open(results_json) as f:
        rj = json.load(f)

    # Build the work list and skip list up front so we know how many tasks
    # there are before showing the progress bar.
    tasks: List[Tuple[Dict[str, Any], str, str]] = []
    skipped: List[Dict[str, Any]] = []

    for entry in rj.get("results", []):
        model_id = entry.get("model_id")
        epoch = entry.get("epoch")
        influence = (entry.get("hessian_analysis") or {}).get("influence") or {}
        paths_map = influence.get("paths") or {}
        if not paths_map:
            logger.warning(
                "[LDS] skip %s epoch=%s — no influence paths in results.json",
                model_id, epoch,
            )
            skipped.append({
                "model_id": model_id,
                "epoch": epoch,
                "reason": "no influence paths",
            })
            continue
        for method, npy_path in paths_map.items():
            if not _entry_passes_filter(entry, method, config):
                continue
            tasks.append((entry, method, npy_path))

    # One-shot setup banner — replaces the per-task banner.
    n_models = len({(t[0].get("model_id"), t[0].get("epoch")) for t in tasks})
    logger.info("=" * 70)
    logger.info(
        "[LDS] %d task(s) across %d (model, epoch) pair(s) from %s",
        len(tasks), n_models, results_json,
    )
    logger.info(
        "[LDS] recipe: K=%d subsets, alpha=%.2f, R=%d reps, queries=%d, seed=%d, cache_elso=%s",
        config.num_subsets, config.subset_fraction, config.reps_per_model,
        config.num_test_examples, config.lds_seed, config.cache_elso,
    )
    if config.filter.model_ids or config.filter.epochs or config.filter.methods:
        logger.info(
            "[LDS] filter: model_ids=%s epochs=%s methods=%s",
            config.filter.model_ids, config.filter.epochs, config.filter.methods,
        )
    logger.info("=" * 70)

    out_results: List[Dict[str, Any]] = []
    pbar = tqdm(tasks, desc="LDS", disable=not tasks)
    for entry, method, npy_path in pbar:
        model_id = entry.get("model_id") or ""
        epoch = entry.get("epoch")
        pbar.set_postfix_str(
            f"{model_id[-12:]} e={epoch if epoch is not None else 'final'} m={method}"
        )
        r = _compute_lds_for_attribution(
            model_directory=entry["model_directory"],
            epoch=epoch,
            attribution_path=npy_path,
            num_subsets=config.num_subsets,
            reps_per_model=config.reps_per_model,
            subset_fraction=config.subset_fraction,
            num_test_examples=config.num_test_examples,
            lds_seed=config.lds_seed,
            cache_elso=config.cache_elso,
        )
        r["model_id"] = model_id
        r["method"] = method
        out_results.append(r)

    if out_results:
        _log_summary_table(out_results)

    return {
        "results_json": str(results_json),
        "run_id": rj.get("run_id"),
        "lds_config": asdict(config),
        "results": out_results,
        "skipped": skipped,
    }


def _log_summary_table(results: List[Dict[str, Any]]) -> None:
    """Pretty per-(model, epoch, method) summary printed once at the end."""
    header = (
        f"{'model_id':<22} {'epoch':>6} {'method':<14}"
        f" {'mean':>9} {'std':>8} {'95% CI':>20} {'valid':>6}"
    )
    rule = "-" * len(header)
    logger.info(rule)
    logger.info(header)
    logger.info(rule)
    for r in results:
        s = r.get("lds_scores") or {}
        mid = (r.get("model_id") or "")[-22:]
        epoch = r.get("epoch")
        epoch_s = "final" if epoch is None else str(epoch)
        method = (r.get("method") or "")[:14]
        ci_low = s.get("ci_low", float("nan"))
        ci_high = s.get("ci_high", float("nan"))
        ci_s = f"[{ci_low:+.3f}, {ci_high:+.3f}]"
        logger.info(
            "%-22s %6s %-14s %9.4f %8.4f %20s %6d",
            mid, epoch_s, method,
            s.get("mean_lds", float("nan")),
            s.get("std_lds", float("nan")),
            ci_s,
            s.get("num_valid_queries", -1),
        )
    logger.info(rule)
