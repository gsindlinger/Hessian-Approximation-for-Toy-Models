from __future__ import annotations

import glob
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

from src.config import DatasetConfig, LDSConfig, LossType, ModelConfig
from src.utils.data.data import Dataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import get_loss
from src.utils.optimizers import optimizer as create_optimizer
from src.utils.train import (
    evaluate_per_example_losses,
    load_model_checkpoint,
    train_model,
)
from src.utils.utils import cleanup_memory, elso_cache_dir

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
    epoch in a single pass.

    ``jax.lax.scan`` replaces the Python epoch/batch loops, so there is no
    per-step Python dispatch overhead.  On GPU with a small model this gives
    ~R× speedup because Python overhead is paid once per subset instead of
    once per rep.

    The function is compiled on the first call and reused for all subsequent
    subsets (shapes are fixed: all complements have the same size).
    """
    checkpoints = sorted(epoch_checkpoints)
    max_epochs = checkpoints[-1]

    seg_lengths = [checkpoints[0]] + [
        checkpoints[i] - checkpoints[i - 1] for i in range(1, len(checkpoints))
    ]

    def _eval_queries(p, query_x, query_y):
        def single(x, y):
            return loss_fn(model.apply(p, x[None]), jnp.atleast_1d(y))

        return jax.vmap(single)(query_x, query_y)

    def train_and_eval_one(key, comp_x, comp_y, query_x, query_y):
        n = comp_x.shape[0]
        n_batches = -(-n // batch_size)  # ceil(n / batch_size)
        n_padded = n_batches * batch_size
        pad = n_padded - n

        params = model.init(key, comp_x[:1])
        opt_state = optimizer.init(params)

        # Pre-generate all epoch keys up-front so the training trajectory is
        # identical to a single max_epochs run (consistent with a sequential run).
        all_epoch_keys = jax.random.split(key, max_epochs)

        # Static validity mask: real samples occupy the first n positions of the
        # padded sequence; the trailing `pad` slots are filled with duplicated
        # indices and zeroed out in the loss so they contribute no gradient.
        mask = jnp.concatenate([jnp.ones(n), jnp.zeros(pad)])

        def epoch_step(carry, epoch_key):
            params, opt_state = carry
            perm = jax.random.permutation(epoch_key, n)
            perm_full = jnp.concatenate([perm, perm[:pad]])
            x_s = comp_x[perm_full].reshape(n_batches, batch_size, -1)
            y_s = comp_y[perm_full].reshape(n_batches, batch_size)
            m_s = mask.reshape(n_batches, batch_size)

            def batch_step(carry, batch):
                params, opt_state = carry
                x, y, m = batch

                def _loss(p):
                    def per_example(xi, yi):
                        return loss_fn(model.apply(p, xi[None]), jnp.atleast_1d(yi))

                    losses = jax.vmap(per_example)(x, y)
                    return (losses * m).sum() / m.sum()

                grads = jax.grad(_loss)(params)
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), None

            (params, opt_state), _ = jax.lax.scan(
                batch_step, (params, opt_state), (x_s, y_s, m_s)
            )
            return (params, opt_state), None

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

    return jax.jit(jax.vmap(train_and_eval_one, in_axes=(0, None, None, None, None)))


def _build_elso_cache_key(
    lds_config: LDSConfig,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    seed: int,
    checkpoints: List[int],
    n_queries: int,
) -> Dict:
    """Cache key capturing everything that influences ELSO rep_mean.

    Does NOT include attribution-score settings so swapping attribution files
    reuses the cached retraining output.
    """
    tr = model_config.training
    return {
        # Bumped when retraining semantics change; v2 = vmap path no longer drops
        # the partial last batch (pad+mask) so it matches the non-vmap path.
        "impl_version": 2,
        "subsets": {
            "num_subsets": lds_config.num_subsets,
            "subset_fraction": lds_config.subset_fraction,
            "seed": seed,
        },
        "reps_per_subset": lds_config.reps_per_model,
        "epoch_checkpoints": sorted(checkpoints),
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
        "dataset": {
            "name": str(getattr(dataset_config.name, "value", dataset_config.name)),
            "test_size": dataset_config.test_size,
        },
    }


def _apply_baseline(
    rep_mean: np.ndarray,
    baseline_losses: "np.ndarray | Dict[int, np.ndarray]",
    checkpoints: List[int],
    multi_epoch: bool,
) -> "np.ndarray | Dict[int, np.ndarray]":
    """Subtract per-query baseline from rep-averaged losses to form Δm.

    Baseline shifting cancels under Spearman, so this step does not change LDS,
    but it is preserved for consistency with the function's original contract.
    """
    if multi_epoch:
        assert isinstance(baseline_losses, dict), (
            "baseline_losses must be Dict[int, np.ndarray] when epoch_checkpoints has multiple entries."
        )
        return {
            ckpt: rep_mean[ci] - baseline_losses[ckpt][:, None]
            for ci, ckpt in enumerate(checkpoints)
        }
    return rep_mean[0] - np.asarray(baseline_losses)[:, None]


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
    cache_directory: Optional[str] = None,
) -> "np.ndarray | Dict[int, np.ndarray]":
    """Compute ELSO ground-truth Δm_j(z_q) for all (query, subset) pairs.

    For each subset S_j, retrains `reps_per_subset` models on D\\S_j and
    averages their per-query losses to estimate:

        Δm_j(z_q) = E_ξ[m(z_q, θ(D\\S_j))] - m(z_q, θ(D))

    Args:
        baseline_losses: For a single checkpoint, a ``(n_queries,)`` array.
            For multiple checkpoints, a dict ``{epoch: (n_queries,) array}``.
        epoch_checkpoints: Epoch numbers at which to evaluate query losses.
            Defaults to ``[model_config.training.epochs]``.
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
    complement_size = len(full_train_inputs) - int(subsets[0].sum())

    n_queries = len(query_inputs)
    K = len(subsets)

    cache_path = (
        os.path.join(cache_directory, "rep_mean.npz")
        if cache_directory is not None
        else None
    )
    if cache_path is not None and os.path.exists(cache_path):
        try:
            with np.load(cache_path) as data:
                cached_rep_mean = data["rep_mean"]
                cached_checkpoints = data["checkpoints"].tolist()
            if (
                cached_rep_mean.shape == (len(checkpoints), n_queries, K)
                and cached_checkpoints == checkpoints
            ):
                logger.info(
                    "[ELSO] Cache hit: loaded rep_mean from %s (shape=%s).",
                    cache_path,
                    cached_rep_mean.shape,
                )
                return _apply_baseline(
                    cached_rep_mean, baseline_losses, checkpoints, multi_epoch
                )
            logger.warning(
                "[ELSO] Cache at %s is stale (shape/checkpoint mismatch) — recomputing.",
                cache_path,
            )
        except (OSError, KeyError, ValueError) as e:
            logger.warning(
                "[ELSO] Failed to load cache %s (%s) — recomputing.", cache_path, e
            )

    loss_fn = get_loss(model_config.loss)
    opt = create_optimizer(
        optimizer_enum=model_config.training.optimizer,
        lr=model_config.training.learning_rate,
        weight_decay=model_config.training.weight_decay,
        lr_schedule=model_config.training.lr_schedule,
        total_steps=max(checkpoints)
        * ceil(complement_size / model_config.training.batch_size),
    )

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
            rep_losses = vmapped_fn(keys, comp_x, comp_y, query_inputs, query_targets)
            rep_mean[:, :, j] = np.array(jnp.mean(rep_losses, axis=0))

            if (j + 1) % max(1, K // 10) == 0 or j == 0:
                logger.info("[ELSO] Subset %d / %d done.", j + 1, K)

    else:
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
                    total_steps = (
                        ceil(len(comp_inputs) / model_config.training.batch_size) * ckpt
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
                            total_steps=total_steps,
                        ),
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

    if cache_path is not None:
        try:
            os.makedirs(cache_directory, exist_ok=True)  # type: ignore[arg-type]
            np.savez(
                cache_path,
                rep_mean=rep_mean,
                checkpoints=np.asarray(checkpoints),
            )
            logger.info("[ELSO] Saved rep_mean cache to %s.", cache_path)
        except OSError as e:
            logger.warning("[ELSO] Failed to save cache %s (%s).", cache_path, e)

    return _apply_baseline(rep_mean, baseline_losses, checkpoints, multi_epoch)


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
    """Compute mean LDS across queries with a bootstrap CI over subsets (K).

    Follows the Hong et al. (2025) protocol (Fig. 4 / App. A.2): the 95% CI
    accounts for randomness in *subset selection* by resampling the K subsets
    with replacement, recomputing Spearman per query on each resample, and
    averaging over queries.

    Args:
        delta_m:   (n_queries, K) ground-truth Δm_j(z_q).
        predicted: (n_queries, K) predicted group attributions g_τ(z_q, S_j).

    Queries whose Spearman correlation is undefined (NaN — occurs when either
    ``delta_m[i, :]`` or ``predicted[i, :]`` is constant across the selected
    subsets, i.e. zero variance) are *excluded* from the mean rather than
    clamped to 0.

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
# Attribution-score loading
# ---------------------------------------------------------------------------


def load_attribution_scores(
    attribution_dirs: "str | List[str]",
    n_test: int,
    n_train: int,
) -> Dict[str, np.ndarray]:
    """Load precomputed attribution scores from one or more directories.

    Each directory holds one ``{method}.npy`` per attribution method, where
    every file is a ``(n_test, n_train)`` influence matrix. Matrices are
    memory-mapped so big models don't have to fit fully in RAM. Results from
    all directories are merged into a single method → matrix dict.

    Raises:
        FileNotFoundError: if any directory does not exist.
        ValueError: if a matrix has the wrong shape, if the merged set is
            empty, or if the same method name appears in two directories
            (which would be ambiguous).
    """
    dirs: List[str] = (
        [attribution_dirs]
        if isinstance(attribution_dirs, str)
        else list(attribution_dirs)
    )
    if not dirs:
        raise ValueError("attribution_dirs is empty.")

    attributions: Dict[str, np.ndarray] = {}
    method_source: Dict[str, str] = {}
    for directory in dirs:
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Attribution-score directory not found: {directory}"
            )
        for path in sorted(glob.glob(os.path.join(directory, "*.npy"))):
            method = os.path.splitext(os.path.basename(path))[0]
            matrix = np.load(path, mmap_mode="r")
            if matrix.shape != (n_test, n_train):
                raise ValueError(
                    f"Attribution matrix '{method}' in {path} has "
                    f"shape {matrix.shape}, expected ({n_test}, {n_train})."
                )
            if method in attributions:
                raise ValueError(
                    f"Duplicate attribution method '{method}' found in "
                    f"{path} and {method_source[method]}."
                )
            attributions[method] = matrix
            method_source[method] = path
    if not attributions:
        raise ValueError(f"No attribution matrices found in {dirs}.")
    return attributions


# ---------------------------------------------------------------------------
# Main pipeline per model
# ---------------------------------------------------------------------------


def compute_lds_for_model(
    model_directory: str,
    attribution_dirs: "str | List[str]",
    train_dataset: Dataset,
    test_dataset: Dataset,
    lds_config: LDSConfig,
    dataset_config: DatasetConfig,
    seed: int,
    epoch: Optional[int] = None,
) -> Dict:
    """Compute ELSO LDS scores for one model + its precomputed attributions.

    ``attribution_dirs`` may be a single directory or a list of directories.
    Each directory holds one ``{method}.npy`` per attribution method, where
    every matrix has shape ``(n_test, n_train)``. Multiple directories are
    merged into one method → matrix dict, so a single model/epoch can carry
    attribution scores produced by separate approximators in separate dirs.

    ``dataset_config`` and ``seed`` come from the shared top-level analysis
    config and are only used for ELSO cache invalidation — the actual
    (train, test) data arrives via ``train_dataset`` / ``test_dataset``.

    Steps:
      1. Load the base model checkpoint (optionally at a specific epoch).
      2. Generate K random subsets S_j.
      3. ELSO: for each S_j retrain R models on D\\S_j → m_j.
      4. Load the precomputed attribution matrices from ``attribution_dirs``.
      5. Predicted group effects g_τ(S_j) = Σ_{z∈S_j} τ(z_q, z) for each method.
      6. LDS = Spearman(m_j, g_τ_j) with 95% bootstrap CI, averaged over queries.
    """

    params, model, model_config, metadata = load_model_checkpoint(
        model_directory, epoch=epoch
    )
    loss_fn = get_loss(model_config.loss)

    dirs_list: List[str] = (
        [attribution_dirs]
        if isinstance(attribution_dirs, str)
        else list(attribution_dirs)
    )

    epoch_str = f"epoch_{epoch}" if epoch is not None else "final"
    logger.info("=" * 70)
    logger.info("[LDS] %s (%s)", model_config.get_model_display_name(), epoch_str)
    logger.info("[LDS] model dir: %s", model_directory)
    logger.info("[LDS] attributions: %s", dirs_list)
    logger.info("=" * 70)

    n_query = min(lds_config.num_test_examples, len(test_dataset.inputs))
    query_inputs = test_dataset.inputs[:n_query]
    query_targets = test_dataset.targets[:n_query]

    if model_config.loss == LossType.MSE:
        train_inputs, query_inputs = Dataset.normalize_data(
            train_dataset.inputs, query_inputs
        )
        train_targets, query_targets = Dataset.normalize_data(
            train_dataset.targets, query_targets
        )
        assert query_inputs is not None and query_targets is not None
        train_dataset = Dataset(train_inputs, train_targets)

    subsets = generate_random_subsets(
        dataset_size=len(train_dataset.inputs),
        num_subsets=lds_config.num_subsets,
        subset_fraction=lds_config.subset_fraction,
        seed=seed,
    )

    # Baseline cancels under Spearman; use the loaded checkpoint directly.
    baseline_losses = np.array(
        evaluate_per_example_losses(
            model=model,
            params=params,
            inputs=query_inputs,
            targets=query_targets,
            loss_fn=loss_fn,
        )
    )

    logger.info(
        "[LDS] ELSO retraining: K=%d subsets × R=%d reps on D\\S_j.",
        len(subsets),
        lds_config.reps_per_model,
    )
    elso_checkpoints = [model_config.training.epochs]
    elso_cache_directory = (
        elso_cache_dir(
            model_directory,
            _build_elso_cache_key(
                lds_config=lds_config,
                model_config=model_config,
                dataset_config=dataset_config,
                seed=seed,
                checkpoints=elso_checkpoints,
                n_queries=n_query,
            ),
        )
        if lds_config.cache_elso
        else None
    )
    delta_m = compute_elso_ground_truth(
        model_config=model_config,
        full_train_inputs=train_dataset.inputs,
        full_train_targets=train_dataset.targets,
        query_inputs=query_inputs,
        query_targets=query_targets,
        subsets=subsets,
        reps_per_subset=lds_config.reps_per_model,
        baseline_losses=baseline_losses,
        base_seed=seed,
        cache_directory=elso_cache_directory,
    )
    assert isinstance(delta_m, np.ndarray)
    cleanup_memory("elso_retraining")

    # Load precomputed attribution matrices, keyed by method name.
    attributions_by_method = load_attribution_scores(
        attribution_dirs=dirs_list,
        n_test=n_query,
        n_train=len(train_dataset.inputs),
    )

    lds_scores: Dict = {}
    for method, attributions in attributions_by_method.items():
        logger.info("[LDS] Attribution method: %s", method)
        predicted = compute_group_attributions(attributions, subsets)
        scores = aggregate_lds_scores(
            delta_m=delta_m,
            predicted=predicted,
            n_bootstrap=1000,
            seed=seed,
        )
        logger.info(
            "[LDS] %s: mean_LDS=%.4f ± %.4f  (95%% CI: [%.4f, %.4f])",
            method,
            scores["mean_lds"],
            scores["std_lds"],
            scores["ci_low"],
            scores["ci_high"],
        )
        lds_scores[method] = scores
        cleanup_memory(f"lds_{method}")

    return {
        "model_name": model_config.get_model_display_name(),
        "model_directory": model_directory,
        "attribution_dirs": dirs_list,
        "epoch": epoch,
        "model_config": asdict(model_config),
        "num_subsets": len(subsets),
        "subset_fraction": lds_config.subset_fraction,
        "reps_per_model": lds_config.reps_per_model,
        "num_queries": n_query,
        "lds_scores": lds_scores,
        "metadata": metadata or {},
    }
