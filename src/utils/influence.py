from __future__ import annotations

import logging
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.utils.loss import loss_wrapper_with_apply_fn

logger = logging.getLogger(__name__)


def compute_per_example_flat_grads(
    model,
    params: Dict,
    inputs: Float[Array, "n_examples ..."],
    targets: Float[Array, "n_examples"],
    loss_fn,
    batch_size: int = 256,
) -> Float[Array, "n_examples n_params"]:
    """Compute per-example loss gradients as flat vectors via ravel_pytree.

    Uses the same parameter flattening as HessianEstimator so the output is
    directly compatible with ``HessianEstimator.estimate_ihvp``.

    Returns:
        Float array of shape (n, n_params).
    """
    flat_params, unravel = flatten_util.ravel_pytree(params)

    def flat_grad_single(args):
        x, y = args
        return jax.grad(
            lambda fp: loss_wrapper_with_apply_fn(
                model.apply,
                fp,
                unravel,
                loss_fn,
                x[None],
                jnp.atleast_1d(y),
            )
        )(flat_params)

    return jax.lax.map(
        flat_grad_single, (inputs, targets), batch_size=batch_size
    )  # (n, n_params)


def compute_influence_matrix(
    test_flat_grads: Float[Array, "n_test n_params"],
    train_flat_grads: Float[Array, "n_train n_params"],
    computer: HessianEstimator,
    damping: Optional[float] = None,
    pseudo_inverse_factor: Optional[float] = None,
) -> Float[Array, "n_test n_train"]:
    """Compute the (n_test, n_train) influence score matrix.

    Assumes that a data point is removed from the training set, i.e.,
    epsilon = -1 in influence function derivation.
    τ(z_q, z_i) = (H^{-1} ∇L(z_q))ᵀ ∇L(z_i)

    Args:
        test_flat_grads:  Shape (n_test, n_params).
        train_flat_grads: Shape (n_train, n_params).
        computer: A built HessianEstimator or HessianComputer (exact Hessian).
        damping: Additive regularisation scalar for ``(H + λI)^{-1}``.
        pseudo_inverse_factor: Threshold for truncated pseudo-inverse.
            Mutually exclusive with ``damping``.

    Returns:
        Attribution matrix of shape (n_test, n_train).
    """
    logger.info(
        "Computing IHVPs: %d test × %d params",
        len(test_flat_grads),
        test_flat_grads.shape[1],
    )

    if not computer.is_built:
        raise RuntimeError(
            "HessianEstimator not built. Please call `.build()` before computing influence scores."
        )
    ihvps = computer.estimate_ihvp(
        test_flat_grads, damping, pseudo_inverse_factor
    )  # (n_test, n_params)
    return ihvps @ train_flat_grads.T  # (n_test, n_train)


def compute_influence_matrix_streaming(
    test_inputs: Float[Array, "n_test ..."],
    test_targets: Float[Array, "n_test"],
    train_inputs: Float[Array, "n_train ..."],
    train_targets: Float[Array, "n_train"],
    model,
    params: Dict,
    loss_fn,
    computer: HessianEstimator,
    damping: Optional[float] = None,
    pseudo_inverse_factor: Optional[float] = None,
    train_batch_size: int = 256,
    test_batch_size: int = 1000,
):
    """Streaming variant of `compute_influence_matrix`, chunked on both axes.

    Avoids materializing the full ``(n_train, n_params)`` and ``(n_test,
    n_params)`` grad / ihvp matrices on device at once. Outer loop chunks
    test; per chunk we compute test grads, ihvps, and stream train via
    ``jax.lax.map``. Result blocks are pulled to host (numpy) and
    concatenated. Returns numpy array of shape ``(n_test, n_train)``.
    """
    import numpy as np

    if not computer.is_built:
        raise RuntimeError(
            "HessianEstimator not built. Please call `.build()` before computing influence scores."
        )

    n_test = len(test_inputs)
    n_train = len(train_inputs)
    flat_params, unravel = flatten_util.ravel_pytree(params)

    def flat_grad_single(args):
        x, y = args
        return jax.grad(
            lambda fp: loss_wrapper_with_apply_fn(
                model.apply,
                fp,
                unravel,
                loss_fn,
                x[None],
                jnp.atleast_1d(y),
            )
        )(flat_params)

    logger.info(
        "Computing influence: %d test × %d train, test_chunk=%d train_chunk=%d",
        n_test, n_train, test_batch_size, train_batch_size,
    )

    test_blocks = []
    for t0 in range(0, n_test, test_batch_size):
        t1 = min(t0 + test_batch_size, n_test)
        test_chunk_grads = jax.lax.map(
            flat_grad_single,
            (test_inputs[t0:t1], test_targets[t0:t1]),
            batch_size=train_batch_size,
        )  # (chunk, n_params)
        ihvps_chunk = computer.estimate_ihvp(
            test_chunk_grads, damping, pseudo_inverse_factor
        )  # (chunk, n_params)
        del test_chunk_grads

        def per_example_dot(args, _ihvps=ihvps_chunk):
            x, y = args
            g = flat_grad_single((x, y))
            return _ihvps @ g  # (chunk,)

        cols = jax.lax.map(
            per_example_dot,
            (train_inputs, train_targets),
            batch_size=train_batch_size,
        )  # (n_train, chunk)
        test_blocks.append(np.asarray(cols.T))  # (chunk, n_train) on host
        del ihvps_chunk, cols

    return np.concatenate(test_blocks, axis=0)  # (n_test, n_train)
