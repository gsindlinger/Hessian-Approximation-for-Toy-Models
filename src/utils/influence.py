from __future__ import annotations

import logging
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.hessian import HessianComputer
from src.utils.loss import loss_wrapper_with_apply_fn

logger = logging.getLogger(__name__)


def compute_per_example_flat_grads(
    model,
    params: Dict,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_fn,
) -> jnp.ndarray:
    """Compute per-example loss gradients as flat vectors via ravel_pytree.

    Uses the same parameter flattening as HessianEstimator so the output is
    directly compatible with ``HessianEstimator.estimate_ihvp``.

    Returns:
        Float array of shape (n, n_params).
    """
    flat_params, unravel = flatten_util.ravel_pytree(params)

    def flat_grad_single(x, y):
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

    return jax.vmap(flat_grad_single)(inputs, targets)  # (n, n_params)


def compute_influence_matrix(
    test_flat_grads: jnp.ndarray,
    train_flat_grads: jnp.ndarray,
    computer: HessianEstimator | HessianComputer,
    damping: float,
) -> np.ndarray:
    """Compute the (n_test, n_train) influence score matrix.

    τ(z_q, z_i) = -(H^{-1} ∇L(z_q))ᵀ ∇L(z_i)

    Args:
        test_flat_grads:  Shape (n_test, n_params).
        train_flat_grads: Shape (n_train, n_params).
        computer: A built HessianEstimator or HessianComputer (exact Hessian).
        damping: Regularisation value for IHVP.

    Returns:
        Attribution matrix of shape (n_test, n_train).
    """
    logger.info(
        "Computing IHVPs: %d test × %d params",
        len(test_flat_grads),
        test_flat_grads.shape[1],
    )

    if isinstance(computer, HessianComputer):
        ihvps = computer.compute_ihvp(test_flat_grads, damping)  # (n_test, n_params)
    else:
        if not computer.is_built:
            raise RuntimeError(
                "HessianComputer not built. Please call the 'build' method before computing influence scores."
            )
        ihvps = computer.estimate_ihvp(test_flat_grads, damping)  # (n_test, n_params)
    return -(np.array(ihvps) @ np.array(train_flat_grads).T)  # (n_test, n_train)
