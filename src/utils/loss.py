from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from src.config import LossType


@partial(jax.jit, static_argnames=("reduction",))
def mse_loss(
    pred: Float[Array, "batch output_dim"], 
    target: Float[Array, "batch output_dim"], 
    reduction="mean"
) -> Float:
    """MSE loss - always averages over output dims, configurable batch reduction
    
    Assumes batch dimension is always present (axis 0).
    """
    squared_error = (pred - target) ** 2
    
    # Average always over output_dim
    per_sample_mse = jnp.mean(squared_error, axis=-1)
    
    if reduction == "mean":
        return jnp.mean(per_sample_mse)
    elif reduction == "sum":
        return jnp.sum(per_sample_mse)
    elif reduction == "none":
        return per_sample_mse
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


@partial(jax.jit, static_argnames=("reduction",))
def cross_entropy_loss(pred: Float[Array, "batch n_classes"], target: Float[Array, "batch"], reduction="mean") -> Float:
    """Cross entropy loss"""
    if reduction == "mean":
        return optax.softmax_cross_entropy_with_integer_labels(pred, target).mean()
    else:
        return optax.softmax_cross_entropy_with_integer_labels(pred, target).sum()


def get_loss(loss_str: LossType) -> Callable:
    """Return loss function."""
    match loss_str:
        case LossType.MSE:
            return mse_loss
        case LossType.CROSS_ENTROPY:
            return cross_entropy_loss
        case _:
            raise ValueError(f"Unknown loss function: {loss_str}")


def get_loss_name(loss_fn: Callable) -> str:
    """Return loss function name."""
    if loss_fn == mse_loss:
        return "mse"
    elif loss_fn == cross_entropy_loss:
        return "cross_entropy"
    else:
        return "unknown"


def loss_wrapper_with_apply_fn(
    apply_fn: Callable,
    params_flat: Any,
    unravel_fn: Callable[[Any], Any],
    loss_fn: Callable,
    inputs: Array,
    targets: Array,
):
    params_unraveled = unravel_fn(params_flat)
    outputs = apply_fn(params_unraveled, inputs)
    return loss_fn(outputs, targets)
