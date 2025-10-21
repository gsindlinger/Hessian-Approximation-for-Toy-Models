from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import optax


@partial(jax.jit, static_argnames=("reduction",))
def mse_loss(pred, target, reduction="mean"):
    """MSE loss (JIT compiled)."""
    if reduction == "mean":
        return jnp.mean((pred - target) ** 2)
    else:
        return jnp.sum((pred - target) ** 2)


@partial(jax.jit, static_argnames=("reduction",))
def cross_entropy_loss(pred, target, reduction="mean"):
    """Cross entropy loss (JIT compiled)."""
    if reduction == "mean":
        return optax.softmax_cross_entropy_with_integer_labels(pred, target).mean()
    else:
        return optax.softmax_cross_entropy_with_integer_labels(pred, target).sum()


def get_loss_fn(loss_str: Literal["mse", "cross_entropy"] = "mse") -> Callable:
    """Return loss function."""
    match loss_str:
        case "mse":
            return mse_loss
        case "cross_entropy":
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
