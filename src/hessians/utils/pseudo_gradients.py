from typing import Callable, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float, Int

from src.utils.loss import get_loss_name


def generate_pseudo_targets(
    model: nn.Module,
    params: Dict,
    inputs: Float[Array, "n_samples features"],
    loss_fn: Callable,
    rng_key: Array | None = None,
) -> Float[Array, "n_samples targets"] | Int[Array, "n_samples"]:
    """
    Generate pseudo-targets based on the model's output distribution.

    This is used to compute the true Fisher Information Matrix rather than
    the empirical Fisher (which would use true labels).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    loss_name = get_loss_name(loss_fn)
    if loss_name == "cross_entropy":
        return _generate_classification_pseudo_targets(model, params, inputs, rng_key)
    elif loss_name == "mse":
        return _generate_regression_pseudo_targets(model, params, inputs, rng_key)
    else:
        raise ValueError(f"Unsupported loss function for EKFAC: {loss_name}")


def _generate_classification_pseudo_targets(
    model: nn.Module,
    params: Dict,
    inputs: Float[Array, "n_samples features"],
    rng_key: Array,
) -> Int[Array, "n_samples"]:
    """Generate pseudo-targets by sampling from softmax probabilities."""
    logits = model.apply(params, inputs)
    if not isinstance(logits, jnp.ndarray):
        raise ValueError("Model predictions must be a jnp.ndarray for classification.")
    probs = jax.nn.softmax(logits, axis=-1)
    return jax.random.categorical(rng_key, jnp.log(probs), axis=-1)


def _generate_regression_pseudo_targets(
    model: nn.Module,
    params: Dict,
    inputs: Float[Array, "n_samples features"],
    rng_key: Array,
    noise_std: float = 1.0,
) -> Float[Array, "n_samples"]:
    """Generate pseudo-targets by adding Gaussian noise to predictions."""
    preds = model.apply(params, inputs)
    if not isinstance(preds, jnp.ndarray):
        raise ValueError("Model predictions must be a jnp.ndarray for regression.")
    noise = noise_std * jax.random.normal(rng_key, preds.shape)
    return preds + noise
