from typing import Callable, Dict

import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import linen as nn
from jax import flatten_util
from jaxtyping import Array, Float, Int

from src.utils.loss import get_loss_name
from src.utils.models.approximation_model import ApproximationModel


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


def sample_gradients(
    model: ApproximationModel,
    params: Dict,
    inputs: Float[Array, "N features"],
    targets: Float[Array, "N n_targets"],
    loss_fn: Callable,
    n_vectors: int = 1,
    rng_key: PRNGKey | None = None,
):
    # assert that inputs and targets have a batch dimension and matching sizes
    assert inputs.shape[0] == targets.shape[0], (
        "Inputs and targets must have the same number of samples."
    )
    # assert that inputs and targets size is larger than n_vectors
    assert inputs.shape[0] >= n_vectors, (
        "Number of input samples must be at least n_vectors."
    )

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    num_train = inputs.shape[0]
    rng_key, subkey = jax.random.split(rng_key)
    idx = jax.random.choice(subkey, num_train, (n_vectors,), replace=False)

    sample_inputs = inputs[idx]

    pseudo_targets = generate_pseudo_targets(
        model=model,
        params=params,
        inputs=sample_inputs,
        loss_fn=loss_fn,
        rng_key=rng_key,
    )

    def grad_and_flatten(params, inputs, pseudo_targets):
        grad = jax.grad(lambda p: loss_fn(model.apply(p, inputs), pseudo_targets))(
            params
        )
        flat, _ = flatten_util.ravel_pytree(grad)
        return flat

    # Vectorize the fused operation with vmap and JIT
    batched_grad_fn = jax.jit(jax.vmap(grad_and_flatten, in_axes=(None, 0, 0)))
    return batched_grad_fn(params, sample_inputs, pseudo_targets)
