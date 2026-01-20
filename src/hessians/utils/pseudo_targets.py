from typing import Callable, Dict

import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import linen as nn
from jax import flatten_util
from jaxtyping import Array, Float, Int

from src.config import VectorAnalysisConfig, VectorSamplingMethod
from src.utils.data.data import Dataset
from src.utils.loss import get_loss_name
from src.utils.models.approximation_model import ApproximationModel


def generate_pseudo_targets_dataset(
    model: nn.Module,
    params: Dict,
    dataset: Dataset,
    loss_fn: Callable,
    rng_key: Array | None = None,
    monte_carlo_repetitions: int = 1,
):
    """
    Generate pseudo-targets for an entire dataset based on the model's output distribution.

    This is used to compute the true Fisher Information Matrix rather than
    the empirical Fisher (which would use true labels).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
        
    keys = jax.random.split(rng_key, monte_carlo_repetitions)
    pseudo_targets = jax.vmap(
        lambda key: generate_pseudo_targets(
            model, params, dataset.inputs, loss_fn, rng_key=key
        )
    )(keys)
    
    # provide dataset with inputs repeated and corresponding pseudo targets
    # first bring pseudo_targets to shape (n_samples * monte_carlo_repetitions, target_shape)
    repeated_inputs = jnp.tile(dataset.inputs, (monte_carlo_repetitions, 1))
    # now bring pseudo_targets from shape (monte_carlo_repetitions, n_samples) to (n_samples * monte_carlo_repetitions,)
    pseudo_targets = pseudo_targets.reshape(-1, *pseudo_targets.shape[2:])
    return dataset.__class__(inputs=repeated_inputs, targets=pseudo_targets)
    

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
    return jax.random.categorical(rng_key, logits, axis=-1)


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


def sample_vectors(
    vector_config: VectorAnalysisConfig,
    model: ApproximationModel,
    params: Dict,
    inputs: Float[Array, "N features"],
    targets: Float[Array, "N n_targets"],
    loss_fn: Callable,
    seed: int = 42,
    repetitions: int = 1,
) -> Float[Array, "repetitions n_vectors num_params"]:
    n_vectors = vector_config.num_samples * repetitions
    if vector_config.sampling_method == VectorSamplingMethod.RANDOM:
        raise NotImplementedError("Random sampling not implemented yet.")
    elif vector_config.sampling_method == VectorSamplingMethod.GRADIENTS:
        gradients = sample_gradients(
            model=model,
            params=params,
            inputs=inputs,
            targets=targets,
            loss_fn=loss_fn,
            n_vectors=n_vectors,
            rng_key=jax.random.PRNGKey(seed),
        )

    else:
        raise ValueError(
            f"Unknown vector sampling method: {vector_config.sampling_method}"
        )

    # if repetitions > 1, reshape to (repetitions, num_samples, num_params)
    if repetitions > 1:
        gradients = gradients.reshape((repetitions, vector_config.num_samples, -1))
    return gradients


def sample_gradients(
    model: ApproximationModel,
    params: Dict,
    inputs: Float[Array, "N features"],
    targets: Float[Array, "N n_targets"],
    loss_fn: Callable,
    n_vectors: int = 1,
    rng_key: PRNGKey | None = None,
) -> Float[Array, "n_vectors num_params"]:
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
    sample_targets = targets[idx]

    def grad_and_flatten(params, inputs, pseudo_targets):
        grad = jax.grad(lambda p: loss_fn(model.apply(p, inputs), pseudo_targets))(
            params
        )
        flat, _ = flatten_util.ravel_pytree(grad)
        return flat

    # Vectorize the fused operation with vmap and JIT
    batched_grad_fn = jax.jit(jax.vmap(grad_and_flatten, in_axes=(None, 0, 0)))
    return batched_grad_fn(params, sample_inputs, sample_targets)
