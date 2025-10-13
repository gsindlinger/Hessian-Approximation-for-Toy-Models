from __future__ import annotations
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util
from typing_extensions import override
from functools import partial

from hessian_approximations.hessian_approximations import HessianApproximation
from models.models import ApproximationModel, get_loss_name


class FisherInformation(HessianApproximation):
    def __init__(self, samples_per_input: int = 1):
        super().__init__()
        self.samples_per_input = samples_per_input

    @override
    def compute_hessian(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> jnp.ndarray:
        """
        Compute exact Fisher Information Matrix (FIM).
        FIM = E[∇log p(y|x,θ) ∇log p(y|x,θ)^T]

        Importantly, for each input x, we need to sample y ~ p(y|x,θ).
        Therefore we apply a repeated sampling strategy.
        """
        training_data = jnp.asarray(training_data)
        training_targets = jnp.asarray(training_targets)

        if get_loss_name(loss_fn) == "cross_entropy":
            return self._compute_classification_fim(model, params, training_data)
        else:
            return self._compute_regression_fim(model, params, training_data)

    def _compute_classification_fim(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Fisher Information Matrix based on pseudo-gradients for classification.
        For generating pseudo-gradients, we sample y ~ p(y|x,θ) from the model's predicted distribution.
        """
        params_flat, unravel_fn = flatten_util.ravel_pytree(params)

        # Create all random keys upfront
        rng = jax.random.PRNGKey(0)
        n_samples = training_data.shape[0]
        rngs = jax.random.split(rng, n_samples * self.samples_per_input)
        rngs = rngs.reshape(n_samples, self.samples_per_input, -1)

        # JIT compile the per-sample-per-mc-sample computation
        @jax.jit
        def compute_single_gradient(p_flat, x_sample, rng_key):
            def logits_fn(p):
                params_unflat = unravel_fn(p)
                logits = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(logits, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return logits[0]

            logits = logits_fn(p_flat)
            sampled_y = jax.random.categorical(rng_key, logits)

            def log_prob_fn(p):
                logits_inner = logits_fn(p)
                log_probs = jax.nn.log_softmax(logits_inner, axis=-1)
                return log_probs[sampled_y]

            grad_vec = jax.grad(log_prob_fn)(p_flat)
            return jnp.outer(grad_vec, grad_vec)

        def compute_sample_fim(x_sample, sample_rngs):
            # Create a closure that captures x_sample
            compute_with_x = lambda rng_key: compute_single_gradient(
                params_flat, x_sample, rng_key
            )
            mc_fims = jax.vmap(compute_with_x)(sample_rngs)
            return mc_fims.mean(axis=0)

        # Vectorize over all training samples
        fim = jax.vmap(compute_sample_fim)(training_data, rngs).sum(axis=0)

        fim /= n_samples
        return fim

    def _compute_regression_fim(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Fisher Information Matrix for regression.
        Assumes Gaussian likelihood: y ~ N(f(x;θ), σ^2)

        For generating pseudo-gradients, we sample y ~ p(y|x,θ) from the model's predicted distribution.
        """
        params_flat, unravel_fn = flatten_util.ravel_pytree(params)
        sigma2 = 1.0

        # Create all random keys upfront
        rng = jax.random.PRNGKey(0)
        n_samples = training_data.shape[0]
        rngs = jax.random.split(rng, n_samples * self.samples_per_input)
        rngs = rngs.reshape(n_samples, self.samples_per_input, -1)

        # JIT compile the per-sample-per-mc-sample computation
        @jax.jit
        def compute_single_gradient(p_flat, x_sample, rng_key):
            def model_fn(p):
                params_unflat = unravel_fn(p)
                output = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(output, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return output.squeeze(0)

            mu = model_fn(p_flat)
            sampled_y = mu + jax.random.normal(rng_key, mu.shape) * jnp.sqrt(sigma2)

            def log_prob_fn(p):
                mu_inner = model_fn(p)
                log_prob = (
                    -0.5 * jnp.log(2 * jnp.pi * sigma2)
                    - 0.5 * ((sampled_y - mu_inner) ** 2) / sigma2
                )
                return jnp.sum(log_prob)

            grad_vec = jax.grad(log_prob_fn)(p_flat)
            return jnp.outer(grad_vec, grad_vec)

        def compute_sample_fim(x_sample, sample_rngs):
            compute_with_x = lambda rng_key: compute_single_gradient(
                params_flat, x_sample, rng_key
            )
            mc_fims = jax.vmap(compute_with_x)(sample_rngs)
            return mc_fims.mean(axis=0)

        # Vectorize over all training samples
        fim = jax.vmap(compute_sample_fim)(training_data, rngs).sum(axis=0)

        fim /= n_samples
        return fim

    def compute_hvp(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the Fisher-vector product (FVP) using the Fisher Information Matrix.
        This is analogous to Hessian-vector product but uses the Fisher Information.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values (not used for Fisher, kept for interface compatibility).
            loss_fn: Loss function to determine task type.
            vector: Vector to multiply with the Fisher Information Matrix.

        Returns:
            FVP result as a 1D array.
        """
        training_data = jnp.asarray(training_data)

        if get_loss_name(loss_fn) == "cross_entropy":
            return self._compute_classification_fvp(
                model, params, training_data, vector
            )
        else:
            return self._compute_regression_fvp(model, params, training_data, vector)

    def _compute_classification_fvp(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Fisher-vector product for classification using efficient JVP.
        """
        params_flat, unravel_fn = flatten_util.ravel_pytree(params)

        # Create all random keys upfront
        rng = jax.random.PRNGKey(0)
        n_samples = training_data.shape[0]
        rngs = jax.random.split(rng, n_samples * self.samples_per_input)
        rngs = rngs.reshape(n_samples, self.samples_per_input, -1)

        # JIT compile the per-sample-per-mc-sample computation
        @jax.jit
        def compute_single_fvp(p_flat, x_sample, rng_key, v):
            def logits_fn(p):
                params_unflat = unravel_fn(p)
                logits = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(logits, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return logits[0]

            logits = logits_fn(p_flat)
            sampled_y = jax.random.categorical(rng_key, logits)

            def log_prob_fn(p):
                logits_inner = logits_fn(p)
                log_probs = jax.nn.log_softmax(logits_inner, axis=-1)
                return log_probs[sampled_y]

            grad_vec = jax.grad(log_prob_fn)(p_flat)
            dot_product = jnp.dot(grad_vec, v)
            return grad_vec * dot_product

        def compute_sample_fvp(x_sample, sample_rngs):
            compute_with_x = lambda rng_key: compute_single_fvp(
                params_flat, x_sample, rng_key, vector
            )
            mc_fvps = jax.vmap(compute_with_x)(sample_rngs)
            return mc_fvps.mean(axis=0)

        # Vectorize over all training samples
        fvp = jax.vmap(compute_sample_fvp)(training_data, rngs).sum(axis=0)

        fvp /= n_samples
        return fvp

    def _compute_regression_fvp(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Fisher-vector product for regression using efficient JVP.
        """
        params_flat, unravel_fn = flatten_util.ravel_pytree(params)
        sigma2 = 1.0

        # Create all random keys upfront
        rng = jax.random.PRNGKey(0)
        n_samples = training_data.shape[0]
        rngs = jax.random.split(rng, n_samples * self.samples_per_input)
        rngs = rngs.reshape(n_samples, self.samples_per_input, -1)

        # JIT compile the per-sample-per-mc-sample computation
        @jax.jit
        def compute_single_fvp(p_flat, x_sample, rng_key, v):
            def model_fn(p):
                params_unflat = unravel_fn(p)
                output = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(output, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return output.squeeze(0)

            mu = model_fn(p_flat)
            sampled_y = mu + jax.random.normal(rng_key, mu.shape) * jnp.sqrt(sigma2)

            def log_prob_fn(p):
                mu_inner = model_fn(p)
                log_prob = (
                    -0.5 * jnp.log(2 * jnp.pi * sigma2)
                    - 0.5 * ((sampled_y - mu_inner) ** 2) / sigma2
                )
                return jnp.sum(log_prob)

            grad_vec = jax.grad(log_prob_fn)(p_flat)
            dot_product = jnp.dot(grad_vec, v)
            return grad_vec * dot_product

        def compute_sample_fvp(x_sample, sample_rngs):
            compute_with_x = lambda rng_key: compute_single_fvp(
                params_flat, x_sample, rng_key, vector
            )
            mc_fvps = jax.vmap(compute_with_x)(sample_rngs)
            return mc_fvps.mean(axis=0)

        # Vectorize over all training samples
        fvp = jax.vmap(compute_sample_fvp)(training_data, rngs).sum(axis=0)

        fvp /= n_samples
        return fvp
