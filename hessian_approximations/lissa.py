from __future__ import annotations
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util
from typing_extensions import override

from hessian_approximations.hessian_approximations import HessianApproximation
from models.models import ApproximationModel


class LiSSA(HessianApproximation):
    """
    LiSSA approximates H^{-1} @ v using iterative sampling without computing the full Hessian.
    Based on: Agarwal et al. "Second-Order Stochastic Optimization for Machine Learning in Linear Time"

    The algorithm uses the recurrence:
    H_j = v + (I - H/scale) @ H_{j-1}
    where H is the Hessian and scale is a damping parameter.
    """

    def __init__(
        self,
        num_samples: int = 1,
        recursion_depth: int = 1000,
        scale: float = 10.0,
        damping: float = 0.0,
    ):
        """
        Args:
            num_samples: Number of independent LiSSA estimates to average
            recursion_depth: Number of iterations for each LiSSA estimate
            scale: Scaling parameter (should be > largest eigenvalue of Hessian)
            damping: Damping factor added to Hessian (for numerical stability)
        """
        super().__init__()
        self.num_samples = num_samples
        self.recursion_depth = recursion_depth
        self.scale = scale
        self.damping = damping

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
        LiSSA is designed for computing Hessian-vector products, not the full Hessian.
        Computing the full Hessian with LiSSA would require O(p) HVP calls where p is
        the number of parameters, which defeats the purpose of LiSSA.

        Use compute_hvp() instead for efficient Hessian-vector products.
        """
        raise NotImplementedError(
            "LiSSA does not support computing the full Hessian matrix. "
            "Use compute_hvp() for efficient Hessian-vector products, or "
            "use a different approximation method (e.g., Hessian, GaussNewton, FisherInformation) "
            "if you need the full Hessian matrix."
        )

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
        Compute Hessian-vector product using LiSSA algorithm.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function.
            vector: Vector to multiply with the Hessian.

        Returns:
            H @ v approximated using LiSSA
        """
        training_data = jnp.asarray(training_data)
        training_targets = jnp.asarray(training_targets)

        params_flat, unravel_fn = flatten_util.ravel_pytree(params)
        n_samples = training_data.shape[0]

        # JIT compile single HVP computation on a batch
        @jax.jit
        def compute_hvp_batch(p_flat, x_batch, y_batch, v):
            def get_predictions(x, params_unflat):
                pred = model.apply(params_unflat, jnp.expand_dims(x, axis=0))
                if not isinstance(pred, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return pred.squeeze(0)

            def batch_loss(p):
                params_unflat = unravel_fn(p)
                predictions = jax.vmap(lambda x: get_predictions(x, params_unflat))(
                    x_batch
                )
                return loss_fn(predictions, y_batch)

            # Compute HVP using forward-over-reverse mode
            _, hvp = jax.jvp(lambda p: jax.grad(batch_loss)(p), (p_flat,), (v,))
            return hvp

        # Run multiple independent LiSSA estimates
        rng = jax.random.PRNGKey(0)
        estimates = []

        for sample_idx in range(self.num_samples):
            rng, subkey = jax.random.split(rng)

            # Initialize H_0 = v
            h_estimate = vector.copy()

            # Perform recursion
            for j in range(self.recursion_depth):
                rng, subkey = jax.random.split(rng)

                # Sample a random mini-batch
                batch_indices = jax.random.choice(
                    subkey, n_samples, shape=(min(n_samples, 32),), replace=False
                )
                x_batch = training_data[batch_indices]
                y_batch = training_targets[batch_indices]

                # Compute HVP on this batch
                hvp = compute_hvp_batch(params_flat, x_batch, y_batch, h_estimate)

                # Add damping: (H + damping * I) @ h_estimate = H @ h_estimate + damping * h_estimate
                hvp = hvp + self.damping * h_estimate

                # LiSSA update: H_j = v + (I - (H + damping*I)/scale) @ H_{j-1}
                #             = v + H_{j-1} - (H @ H_{j-1} + damping * H_{j-1})/scale
                h_estimate = vector + h_estimate - hvp / self.scale

            estimates.append(h_estimate)

        # Average all estimates
        hvp_estimate = jnp.stack(estimates).mean(axis=0)

        return hvp_estimate

    def compute_inverse_hvp(
        self,
        model: ApproximationModel,
        params: Any,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute inverse Hessian-vector product using LiSSA algorithm.
        This is the primary use case for LiSSA: approximating H^{-1} @ v.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function.
            vector: Vector to multiply with the inverse Hessian.

        Returns:
            H^{-1} @ v approximated using LiSSA
        """
        training_data = jnp.asarray(training_data)
        training_targets = jnp.asarray(training_targets)

        params_flat, unravel_fn = flatten_util.ravel_pytree(params)
        n_samples = training_data.shape[0]

        # JIT compile single HVP computation on a batch
        @jax.jit
        def compute_hvp_batch(p_flat, x_batch, y_batch, v):
            def batch_loss(p):
                params_unflat = unravel_fn(p)

                def get_predictions(x, params_unflat):
                    pred = model.apply(params_unflat, jnp.expand_dims(x, axis=0))
                    if not isinstance(pred, jnp.ndarray):
                        raise ValueError("Model output is not a JAX array.")
                    return pred.squeeze(0)

                predictions = jax.vmap(lambda x: get_predictions(x, params_unflat))(
                    x_batch
                )
                return loss_fn(predictions, y_batch)

            # Compute HVP using forward-over-reverse mode
            _, hvp = jax.jvp(lambda p: jax.grad(batch_loss)(p), (p_flat,), (v,))
            return hvp

        # Run multiple independent LiSSA estimates
        rng = jax.random.PRNGKey(0)
        estimates = []

        for sample_idx in range(self.num_samples):
            rng, subkey = jax.random.split(rng)

            # Initialize H_0 = v
            h_estimate = vector.copy()

            # Perform recursion for inverse
            for j in range(self.recursion_depth):
                rng, subkey = jax.random.split(rng)

                # Sample a random mini-batch
                batch_indices = jax.random.choice(
                    subkey, n_samples, shape=(min(n_samples, 32),), replace=False
                )
                x_batch = training_data[batch_indices]
                y_batch = training_targets[batch_indices]

                # Compute HVP on this batch
                hvp = compute_hvp_batch(params_flat, x_batch, y_batch, h_estimate)

                # Add damping
                hvp = hvp + self.damping * h_estimate

                # LiSSA update for inverse: H_j = v + (I - (H + damping*I)/scale) @ H_{j-1}
                h_estimate = vector + h_estimate - hvp / self.scale

            estimates.append(h_estimate)

        # Average all estimates
        inverse_hvp = jnp.stack(estimates).mean(axis=0)

        return inverse_hvp
