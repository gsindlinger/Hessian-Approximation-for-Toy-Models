from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util
from typing_extensions import override

from hessian_approximations.hessian_approximations import HessianApproximation
from models.loss import get_loss_name
from models.train import ApproximationModel


class GaussNewton(HessianApproximation):
    """
    Gauss-Newton Hessian approximation.

    The Gauss-Newton approximation is defined as:
    GNH = J^T H_L J

    where:
    - J is the Jacobian of the model output w.r.t. parameters
    - H_L is the Hessian of the loss w.r.t. the model output

    For exponential family losses (e.g., CrossEntropy), GNH equals FIM.
    GNH is always positive semi-definite, unlike the full Hessian.
    """

    def __init__(self):
        super().__init__()

    @override
    def compute_hessian(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> jnp.ndarray:
        """
        Compute the Generalized Gauss-Newton approximation of the Hessian.
        """
        training_data = jnp.asarray(training_data)
        training_targets = jnp.asarray(training_targets)

        if get_loss_name(loss_fn) == "cross_entropy":
            return self._compute_crossentropy_gnh(
                model, params, training_data, training_targets
            )
        else:
            # Default to MSE/regression
            return self._compute_mse_gnh(model, params, training_data, training_targets)

    @override
    def compute_hvp(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the Gauss-Newton-vector product (GNVP).

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function to determine task type.
            vector: Vector to multiply with the Gauss-Newton matrix.

        Returns:
            GNVP result as a 1D array.
        """
        training_data = jnp.asarray(training_data)
        training_targets = jnp.asarray(training_targets)

        if get_loss_name(loss_fn) == "cross_entropy":
            return self._compute_crossentropy_gnvp(
                model, params, training_data, training_targets, vector
            )
        else:
            return self._compute_mse_gnvp(
                model, params, training_data, training_targets, vector
            )

    @override
    def compute_ihvp(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        loss_fn: Callable,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the inverse Gauss-Newton-vector product (GNVP).

        Note: This is a placeholder implementation. In practice, computing the
        inverse GNVP may require iterative methods or approximations.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function to determine task type.
            vector: Vector to multiply with the inverse Gauss-Newton matrix.

        Returns:
            Inverse GNVP result as a 1D array.
        """
        training_data = jnp.asarray(training_data)
        training_targets = jnp.asarray(training_targets)

        raise NotImplementedError(
            "Inverse Gauss-Newton-vector product not implemented."
        )

    def _compute_mse_gnh(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute GGN for MSE loss with reduction='mean'.

        For MSE loss L = (1/n) Σ ||f(x_i) - y_i||²:
        The Hessian of L w.r.t. output f(x_i) is:
        ∂²L/∂f² = (2/n) I

        Therefore: H_GGN = (2/n) Σ J_i^T J_i
        where J_i is the Jacobian of f(x_i) w.r.t. parameters
        """
        params_flat, unravel_fn = flatten_util.ravel_pytree(params)
        d_out = training_targets.shape[1] if training_targets.ndim > 1 else 1

        # JIT compile the per-sample computation
        @jax.jit
        def compute_sample_contribution(p_flat, x_sample):
            def model_output_fn(p):
                params_unflat = unravel_fn(p)
                output = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(output, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return output.flatten()

            jacobian = jax.jacfwd(model_output_fn)(p_flat)
            return jacobian.T @ jacobian

        # Vectorized computation over all samples using vmap
        gnh = jax.vmap(lambda x: compute_sample_contribution(params_flat, x))(
            training_data
        ).sum(axis=0)

        n_samples = training_data.shape[0]
        gnh *= 2.0 / (n_samples * d_out)

        numpy_gnh = np.array(gnh)
        return gnh

    def _compute_crossentropy_gnh(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute GGN for Cross-Entropy loss.

        For cross-entropy with softmax output:
        H_L = diag(p) - p p^T  (Hessian of loss w.r.t. logits)
        where p is the softmax probability vector

        H_GGN = (1/n) Σ J_i^T H_L J_i
        """
        params_flat, unravel_fn = flatten_util.ravel_pytree(params)

        # JIT compile the per-sample computation
        @jax.jit
        def compute_sample_contribution(p_flat, x_sample):
            def logits_fn(p):
                params_unflat = unravel_fn(p)
                logits = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(logits, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return logits.squeeze(0)

            logits = logits_fn(p_flat)
            probs = jax.nn.softmax(logits, axis=-1)

            jacobian = jax.jacfwd(logits_fn)(p_flat)
            H_L = jnp.diag(probs) - jnp.outer(probs, probs)

            return jacobian.T @ H_L @ jacobian

        # Vectorized computation over all samples using vmap
        gnh = jax.vmap(lambda x: compute_sample_contribution(params_flat, x))(
            training_data
        ).sum(axis=0)

        n_samples = training_data.shape[0]
        gnh /= n_samples
        return gnh

    def _compute_mse_gnvp(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Gauss-Newton-vector product for MSE loss efficiently.
        GNVP = (2/n) Σ J_i^T @ J_i @ v
        """
        params_flat, unravel_fn = flatten_util.ravel_pytree(params)
        d_out = training_targets.shape[1] if training_targets.ndim > 1 else 1

        # JIT compile the per-sample computation
        @jax.jit
        def compute_sample_gnvp(p_flat, x_sample, v):
            def model_output_fn(p):
                params_unflat = unravel_fn(p)
                output = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(output, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return output.flatten()

            # Use JVP for efficient computation: J @ v
            _, jvp_result = jax.jvp(model_output_fn, (p_flat,), (v,))

            # Compute J^T @ (J @ v) using VJP
            vjp_fn = jax.vjp(model_output_fn, p_flat)[1]
            return vjp_fn(jvp_result)[0]

        # Vectorized computation over all samples using vmap
        gnvp = jax.vmap(lambda x: compute_sample_gnvp(params_flat, x, vector))(
            training_data
        ).sum(axis=0)

        n_samples = training_data.shape[0]
        gnvp *= 2.0 / (n_samples * d_out)
        return gnvp

    def _compute_crossentropy_gnvp(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Gauss-Newton-vector product for cross-entropy loss efficiently.
        GNVP = (1/n) Σ J_i^T @ H_L @ J_i @ v
        """
        params_flat, unravel_fn = flatten_util.ravel_pytree(params)

        # JIT compile the per-sample computation
        @jax.jit
        def compute_sample_gnvp(p_flat, x_sample, v):
            def logits_fn(p):
                params_unflat = unravel_fn(p)
                logits = model.apply(params_unflat, jnp.expand_dims(x_sample, axis=0))
                if not isinstance(logits, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return logits.squeeze(0)

            logits = logits_fn(p_flat)
            probs = jax.nn.softmax(logits, axis=-1)
            H_L = jnp.diag(probs) - jnp.outer(probs, probs)

            # Use JVP for efficient computation: J @ v
            _, jvp_result = jax.jvp(logits_fn, (p_flat,), (v,))

            # Compute H_L @ (J @ v)
            h_jv = H_L @ jvp_result

            # Compute J^T @ (H_L @ J @ v) using VJP
            vjp_fn = jax.vjp(logits_fn, p_flat)[1]
            return vjp_fn(h_jv)[0]

        # Vectorized computation over all samples using vmap
        gnvp = jax.vmap(lambda x: compute_sample_gnvp(params_flat, x, vector))(
            training_data
        ).sum(axis=0)

        n_samples = training_data.shape[0]
        gnvp /= n_samples
        return gnvp
