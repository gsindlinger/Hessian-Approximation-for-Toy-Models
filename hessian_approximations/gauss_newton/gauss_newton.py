from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from hessian_approximations.hessian_approximations import HessianApproximation
from models.dataclasses.hessian_compute_context import HessianComputeContext
from models.utils.loss import get_loss_name


@dataclass
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

    @override
    def compute_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the Generalized Gauss-Newton approximation of the Hessian.
        """

        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )
        damping = damping if damping is not None else 0.0
        if get_loss_name(self.model_context.loss) == "cross_entropy":
            return self._compute_crossentropy_gnh(compute_data, damping)
        else:
            return self._compute_mse_gnh(compute_data, damping)

    @override
    def compute_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Gauss-Newton-vector product (GNVP).
        """
        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )

        damping = damping if damping is not None else 0.0
        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        if get_loss_name(self.model_context.loss) == "cross_entropy":
            result_2D = self._compute_crossentropy_gnvp(
                compute_data, vectors_2D, damping
            )
        else:
            result_2D = self._compute_mse_gnvp(compute_data, vectors_2D, damping)
        return result_2D.squeeze(0) if is_single else result_2D

    @override
    def compute_ihvp(
        self,
        vectors: jnp.ndarray,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the inverse Gauss-Newton-vector product (GNVP).
        """
        compute_data = HessianComputeContext.get_data_and_params_for_hessian(
            self.model_context
        )
        damping = damping if damping is not None else 0.0

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        result_2D = self._compute_ihvp_batched(
            compute_data,
            vectors_2D,
            damping,
            get_loss_name(self.model_context.loss),
        )
        return result_2D.squeeze(0) if is_single else result_2D

    @staticmethod
    @partial(jax.jit, static_argnames=("loss_name",))
    def _compute_ihvp_batched(
        compute_data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
        loss_name: str,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Compute inverse Gauss-Newton-vector product (IHVP) using JAX's automatic differentiation.

        Args:
            model: The Flax model.
            params: Model parameters (PyTree structure).
            training_data: Input data.
            training_targets: Target values.
            loss_fn: Loss function (e.g., mse_loss or cross_entropy_loss).
            vector: Vector to multiply with the inverse Gauss-Newton matrix.

        Returns:
            IHVP result as a 1D array.
        """
        if loss_name == "cross_entropy":
            gnh = GaussNewton._compute_crossentropy_gnh(compute_data, damping)
        else:
            gnh = GaussNewton._compute_mse_gnh(compute_data, damping)

        return jnp.linalg.solve(gnh, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_mse_gnh(
        compute_data: HessianComputeContext, damping: Float
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute GGN for MSE loss with reduction='mean'.

        For MSE loss L = (1/n) Σ ||f(x_i) - y_i||²:
        The Hessian of L w.r.t. output f(x_i) is:
        ∂²L/∂f² = (2/n) I

        Therefore: H_GGN = (2/n) Σ J_i^T J_i
        where J_i is the Jacobian of f(x_i) w.r.t. parameters
        """
        d_out = (
            compute_data.training_targets.shape[1]
            if compute_data.training_targets.ndim > 1
            else 1
        )

        # JIT compile the per-sample computation
        @jax.jit
        def compute_sample_contribution(p_flat, x_sample):
            def model_output_fn(p):
                params_unflat = compute_data.unravel_fn(p)
                output = compute_data.model_apply_fn(
                    params_unflat, jnp.expand_dims(x_sample, axis=0)
                )
                if not isinstance(output, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return output.flatten()

            jacobian = jax.jacfwd(model_output_fn)(p_flat)
            return jacobian.T @ jacobian

        # Vectorized computation over all samples using vmap
        gnh = jax.vmap(
            lambda x: compute_sample_contribution(compute_data.params_flat, x)
        )(compute_data.training_data).sum(axis=0)

        n_samples = compute_data.training_data.shape[0]
        gnh *= 2.0 / (n_samples * d_out)

        return gnh + damping * jnp.eye(gnh.shape[0])

    @staticmethod
    @jax.jit
    def _compute_crossentropy_gnh(
        compute_data: HessianComputeContext, damping: Float
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute GGN for Cross-Entropy loss.

        For cross-entropy with softmax output:
        H_L = diag(p) - p p^T  (Hessian of loss w.r.t. logits)
        where p is the softmax probability vector

        H_GGN = (1/n) Σ J_i^T H_L J_i
        """

        # JIT compile the per-sample computation
        @jax.jit
        def compute_sample_contribution(p_flat, x_sample):
            def logits_fn(p):
                params_unflat = compute_data.unravel_fn(p)
                logits = compute_data.model_apply_fn(
                    params_unflat, jnp.expand_dims(x_sample, axis=0)
                )
                if not isinstance(logits, jnp.ndarray):
                    raise ValueError("Model output is not a JAX array.")
                return logits.squeeze(0)

            logits = logits_fn(p_flat)
            probs = jax.nn.softmax(logits, axis=-1)

            jacobian = jax.jacfwd(logits_fn)(p_flat)
            H_L = jnp.diag(probs) - jnp.outer(probs, probs)

            return jacobian.T @ H_L @ jacobian

        # Vectorized computation over all samples using vmap
        gnh = jax.vmap(
            lambda x: compute_sample_contribution(compute_data.params_flat, x)
        )(compute_data.training_data).sum(axis=0)

        gnh /= compute_data.training_data.shape[0]
        return gnh + damping * jnp.eye(gnh.shape[0])

    @staticmethod
    @jax.jit
    def _compute_crossentropy_gnvp(
        compute_data: HessianComputeContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> jnp.ndarray:
        """
        Efficient matrix-free cross-entropy GNVP computation.
        Now supports batched vectors with shape (B, P).
        """

        params_flat = compute_data.params_flat
        X = compute_data.training_data

        # Ensure vectors are of shape batch_size x n_params
        if vectors.ndim == 1:
            vectors = vectors[None, :]

        def logits_fn(p, x_sample):
            params_unflat = compute_data.unravel_fn(p)
            logits = compute_data.model_apply_fn(params_unflat, x_sample[None])
            return logits.squeeze(0)

        def compute_sample_vector_gnvp(p_flat, x_sample, v):
            logits = logits_fn(p_flat, x_sample)
            probs = jax.nn.softmax(logits, axis=-1)
            H_L = jnp.diag(probs) - jnp.outer(probs, probs)

            _, jvp_result = jax.jvp(
                lambda p: logits_fn(p, x_sample), (p_flat,), (v,)
            )  # J @ v
            h_jv = H_L @ jvp_result  # H_L @ (J v)
            vjp_fn = jax.vjp(lambda p: logits_fn(p, x_sample), p_flat)[1]  # J^T
            jt_h_jv = vjp_fn(h_jv)[0]  # J^T @ (H_L @ (J v))

            return jt_h_jv + damping * v

        # Vectorize over samples and vector batch dimension
        batched_gnvp = jax.vmap(
            jax.vmap(
                lambda x, v: compute_sample_vector_gnvp(params_flat, x, v),
                in_axes=(0, None),
            ),
            in_axes=(None, 0),
        )(X, vectors)
        return batched_gnvp.mean(axis=1)

    @staticmethod
    @jax.jit
    def _compute_mse_gnvp(
        compute_data: HessianComputeContext,
        vectors: jnp.ndarray,  # (P,) or (B, P)
        damping: Float,
    ) -> jnp.ndarray:
        """
        Efficient matrix-free MSE GNVP computation.
        GNVP(v) = (2/n) Σ_i J_i^T J_i v  +  damping * v
        """

        params_flat = compute_data.params_flat
        X = compute_data.training_data
        y = compute_data.training_targets

        # output dimension factor for MSE Hessian scaling
        d_out = y.shape[1] if y.ndim > 1 else 1

        # ensure vectors is (B, P)
        if vectors.ndim == 1:
            vectors = vectors[None, :]

        def model_output_fn(p, x):
            params_unflat = compute_data.unravel_fn(p)
            out = compute_data.model_apply_fn(params_unflat, x[None])
            return out.flatten()  # shape (d_out,)

        def compute_sample_vector_gnvp(p_flat, x_sample, v):
            """Compute J^T J v for one sample and one vector."""
            # Forward-mode: Jv
            _, jvp_res = jax.jvp(
                lambda p: model_output_fn(p, x_sample), (p_flat,), (v,)
            )

            # Reverse-mode: J^T(Jv)
            vjp_fn = jax.vjp(lambda p: model_output_fn(p, x_sample), p_flat)[1]
            return vjp_fn(jvp_res)[0]

        # vmap over samples, then over vector batch
        gnvp = jax.vmap(
            jax.vmap(
                lambda x, v: compute_sample_vector_gnvp(params_flat, x, v),
                in_axes=(0, None),
            ),
            in_axes=(None, 0),
        )(X, vectors)  # → (B, n_samples, n_params)

        # average and scale by 2/(n * d_out)
        gnvp = gnvp.mean(axis=1) * (2.0 / d_out)

        # Correct Gauss–Newton damping
        return gnvp + damping * vectors
