from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.utils.data import ModelContext
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class FIMComputer(HessianEstimator):
    """
    Fisher Information Matrix approximation.

    The Fisher Information Matrix is defined as:
    FIM = E[∇log p(y|x) ∇log p(y|x)^T]

    Two variants are supported:
    - Empirical FIM: Uses actual training data and labels
    - True FIM: Samples labels from the model's predictive distribution
    """

    compute_context: ModelContext

    def estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the Fisher Information Matrix.
        """
        damping = 0.0 if damping is None else damping
        return self._compute_fim(self.compute_context, damping)

    def compare_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare the Fisher Information Matrix with another Hessian matrix using the specified metric.
        """
        damping = 0.0 if damping is None else damping
        return metric.compute_fn()(
            comparison_matrix,
            self._compute_fim(self.compute_context, damping),
        )

    @staticmethod
    @jax.jit
    def _compute_fim(
        compute_context: ModelContext,
        damping: Float = 0.0,
    ) -> Float[Array, "n_params n_params"]:
        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), y)

        # per-sample gradient of loss
        grad_loss = jax.grad(loss_single)

        @jax.jit
        def compute_sample_fim(p_flat, x, y):
            g = grad_loss(p_flat, x, y)  # shape = (n_params,)
            return jnp.outer(g, g)  # (n_params, n_params)

        def scan_body(carry, xy):
            p_flat, F = carry
            x_i, y_i = xy
            F_i = compute_sample_fim(p_flat, x_i, y_i)
            return (p_flat, F + F_i), None

        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        F0 = jnp.zeros((p_flat.size, p_flat.size))

        (_, F_full), _ = jax.lax.scan(scan_body, init=(p_flat, F0), xs=(X, Y))

        F_full = F_full / X.shape[0]  # average

        # add damping like in your Hessian
        return F_full + damping * jnp.eye(F_full.shape[0])

    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Fisher-vector product (FVP).
        """
        damping = 0.0 if damping is None else damping

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )
        result = self._compute_fvp(self.compute_context, vectors_2D, damping)
        return result.squeeze(0) if is_single else result

    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the inverse Fisher-vector product (IFVP) using Conjugate Gradient.
        """
        damping = 0.0 if damping is None else damping

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        result_2D = self._estimate_ifvp_batched(
            self.compute_context, vectors_2D, damping
        )
        return result_2D.squeeze(0) if is_single else result_2D

    @staticmethod
    @jax.jit
    def _estimate_ifvp_batched(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        # Take the simplest approach: calculate full FIM and solve by linalg solve
        fim = FIMComputer._compute_fim(compute_context, damping)
        return jnp.linalg.solve(fim, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_fvp(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        x, y = compute_context.inputs, compute_context.targets
        p0 = compute_context.params_flat
        N = x.shape[0]

        # per-sample loss
        def loss_single(p, xi, yi):
            params = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params, xi[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), yi)

        # per-sample grads of the loss
        grads = jax.vmap(lambda xi, yi: jax.grad(loss_single)(p0, xi, yi))(
            x, y
        )  # shape (N, D)

        def fvp_single(v):
            proj = grads @ v  # (N,)
            fvp = grads.T @ proj / N  # (D,)
            return fvp + damping * v

        return jax.vmap(fvp_single)(vectors)
