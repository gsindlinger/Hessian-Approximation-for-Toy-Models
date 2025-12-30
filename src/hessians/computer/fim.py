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

    def compare_full_hessian_estimates(
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

        # add damping
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

        result_2D = self._estimate_ifvp(self.compute_context, vectors_2D, damping)
        return result_2D.squeeze(0) if is_single else result_2D

    @staticmethod
    @jax.jit
    def _estimate_ifvp(
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
        sample_batch_size: int = 64,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Fisher-vector product using dynamic slicing (JAX-safe).

        Uses:
        - lax.dynamic_slice for batching samples
        - scan over batch indices
        """

        X = compute_context.inputs
        Y = compute_context.targets
        assert Y is not None, "Targets must be provided in compute_context for FIM."

        p0 = compute_context.params_flat
        N = X.shape[0]
        D = p0.size

        # ------------------------------------------------------------
        # Pad data so all batches are full
        # ------------------------------------------------------------
        n_batches = (N + sample_batch_size - 1) // sample_batch_size
        padded_N = n_batches * sample_batch_size
        pad = padded_N - N

        Xp = jnp.pad(X, ((0, pad),) + ((0, 0),) * (X.ndim - 1))
        Yp = jnp.pad(Y, ((0, pad),) + ((0, 0),) * (Y.ndim - 1))

        def loss_single(p, xi, yi):
            params = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params, xi[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), yi)

        # Single-vector FVP
        def fvp_single(v):
            def scan_body(accum, batch_idx):
                start = batch_idx * sample_batch_size

                # Dynamic slice
                x_batch = jax.lax.dynamic_slice(
                    Xp,
                    (start,) + (0,) * (Xp.ndim - 1),
                    (sample_batch_size,) + Xp.shape[1:],
                )
                y_batch = jax.lax.dynamic_slice(
                    Yp,
                    (start,) + (0,) * (Yp.ndim - 1),
                    (sample_batch_size,) + Yp.shape[1:],
                )

                # Compute per-sample gradients
                grads = jax.vmap(lambda xi, yi: jax.grad(loss_single)(p0, xi, yi))(
                    x_batch, y_batch
                )  # (B, D)

                # Fisher contribution: sum_i (g_i^T v) g_i
                proj = grads @ v  # (B,)
                fvp_batch = grads.T @ proj  # (D,)

                return accum + fvp_batch, None

            fvp_sum, _ = jax.lax.scan(
                scan_body,
                jnp.zeros((D,), dtype=v.dtype),
                jnp.arange(n_batches),
            )

            return fvp_sum / N + damping * v

        # Vector batching
        CHUNK_SIZE = 32
        n_vectors = vectors.shape[0]

        if n_vectors <= CHUNK_SIZE:
            return jax.vmap(fvp_single)(vectors)
        else:
            outs = []
            for i in range(0, n_vectors, CHUNK_SIZE):
                outs.append(jax.vmap(fvp_single)(vectors[i : i + CHUNK_SIZE]))
            return jnp.concatenate(outs, axis=0)
