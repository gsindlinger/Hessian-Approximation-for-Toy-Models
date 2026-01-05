from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.utils.data import DataActivationsGradients
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class FIMComputer(HessianEstimator):
    """
    Fisher Information Matrix approximation.

    The Fisher Information Matrix is defined as:
    FIM = E[∇log p(y|x) ∇log p(y|x)^T]

    Use previously collected gradients to compute the FIM using its outer product.
    """

    compute_context: DataActivationsGradients

    def build_full_gradients(self) -> Float[Array, "n_samples n_params"]:
        grads_per_layer = []

        for layer in self.compute_context.layer_names:
            a = self.compute_context.activations[layer]  # (N, I_l)
            g = self.compute_context.gradients[layer]  # (N, O_l)

            # Per-layer parameter gradients
            # (N, I_l, O_l) -> (N, I_l * O_l)
            G_l = jnp.einsum("ni,no->nio", a, g).reshape(a.shape[0], -1)
            grads_per_layer.append(G_l)

        # (N, n_params)
        return jnp.concatenate(grads_per_layer, axis=1)

    def estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the Fisher Information Matrix from pre-collected gradients.
        """
        gradients = self.build_full_gradients()
        damping = 0.0 if damping is None else damping

        return self._compute_fim(gradients, damping)

    def compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare the Fisher Information Matrix with another Hessian matrix using the specified metric.
        """
        gradients = self.build_full_gradients()
        damping = 0.0 if damping is None else damping

        return metric.compute_fn()(
            comparison_matrix,
            self._compute_fim(gradients, damping),
        )

    @staticmethod
    @jax.jit
    def _compute_fim(
        gradients: Float[Array, "n_samples n_params"],
        damping: Float = 0.0,
    ) -> Float[Array, "n_params n_params"]:
        """
        Memory-efficient Fisher Information Matrix computation using scan.
        F = (1/N) * sum_i g_i g_i^T + damping * Eye(n_params)
        """
        n_samples = gradients.shape[0]
        n_params = gradients.shape[1]

        def body_fn(accum, g):
            return accum + jnp.outer(g, g), None

        fim0 = jnp.zeros((n_params, n_params), dtype=gradients.dtype)

        fim_sum, _ = jax.lax.scan(body_fn, fim0, gradients)
        fim = fim_sum / n_samples
        fim += damping * jnp.eye(n_params, dtype=fim.dtype)
        return fim

    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Fisher-vector product (FVP) from pre-collected gradients.
        """
        gradients = self.build_full_gradients()
        damping = 0.0 if damping is None else damping

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )
        result = self._compute_fvp(gradients, vectors_2D, damping)
        return result.squeeze(0) if is_single else result

    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the inverse Fisher-vector product (IFVP) using direct solve.
        """
        gradients = self.build_full_gradients()
        damping = 0.0 if damping is None else damping

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        result_2D = self._estimate_ifvp(gradients, vectors_2D, damping)
        return result_2D.squeeze(0) if is_single else result_2D

    @staticmethod
    @jax.jit
    def _estimate_ifvp(
        gradients: Float[Array, "n_samples n_params"],
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Solve FIM^{-1} @ v for each vector v using direct linear solve.
        """
        fim = FIMComputer._compute_fim(gradients, damping)
        return jnp.linalg.solve(fim, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_fvp(
        gradients: Float[Array, "n_samples n_params"],
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
        sample_batch_size: int = 32,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Fisher-vector product: FIM @ v for each vector v.

        Efficiently computes: (1/N) * sum_i (g_i @ v) * g_i + damping * v
        without materializing the full FIM matrix.

        Uses batched processing over samples to reduce memory overhead.

        Args:
            gradients: Pre-collected gradients (n_samples, n_params)
            vectors: Vectors to multiply (batch_size, n_params)
            damping: Damping factor
            sample_batch_size: Number of samples to process at once

        Returns:
            FVP results (batch_size, n_params)
        """
        n_samples, n_params = gradients.shape

        # Pad gradients to make batching exact
        n_batches = (n_samples + sample_batch_size - 1) // sample_batch_size
        padded_n_samples = n_batches * sample_batch_size
        pad = padded_n_samples - n_samples

        if pad > 0:
            gradients_padded = jnp.pad(gradients, ((0, pad), (0, 0)))
        else:
            gradients_padded = gradients

        # Single-vector FVP with batched sample processing
        def fvp_single(v):
            def scan_body(accum, batch_idx):
                start = batch_idx * sample_batch_size

                # Extract batch of gradients
                grad_batch = jax.lax.dynamic_slice(
                    gradients_padded,
                    (start, 0),
                    (sample_batch_size, n_params),
                )

                # Compute projections: g_i^T @ v for this batch
                projections = grad_batch @ v  # (sample_batch_size,)

                # Accumulate: sum_i (g_i^T @ v) * g_i
                fvp_batch = grad_batch.T @ projections  # (n_params,)

                return accum + fvp_batch, None

            fvp_sum, _ = jax.lax.scan(
                scan_body,
                jnp.zeros((n_params,), dtype=v.dtype),
                jnp.arange(n_batches),
            )

            # Average and add damping
            return fvp_sum / n_samples + damping * v

        # Apply to all vectors in batch
        return jax.vmap(fvp_single)(vectors)
