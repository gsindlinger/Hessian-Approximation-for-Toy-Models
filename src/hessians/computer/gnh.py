from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.utils.data import ModelContext
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class GNHComputer(HessianEstimator):
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

    compute_context: ModelContext

    def estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the Generalized Gauss-Newton approximation of the Hessian.
        """

        damping = damping if damping is not None else 0.0
        return self._compute_gnh(self.compute_context, damping)

    def compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare the Gauss-Newton Hessian with another Hessian matrix using the specified metric.
        """
        damping = 0.0 if damping is None else damping

        return metric.compute_fn()(
            comparison_matrix,
            self._compute_gnh(self.compute_context, damping),
        )

    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Gauss-Newton-vector product (GNVP).
        """
        damping = damping if damping is not None else 0.0
        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        result_2D = self._compute_gnhvp(self.compute_context, vectors_2D, damping)
        return result_2D.squeeze(0) if is_single else result_2D

    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the inverse Gauss-Newton-vector product (GNVP).
        """
        result = self._compute_ignhvp_batched(
            data=self.compute_context,
            vectors=vectors,
            damping=0.0 if damping is None else damping,
        )
        return result

    @staticmethod
    @jax.jit
    def _compute_ignhvp_batched(
        data: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Compute inverse Gauss-Newton-vector products for a batch of vectors.
        """
        gnh = GNHComputer._compute_gnh(data, damping)
        return jnp.linalg.solve(gnh, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_gnh(
        compute_context: ModelContext, damping: Float
    ) -> Float[Array, "n_params n_params"]:
        """
        Computes full Gauss-Newton Hessian for any loss w.r.t. outputs.
        """

        def model_out(p_flat, x):
            params_unflat = compute_context.unravel_fn(p_flat)
            return compute_context.model_apply_fn(params_unflat, x[None, ...]).squeeze(
                0
            )

        def loss_wrt_output(z, y):
            return compute_context.loss_fn(z, y)

        @jax.jit
        def per_sample_gn(p_flat, x_i, y_i):
            z = model_out(p_flat, x_i)  # model output for sample i
            H_z = jax.hessian(lambda z_: loss_wrt_output(z_, y_i))(z)

            def logits_fn(p):
                return model_out(p, x_i)  # model output function

            J = jax.jacrev(logits_fn)(p_flat)  # Jacobian of model output w.r.t. params

            return J.T @ H_z @ J

        # Loop through data
        def scan_body(carry, xy):
            p_flat, G = carry
            x_i, y_i = xy
            G_i = per_sample_gn(p_flat, x_i, y_i)
            return (p_flat, G + G_i), None

        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets
        n_params = p_flat.size

        G0 = jnp.zeros((n_params, n_params))

        (_, G_full), _ = jax.lax.scan(scan_body, init=(p_flat, G0), xs=(X, Y))

        # Average over dataset + damping
        G_full = G_full / X.shape[0]
        return G_full + damping * jnp.eye(n_params)

    @staticmethod
    @jax.jit
    def _compute_gnhvp(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: float,
    ) -> jnp.ndarray:
        """
        Minimal memory version: double scan over vectors AND data samples.
        Processes one vector and one data sample at a time.

        Memory usage: O(n_params) instead of O(batch_size * n_samples * n_outputs)
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets
        n_samples = X.shape[0]

        def model_out(p, x):
            params = compute_context.unravel_fn(p)
            return compute_context.model_apply_fn(params, x[None, ...]).squeeze(0)

        def loss_wrt_output(z, y):
            return compute_context.loss_fn(z, y)

        def gnhvp_single_data_single_vector(x_i, y_i, v):
            """
            Compute GN-vector product for ONE data point and ONE vector.
            This is the minimal memory unit.
            """
            z = model_out(p_flat, x_i)
            H_z = jax.hessian(lambda z_: loss_wrt_output(z_, y_i))(z)

            def logits_fn(p):
                return model_out(p, x_i)

            # Use JVP/VJP / never materialize Jacobian
            _, Jv = jax.jvp(logits_fn, (p_flat,), (v,))  # [n_outputs]
            HJv = H_z @ Jv  # [n_outputs]
            JT_HJv = jax.vjp(logits_fn, p_flat)[1](HJv)[0]  # [n_params]

            return JT_HJv

        def process_single_vector(v):
            """Process one vector across all data samples."""

            # Scan over data samples for this vector
            def data_scan_body(accum, xy):
                x_i, y_i = xy
                contribution = gnhvp_single_data_single_vector(x_i, y_i, v)
                return accum + contribution, None

            result, _ = jax.lax.scan(data_scan_body, jnp.zeros_like(v), (X, Y))
            return result

        # Scan over vectors
        def vector_scan_body(results_accum, i):
            v = vectors[i]
            result = process_single_vector(v)
            # Update the i-th row of results
            results_accum = results_accum.at[i].set(result)
            return results_accum, None

        n_vectors = vectors.shape[0]
        results, _ = jax.lax.scan(
            vector_scan_body, jnp.zeros_like(vectors), jnp.arange(n_vectors)
        )

        # Average over data samples + damping
        return results / n_samples + damping * vectors
