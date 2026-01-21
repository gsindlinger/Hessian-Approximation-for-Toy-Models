from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import ModelBasedHessianEstimator
from src.hessians.utils.data import ModelContext
from src.utils.loss import get_loss_name
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class GNHComputer(ModelBasedHessianEstimator):
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

    def _estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> jnp.ndarray:
        """
        Compute the Generalized Gauss-Newton approximation of the Hessian.
        """

        damping = damping if damping is not None else 0.0
        if get_loss_name(self.compute_context.loss_fn) == "mse":
            return self._compute_gnh_mse(self.compute_context, damping)
        elif get_loss_name(self.compute_context.loss_fn) == "cross_entropy":
            return self._compute_gnh_cross_entropy(self.compute_context, damping)
        else:
            return self._compute_gnh(self.compute_context, damping)

    def _compare_full_hessian_estimates(
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

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Gauss-Newton-vector product (GNVP).
        """
        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        result_2D = self._compute_gnhvp(
            self.compute_context,
            vectors_2D,
            damping=0.0 if damping is None else damping,
        )
        return result_2D.squeeze(0) if is_single else result_2D

    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
        pseudo_inverse_factor: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Compute the inverse Gauss-Newton-vector product (GNVP).
        """
        result = self._compute_ignhvp_batched(
            data=self.compute_context,
            vectors=vectors,
            damping=0.0 if damping is None else damping,
            pseudo_inverse_factor=(
                0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
            ),
        )
        return result

    @staticmethod
    @jax.jit
    def _compute_ignhvp_batched(
        data: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
        pseudo_inverse_factor: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Compute inverse Gauss-Newton-vector products for a batch of vectors.
        """
        gnh = GNHComputer._compute_gnh(data, damping)
        if pseudo_inverse_factor > 0.0:
            jax.config.update("jax_enable_x64", True)
            eigvals, eigvecs = jnp.linalg.eigh(gnh)
            jax.config.update("jax_enable_x64", False)
            eigvals_inv = jnp.where(
                jnp.abs(eigvals) > pseudo_inverse_factor, 1.0 / eigvals, 0.0
            )
            return (eigvecs * eigvals_inv) @ (eigvecs.T @ vectors.T).T
        else:
            return jnp.linalg.solve(gnh, vectors.T).T

    @staticmethod
    @jax.jit
    def _compute_gnh_mse(
        compute_context: ModelContext, damping: Float
    ) -> Float[Array, "n_params n_params"]:
        """
        GNH for MSE loss: L = (1/2)||f(x) - y||²
        ∇²_z L = I (constant!)
        GNH = (1/n) Σ J_i^T J_i + λI
        """

        def model_out(p_flat, x):
            params_unflat = compute_context.unravel_fn(p_flat)
            return compute_context.model_apply_fn(params_unflat, x[None, ...]).squeeze(
                0
            )

        @jax.jit
        def per_sample_gn(p_flat, x_i):
            # Get number of outputs
            z = model_out(p_flat, x_i)
            n_outputs = z.size

            # For MSE with mean reduction: H_z = 2/n I
            # J^T @ I @ J sums over outputs, so scale by 2/(n * n_outputs)
            def jvp_fn(v):
                # J @ v (forward mode)
                _, Jv = jax.jvp(lambda p: model_out(p, x_i), (p_flat,), (v,))
                # J^T @ (J @ v) (backward mode)
                return jax.vjp(lambda p: model_out(p, x_i), p_flat)[1](Jv)[0]

            # Build J^T @ J by computing columns
            JtJ = jax.vmap(jvp_fn)(jnp.eye(p_flat.size))

            # Divide by n_outputs because J^T @ I @ J sums over output dimension
            return JtJ / n_outputs

        def scan_body(carry, x_i):
            p_flat, G = carry
            G_i = per_sample_gn(p_flat, x_i)
            return (p_flat, G + G_i), None

        p_flat = compute_context.params_flat
        X = compute_context.inputs
        n_params = p_flat.size

        G0 = jnp.zeros((n_params, n_params))
        (_, G_full), _ = jax.lax.scan(scan_body, init=(p_flat, G0), xs=X)

        G_full = 2 * G_full / X.shape[0]
        return G_full + damping * jnp.eye(n_params)

    @staticmethod
    @jax.jit
    def _compute_gnh_cross_entropy(
        compute_context: ModelContext, damping: Float
    ) -> Float[Array, "n_params n_params"]:
        """
        GNH for cross-entropy loss with softmax.
        For classification: ∇²_z L = diag(p) - p p^T where p = softmax(z)
        """

        def model_out(p_flat, x):
            params_unflat = compute_context.unravel_fn(p_flat)
            return compute_context.model_apply_fn(params_unflat, x[None, ...]).squeeze(
                0
            )

        @jax.jit
        def per_sample_gn(p_flat, x_i, y_i):
            # Get logits and compute softmax probabilities
            logits = model_out(p_flat, x_i)
            probs = jax.nn.softmax(logits)

            # Hessian of cross-entropy w.r.t. logits: H_z = diag(p) - p p^T
            H_z = jnp.diag(probs) - jnp.outer(probs, probs)

            # Compute J^T @ H_z @ J without materializing J
            def jvp_fn(v):
                # J @ v (forward mode)
                _, Jv = jax.jvp(lambda p: model_out(p, x_i), (p_flat,), (v,))
                # H_z @ (J @ v)
                HJv = H_z @ Jv
                # J^T @ (H_z @ J @ v) (backward mode)
                return jax.vjp(lambda p: model_out(p, x_i), p_flat)[1](HJv)[0]

            # Build J^T @ H_z @ J by computing columns
            return jax.vmap(jvp_fn)(jnp.eye(p_flat.size))

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

        G_full = G_full / X.shape[0]
        return G_full + damping * jnp.eye(n_params)

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
            z = model_out(p_flat, x_i)
            H_z = jax.hessian(lambda z_: loss_wrt_output(z_, y_i))(z)

            # Compute J.T @ H_z @ J without materializing J
            def jvp_fn(v):
                # J @ v (forward mode)
                _, Jv = jax.jvp(lambda p: model_out(p, x_i), (p_flat,), (v,))
                # H_z @ (J @ v)
                HJv = H_z @ Jv
                # J.T @ (H_z @ J @ v) (backward mode)
                return jax.vjp(lambda p: model_out(p, x_i), p_flat)[1](HJv)[0]

            # Build GNH by computing columns
            return jax.vmap(jvp_fn)(jnp.eye(p_flat.size))

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
    def _compute_gnhvp(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: float,
    ) -> jnp.ndarray:
        """
        Efficient Gauss-Newton vector product (GNVP) computation.

        - Handles mse and cross_entropy losses analytically.
        - Avoids nested Hessian inside scans.
        - Uses vmap over vectors with chunking for memory efficiency.
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets
        loss_name = get_loss_name(compute_context.loss_fn)

        def model_out(p, x):
            params = compute_context.unravel_fn(p)
            return compute_context.model_apply_fn(params, x[None, ...]).squeeze(0)

        # ------------------------------------------------------------
        # Per-vector GNVP function
        # ------------------------------------------------------------

        if loss_name == "mse":

            @jax.jit
            def gnvp_single(v):
                def body_fn(accum, x_i):
                    # J @ v and then J.T @ Jv
                    _, Jv = jax.jvp(lambda p: model_out(p, x_i), (p_flat,), (v,))
                    JTJv = jax.vjp(lambda p: model_out(p, x_i), p_flat)[1](Jv)[0]
                    return accum + JTJv, None

                summed, _ = jax.lax.scan(body_fn, jnp.zeros_like(v), X)
                # Effective output Hessian = identity scaled by constant for MSE
                z0 = model_out(p_flat, X[0])
                n_outputs = z0.size
                gnvp = (2.0 / (X.shape[0] * n_outputs)) * summed
                return gnvp + damping * v

        elif loss_name == "cross_entropy":

            @jax.jit
            def gnvp_single(v):
                def body_fn(accum, xy):
                    x_i, y_i = xy
                    logits = model_out(p_flat, x_i)
                    probs = jax.nn.softmax(logits)
                    _, Jv = jax.jvp(lambda p: model_out(p, x_i), (p_flat,), (v,))

                    # Analytical H_z @ Jv
                    HJv = probs * Jv - probs * jnp.dot(probs, Jv)

                    JT_HJv = jax.vjp(lambda p: model_out(p, x_i), p_flat)[1](HJv)[0]
                    return accum + JT_HJv, None

                summed, _ = jax.lax.scan(body_fn, jnp.zeros_like(v), (X, Y))
                gnvp = summed / X.shape[0]
                return gnvp + damping * v

        else:
            # Fallback: only for arbitrary losses, not recommended for large outputs
            @jax.jit
            def compute_output_hessians():
                def loss_wrt_output(z, y):
                    return compute_context.loss_fn(z, y)

                def compute_hz(x_i, y_i):
                    z = model_out(p_flat, x_i)
                    return jax.hessian(lambda z_: loss_wrt_output(z_, y_i))(z)

                return jax.vmap(compute_hz)(X, Y)

            H_z_all = compute_output_hessians()

            @jax.jit
            def gnvp_single(v):
                def body_fn(accum, data):
                    x_i, H_z_i = data
                    _, Jv = jax.jvp(lambda p: model_out(p, x_i), (p_flat,), (v,))
                    HJv = H_z_i @ Jv
                    JT_HJv = jax.vjp(lambda p: model_out(p, x_i), p_flat)[1](HJv)[0]
                    return accum + JT_HJv, None

                summed, _ = jax.lax.scan(body_fn, jnp.zeros_like(v), (X, H_z_all))
                gnvp = summed / X.shape[0]
                return gnvp + damping * v

        # ------------------------------------------------------------
        # Chunking vectors to avoid OOM
        # ------------------------------------------------------------
        CHUNK_SIZE = 32
        n_vectors = vectors.shape[0]

        if n_vectors <= CHUNK_SIZE:
            return jax.vmap(gnvp_single)(vectors)
        else:
            outs = []
            for i in range(0, n_vectors, CHUNK_SIZE):
                chunk = vectors[i : i + CHUNK_SIZE]
                outs.append(jax.vmap(gnvp_single)(chunk))
            return jnp.concatenate(outs, axis=0)
