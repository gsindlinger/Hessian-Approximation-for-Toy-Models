from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import ModelBasedHessianEstimator
from src.hessians.layer_matrix import LayerMatrix, LayerVector
from src.hessians.utils.data import ModelContext, layer_shapes_from_model_context
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

    Note: this estimator materializes the full GNH and slices it into per-layer
    `(I_i*O_i, I_j*O_j)` `DenseBlock`s via `LayerMatrix.from_dense`.  For big
    models where materialization is not affordable, the lazy `_compute_gnhvp`
    helper below remains available — a future big-model subclass can override
    `_estimate_hvp` to call it directly and bypass `LayerMatrix` entirely.
    """

    # ------------------------------------------------------------------
    # LayerMatrix construction
    # ------------------------------------------------------------------

    def get_layer_names(self) -> List[str]:
        return list(self.compute_context.model.get_layer_names())

    def _layer_shapes(self) -> Dict[str, Tuple[int, int]]:
        return layer_shapes_from_model_context(self.compute_context)

    def _get_layer_matrix(self) -> LayerMatrix:
        """Materialize the full GNH and slice it into per-layer DenseBlocks."""
        loss_name = get_loss_name(self.compute_context.loss_fn)
        if loss_name == "mse":
            dense = self._compute_gnh_mse(self.compute_context, 0.0)
        elif loss_name == "cross_entropy":
            dense = self._compute_gnh_cross_entropy(self.compute_context, 0.0)
        else:
            dense = self._compute_gnh(self.compute_context, 0.0)
        return LayerMatrix.from_dense(
            dense,
            param_groups=self.get_layer_names(),
            layer_shapes=self._layer_shapes(),
        )

    # ------------------------------------------------------------------
    # HessianEstimator interface (thin wrappers over LayerMatrix)
    # ------------------------------------------------------------------

    def _estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        d = 0.0 if damping is None else damping
        return self._get_layer_matrix().damped(d).to_dense()

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        d = 0.0 if damping is None else damping
        gnh = self._estimate_hessian(d)
        return metric.compute_fn()(comparison_matrix, gnh)

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        d = 0.0 if damping is None else damping
        lmat = self._get_layer_matrix().damped(d)
        lvec = LayerVector.from_flat(
            flat=vectors,
            shapes=lmat.vector_shapes(),
            param_groups=self.get_layer_names(),
        )
        return (lmat @ lvec).to_flat()

    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
        pseudo_inverse_factor: Optional[float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        d = 0.0 if damping is None else damping
        p = 0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
        lmat = self._get_layer_matrix().inverse(
            damping=d, pseudo_inverse_factor=p
        )
        lvec = LayerVector.from_flat(
            flat=vectors,
            shapes=lmat.vector_shapes(),
            param_groups=self.get_layer_names(),
        )
        return (lmat @ lvec).to_flat()

    # ------------------------------------------------------------------
    # Materialization helpers (used by _get_layer_matrix)
    # ------------------------------------------------------------------

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
            JtJ = jax.vmap(jvp_fn)(jnp.eye(p_flat.size, dtype=p_flat.dtype))

            # Divide by n_outputs because J^T @ I @ J sums over output dimension
            return JtJ / n_outputs

        def scan_body(carry, x_i):
            p_flat, G = carry
            G_i = per_sample_gn(p_flat, x_i)
            return (p_flat, G + G_i), None

        p_flat = compute_context.params_flat
        X = compute_context.inputs
        X = X.astype(jnp.float64)
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
            return jax.vmap(jvp_fn)(jnp.eye(p_flat.size, dtype=p_flat.dtype))

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

        def model_out_single(p_flat, x):
            params_unflat = compute_context.unravel_fn(p_flat)
            return compute_context.model_apply_fn(params_unflat, x[None, ...]).squeeze(
                0
            )

        def loss_wrt_output(z, y):
            return compute_context.loss_fn(z[None, ...], y[None, ...])

        @jax.jit
        def per_sample_gn(p_flat, x_i, y_i):
            z = model_out_single(p_flat, x_i)
            H_z = jax.hessian(lambda z_: loss_wrt_output(z_, y_i))(z)

            # Compute J.T @ H_z @ J without materializing J
            def jvp_fn(v):
                # J @ v (forward mode)
                _, Jv = jax.jvp(lambda p: model_out_single(p, x_i), (p_flat,), (v,))
                # H_z @ (J @ v)
                HJv = H_z @ Jv
                # J.T @ (H_z @ J @ v) (backward mode)
                return jax.vjp(lambda p: model_out_single(p, x_i), p_flat)[1](HJv)[0]

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

    # ------------------------------------------------------------------
    # Lazy HVP escape hatch (retained for future big-model overrides;
    # not used by the refactored `_estimate_hvp` path).
    # ------------------------------------------------------------------

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

        Currently unused — retained as the lazy HVP escape hatch for a future
        big-model subclass that overrides `_estimate_hvp` to bypass
        `LayerMatrix`.
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
