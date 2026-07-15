from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import LayerMatrix
from src.hessians.utils.data import ModelContext, layer_shapes_from_model_context
from src.utils.loss import get_loss_name


@dataclass
class GNHComputer(HessianEstimator):
    compute_context: ModelContext
    """
    Gauss-Newton Hessian approximation.

    The Gauss-Newton approximation is defined as:
    GNH = J^T H_L J

    where:
    - J is the Jacobian of the model output w.r.t. parameters
    - H_L is the Hessian of the loss w.r.t. the model output

    For exponential family losses (e.g., CrossEntropy), GNH equals FIM.
    GNH is always positive semi-definite, unlike the full Hessian.

    `.build()` materializes the full GNH and slices it into per-layer
    `DenseBlock`s; all `estimate_*` methods use the cached matrix (build is
    required).
    """

    # ------------------------------------------------------------------
    # LayerMatrix construction
    # ------------------------------------------------------------------

    def get_layer_names(self) -> List[str]:
        return list(self.compute_context.model.get_layer_names())

    def _layer_shapes(self) -> Dict[str, Tuple[int, int]]:
        return layer_shapes_from_model_context(self.compute_context)

    def _build(self) -> LayerMatrix:
        """Materialize the full GNH and slice it into per-layer DenseBlocks."""
        ctx = self.compute_context
        loss_name = get_loss_name(ctx.loss_fn)
        if loss_name == "mse":
            dense = self._compute_gnh_mse(ctx, 0.0)
        elif loss_name == "cross_entropy":
            dense = self._compute_gnh_cross_entropy(ctx, 0.0)
        else:
            dense = self._compute_gnh(ctx, 0.0)
        return LayerMatrix.from_dense(
            dense,
            param_groups=self.get_layer_names(),
            layer_shapes=self._layer_shapes(),
        )

    # ------------------------------------------------------------------
    # Materialization helpers (used by _build)
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

        n_params = p_flat.size

        G0 = jnp.zeros((n_params, n_params), dtype=p_flat.dtype)
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

        G0 = jnp.zeros((n_params, n_params), dtype=p_flat.dtype)
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

        G0 = jnp.zeros((n_params, n_params), dtype=p_flat.dtype)

        (_, G_full), _ = jax.lax.scan(scan_body, init=(p_flat, G0), xs=(X, Y))

        # Average over dataset + damping
        G_full = G_full / X.shape[0]
        return G_full + damping * jnp.eye(n_params)

