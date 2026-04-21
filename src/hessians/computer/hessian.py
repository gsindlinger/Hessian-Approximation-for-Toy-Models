from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import LayerMatrix
from src.hessians.utils.data import ModelContext, layer_shapes_from_model_context


@dataclass
class HessianComputer(HessianEstimator):
    compute_context: ModelContext
    """
    Exact Hessian computation via JAX automatic differentiation.

    Two modes:
    - `.build()` materializes the full `(n_params, n_params)` Hessian and
      slices it into per-layer `DenseBlock`s via `LayerMatrix.from_dense`.
      `estimate_hessian` / `estimate_hvp` / `estimate_ihvp` then use the
      cached matrix.
    - Skip `.build()` → `estimate_hvp` falls through to `_lazy_hvp`, which
      does JVP-through-the-model via `_compute_hvp` and never materializes
      the dense matrix.  `estimate_hessian` / `estimate_ihvp` still require
      `.build()`.
    """

    @staticmethod
    def get_param_index_mapping(params: Dict):
        """
        Build a mapping from parameter names to index ranges in the flattened vector.
        Helps to debug by identifying which parameter in the flattened parameter array corresponds to which entry.
        Returns:
            dict[str, tuple[int, int]] mapping each param path to (start, end)
        """
        leaves, _ = jax.tree_util.tree_flatten(params)
        flat_params, _ = flatten_util.ravel_pytree(params)

        index_mapping = {}
        idx = 0

        # Traverse parameter tree with paths
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
            path_str = "/".join(str(k) for k in path)
            size = leaf.size
            index_mapping[path_str] = (idx, idx + size)
            idx += size

        return index_mapping, flat_params.size

    # ------------------------------------------------------------------
    # LayerMatrix construction
    # ------------------------------------------------------------------

    def get_layer_names(self) -> List[str]:
        return list(self.compute_context.model.get_layer_names())

    def _layer_shapes(self) -> Dict[str, Tuple[int, int]]:
        return layer_shapes_from_model_context(self.compute_context)

    def _build(self) -> LayerMatrix:
        """Materialize the full Hessian and slice it into per-layer DenseBlocks."""
        dense = self._compute_hessian(self.compute_context, 0.0)
        return LayerMatrix.from_dense(
            dense,
            param_groups=self.get_layer_names(),
            layer_shapes=self._layer_shapes(),
        )

    # ------------------------------------------------------------------
    # Materialization helper (used by `_build`)
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _compute_hessian(
        compute_context: ModelContext,
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds, y[None, ...])

        # This computes the per-sample Hessian: ∂²L_i/∂θ²
        @jax.jit
        def compute_sample_hessian(p_flat, x, y):
            return jax.hessian(lambda p: loss_single(p, x, y))(p_flat)

        def scan_body(carry, xy):
            p_flat, H = carry
            x_i, y_i = xy
            H_i = compute_sample_hessian(p_flat, x_i, y_i)
            return (p_flat, H + H_i), None

        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        H0 = jnp.zeros((p_flat.size, p_flat.size))

        (_, H_full), _ = jax.lax.scan(scan_body, init=(p_flat, H0), xs=(X, Y))

        H_full = H_full / X.shape[0]

        H_full = 0.5 * (H_full + H_full.T)  # Ensure symmetry
        return H_full + damping * jnp.eye(H_full.shape[0])

    # ------------------------------------------------------------------
    # Lazy HVP — dispatched from `estimate_hvp` when `.build()` was skipped.
    # ------------------------------------------------------------------

    def _lazy_hvp(
        self,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """JVP-through-the-model HVP — no matrix materialized.  Expects 2D input."""
        return self._compute_hvp(self.compute_context, vectors, damping)

    @staticmethod
    @jax.jit
    def _compute_hvp(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Memory-efficient Hessian-vector product computation.
        Uses scan over samples and vmap over vectors, matching GNH strategy.
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        assert Y is not None, (
            "Targets must be provided in ModelContext for HVP computation."
        )

        def model_out(p, x):
            params = compute_context.unravel_fn(p)
            return compute_context.model_apply_fn(params, x[None, ...]).squeeze(0)

        def loss_single(p, x, y):
            """Loss for a single sample"""
            z = model_out(p, x)
            return compute_context.loss_fn(z, y)

        # Per-vector HVP function (scans over samples)
        @jax.jit
        def hvp_single(v):
            """Compute H @ v by accumulating over samples"""

            def body_fn(accum, xy):
                x_i, y_i = xy

                # Compute per-sample Hessian-vector product
                # hvp = ∇²L_i @ v for sample i
                def grad_fn(p):
                    return jax.grad(lambda p_: loss_single(p_, x_i, y_i))(p)

                # Use JVP to compute Hessian-vector product efficiently
                _, hvp_i = jax.jvp(grad_fn, (p_flat,), (v,))

                return accum + hvp_i, None

            # Accumulate HVP contributions across all samples
            summed, _ = jax.lax.scan(body_fn, jnp.zeros_like(v), (X, Y))

            # Average and add damping
            hvp = summed / X.shape[0]
            return hvp + damping * v

        # ------------------------------------------------------------
        # Chunking vectors to avoid OOM
        # ------------------------------------------------------------
        CHUNK_SIZE = 32
        n_vectors = vectors.shape[0]

        if n_vectors <= CHUNK_SIZE:
            return jax.vmap(hvp_single)(vectors)
        else:
            outs = []
            for i in range(0, n_vectors, CHUNK_SIZE):
                chunk = vectors[i : i + CHUNK_SIZE]
                outs.append(jax.vmap(hvp_single)(chunk))
            return jnp.concatenate(outs, axis=0)

    # ------------------------------------------------------------------
    # Persistence helpers (unchanged)
    # ------------------------------------------------------------------

    def save_hessian(
        self, hessian: Optional[Float[Array, "n_params n_params"]], path: str
    ) -> None:
        """Save the Hessian matrix to a file."""
        if hessian is None:
            hessian = self._compute_hessian(self.compute_context, damping=0.0)

        assert isinstance(hessian, jnp.ndarray), "Hessian must be a JAX array."
        jnp.save(path, hessian)

    def load_hessian(self, path: str) -> Float[Array, "n_params n_params"]:
        """Load the Hessian matrix from a file, if the file exists. Otherwise compute and save it."""
        try:
            hessian = jnp.load(path)
            assert isinstance(hessian, jnp.ndarray), (
                "Loaded Hessian must be a JAX array."
            )
            return hessian
        except FileNotFoundError:
            hessian = self._compute_hessian(self.compute_context, damping=0.0)
            self.save_hessian(hessian, path)
            return hessian
