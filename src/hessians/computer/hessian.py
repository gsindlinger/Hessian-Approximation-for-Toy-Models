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

    `.build()` materializes the full `(n_params, n_params)` Hessian and
    slices it into per-layer `DenseBlock`s via `LayerMatrix.from_dense`;
    all `estimate_*` methods use the cached matrix (build is required).
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
