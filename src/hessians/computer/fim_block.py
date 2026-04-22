from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import DenseBlock, LayerMatrix
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class FIMBlockComputer(HessianEstimator):
    compute_context: DataActivationsGradients
    """
    Block-diagonal Fisher Information Matrix approximation.

    F = block_diag(F_1, ..., F_L) where each F_l uses the unified formula

        F_l = (Σ_n Σ_k p[n,k] · v_{n,k} v_{n,k}^T) / Σ p

    with v_{n,k} = a[n] ⊗ g[n,:,k] the layer's per-(sample,draw) parameter
    gradient vector.
    """

    def get_layer_names(self) -> List[str]:
        return self.compute_context.layer_names

    def _layer_shapes_from_context(
        self, compute_context: DataActivationsGradients
    ) -> Dict[str, Tuple[int, int]]:
        return {
            l: (
                int(compute_context.activations[l].shape[-1]),
                int(compute_context.gradients[l].shape[-2]),
            )
            for l in compute_context.layer_names
        }

    def _build(self) -> LayerMatrix:
        """Build a block-diagonal `LayerMatrix` of per-layer dense FIM blocks."""
        ctx = self.compute_context
        layer_names = list(ctx.layer_names)
        shapes = self._layer_shapes_from_context(ctx)

        diag_blocks: Dict[str, DenseBlock] = {}
        for layer in layer_names:
            act = ctx.activations[layer]
            grad = ctx.gradients[layer]
            block = self._compute_layer_block(act, grad, ctx.probs)
            diag_blocks[layer] = DenseBlock(
                matrix=block,
                row_shape=shapes[layer],
                col_shape=shapes[layer],
            )
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks,
            param_groups=layer_names,
            layer_shapes=shapes,
        )

    @staticmethod
    @jax.jit
    def _compute_layer_block(
        act: Float[Array, "N I"],
        grad: Float[Array, "N O k"],
        probs: Float[Array, "N k"],
    ) -> Float[Array, "n n"]:
        """
        F_l = (Σ_n Σ_k p[n,k] v_{n,k} v_{n,k}^T) / Σ p
        """
        # (N, I, O, k) -> (N, I*O, k)
        N = act.shape[0]
        vecs = jnp.einsum("ni,nok->niok", act, grad).reshape(N, -1, grad.shape[-1])
        block = jnp.einsum("npk,nqk,nk->pq", vecs, vecs, probs) / probs.sum()
        return 0.5 * (block + block.T)
