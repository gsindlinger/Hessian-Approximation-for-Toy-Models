from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator, _accumulate_chunks
from src.hessians.layer_matrix import DenseBlock, LayerMatrix
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class FIMBlockComputer(HessianEstimator):
    compute_context: DataActivationsGradients
    batch_size: Optional[int] = field(default=None)
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
        """Stream per-chunk block-diagonal FIM contributions across all layers,
        so we never materialize the full `(N, I·O, k)` per-layer tensor."""
        ctx = self.compute_context
        layer_names = list(ctx.layer_names)
        shapes = self._layer_shapes_from_context(ctx)
        N = ctx.probs.shape[0]

        def _chunk(sl: slice):
            return self._block_chunk_sums(
                {l: ctx.activations[l][sl] for l in layer_names},
                {l: ctx.gradients[l][sl] for l in layer_names},
                ctx.probs[sl],
            )

        block_sums = _accumulate_chunks(N, self.batch_size, _chunk)
        total_prob = ctx.probs.sum()

        diag_blocks: Dict[str, DenseBlock] = {}
        for layer in layer_names:
            block = block_sums[layer] / total_prob
            block = 0.5 * (block + block.T)
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
    def _block_chunk_sums(
        activations_dict: Dict[str, Float[Array, "n I"]],
        gradients_dict: Dict[str, Float[Array, "n O k"]],
        probs: Float[Array, "n k"],
    ) -> Dict[str, Float[Array, "n_l n_l"]]:
        """Unnormalized per-layer chunk contribution:
            Σ_{n∈chunk} Σ_k p[n,k] v_{n,k,l} v_{n,k,l}^T
        with v_{n,k,l} = vec(a_{n,l} ⊗ g_{n,l,k}).
        """
        out: Dict[str, Array] = {}
        for layer in activations_dict.keys():
            a = activations_dict[layer]
            g = gradients_dict[layer]
            vecs = jnp.einsum("ni,nok->niok", a, g).reshape(a.shape[0], -1, g.shape[-1])
            out[layer] = jnp.einsum("npk,nqk,nk->pq", vecs, vecs, probs)
        return out
