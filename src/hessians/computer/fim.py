from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator, _accumulate_chunks
from src.hessians.layer_matrix import LayerMatrix
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class FIMComputer(HessianEstimator):
    compute_context: DataActivationsGradients
    batch_size: Optional[int] = field(default=None)
    """
    Fisher Information Matrix approximation.

    FIM = E[∇log p(y|x) ∇log p(y|x)^T]

    Unified weighted sum across pseudo-target draws:

        FIM = (Σ_n Σ_k p[n,k] · g_{n,k} g_{n,k}^T) / Σ_n Σ_k p[n,k]

    which recovers the three supported regimes:
    - EMPIRICAL_FISHER (k=1, p=ones):    (1/N) Σ_n g_n g_n^T
    - MCMC            (p=ones):          (1/(Nk)) Σ_n Σ_k g_{n,k} g_{n,k}^T
    - ALL_CLASSES     (p=softmax logits): (1/N) Σ_n Σ_k p(k|n) g_{n,k} g_{n,k}^T
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
        """Stream per-chunk FIM contributions and assemble the LayerMatrix.

        Each chunk reconstructs per-sample weight gradients from the
        collector's per-layer `(a, g)` and accumulates the outer-product
        sum — so we never materialize the full `(N, n_params, k)` tensor.
        """
        ctx = self.compute_context
        layer_names = list(ctx.layer_names)
        layer_shapes = self._layer_shapes_from_context(ctx)
        N = ctx.probs.shape[0]

        def _chunk(sl: slice):
            return self._fim_chunk_sum(
                {l: ctx.activations[l][sl] for l in layer_names},
                {l: ctx.gradients[l][sl] for l in layer_names},
                ctx.probs[sl],
            )

        fim_sum = _accumulate_chunks(N, self.batch_size, _chunk)
        fim = fim_sum / ctx.probs.sum()
        fim = 0.5 * (fim + fim.T)

        return LayerMatrix.from_dense(
            fim,
            param_groups=layer_names,
            layer_shapes=layer_shapes,
        )

    @staticmethod
    @jax.jit
    def _fim_chunk_sum(
        activations_dict: Dict[str, Float[Array, "n I"]],
        gradients_dict: Dict[str, Float[Array, "n O k"]],
        probs: Float[Array, "n k"],
    ) -> Float[Array, "n_params n_params"]:
        """Unnormalized FIM contribution for one chunk:
            Σ_{n∈chunk} Σ_k p[n,k] · g_{n,k} g_{n,k}^T
        where g_{n,k} = concat_l( vec(a_{n,l} ⊗ (∂L/∂z_{n,l,k})) )."""
        grads_per_layer = []
        for layer in activations_dict.keys():
            a = activations_dict[layer]  # (n, I_l)
            g = gradients_dict[layer]  # (n, O_l, k)
            G_l = jnp.einsum("ni,nok->niok", a, g).reshape(a.shape[0], -1, g.shape[-1])
            grads_per_layer.append(G_l)
        grads_chunk = jnp.concatenate(grads_per_layer, axis=1)  # (n, n_params, k)
        return jnp.einsum("npk,nqk,nk->pq", grads_chunk, grads_chunk, probs)
