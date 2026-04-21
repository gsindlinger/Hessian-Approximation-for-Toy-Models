from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import LayerMatrix
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class FIMComputer(HessianEstimator):
    compute_context: DataActivationsGradients
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
        """Assemble per-sample parameter gradients, materialize the FIM, and slice."""
        ctx = self.compute_context
        layer_names = list(ctx.layer_names)
        layer_shapes = self._layer_shapes_from_context(ctx)

        grads_per_layer = []
        for layer in layer_names:
            a = ctx.activations[layer]  # (N, I_l)
            g = ctx.gradients[layer]  # (N, O_l, k)
            # (N, I_l, O_l, k) -> (N, I_l*O_l, k)
            G_l = jnp.einsum("ni,nok->niok", a, g).reshape(a.shape[0], -1, g.shape[-1])
            grads_per_layer.append(G_l)
        grads_all = jnp.concatenate(grads_per_layer, axis=1)  # (N, n_params, k)

        dense = self._compute_fim(grads_all, ctx.probs)

        return LayerMatrix.from_dense(
            dense,
            param_groups=layer_names,
            layer_shapes=layer_shapes,
        )

    @staticmethod
    @jax.jit
    def _compute_fim(
        gradients: Float[Array, "N n_params k"],
        probs: Float[Array, "N k"],
    ) -> Float[Array, "n_params n_params"]:
        """
        FIM = (Σ_n Σ_k p[n,k] g_{n,k} g_{n,k}^T) / Σ p  + damping · I
        """
        fim_sum = jnp.einsum("npk,nqk,nk->pq", gradients, gradients, probs)
        fim = fim_sum / probs.sum()

        return fim
