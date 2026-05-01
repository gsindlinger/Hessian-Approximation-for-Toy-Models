from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import KroneckerEstimator


@dataclass
class EMacComputer(KroneckerEstimator):
    """
    Mean Activation Covariance (MAC) and Eigenvalue-Corrected MAC (EMAC)
    Hessian approximation.

    Same scaffolding as (E)KFAC, but the gradient covariance is fixed to
    the identity:

        A[layer] = (Σ_n a_n a_nᵀ) / N
        G[layer] = I_O

    With `apply_eigenvalue_correction=False` (plain MAC), the assembled
    block is `A ⊗ I_O`.  With the eigenvalue correction enabled (EMAC),
    Q_G = I and λ_G = 1, so `Λ[i, o]` collapses to
    `(Σ p_{n,k} · (Q_Aᵀa)_{n,i}² · g_{n,o,k}²) / Σp` — activation
    directions are still rotated, but per-output gradient magnitudes are
    used as-is rather than whitened by G.
    """

    @staticmethod
    @jax.jit
    def _cov_chunk_sums(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O k"]],
        probs: Float[Array, "N k"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Per-chunk sums:
            act_sum[l]  = Σ_n a_n a_nᵀ
            grad_sum[l] = 0  (placeholder; gradient cov is fixed to I in
                              `_finalize_covariances`).  We carry the (O, O)
                              shape so the finalizer knows the layer width.
        """
        act_sums: Dict[str, Array] = {}
        grad_sums: Dict[str, Array] = {}
        for layer in activations_dict.keys():
            a = activations_dict[layer]  # (N, I)
            g = gradients_dict[layer]  # (N, O, k)
            O = g.shape[1]
            act_sums[layer] = jnp.einsum("ni,nj->ij", a, a)
            grad_sums[layer] = jnp.zeros((O, O), dtype=a.dtype)
        return {"act": act_sums, "grad": grad_sums}

    @staticmethod
    def _finalize_covariances(
        act_sums: Dict[str, Float[Array, "I I"]],
        grad_sums: Dict[str, Float[Array, "O O"]],
        N: int,
        total_prob: Float,
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Normalize: A = act_sum / N, G = I_O (gradient cov ignored)."""
        return {
            "activation_cov": {layer: act_sums[layer] / N for layer in act_sums},
            "gradient_cov": {
                layer: jnp.eye(grad_sums[layer].shape[0], dtype=grad_sums[layer].dtype)
                for layer in grad_sums
            },
        }
