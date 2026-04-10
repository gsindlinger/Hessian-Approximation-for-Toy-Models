from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import KroneckerEstimator


@dataclass
class EKFACComputer(KroneckerEstimator):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected
    KFAC (EKFAC) Hessian approximation.

    Covariances are the plain (unweighted) activation/gradient outer-product
    averages:

        A[layer] = (Σ_n a_n a_nᵀ) / N
        G[layer] = (Σ_{n,k} p[n,k] g_{n,k} g_{n,k}ᵀ) / Σp

    With `apply_eigenvalue_correction=False` this collapses to plain KFAC
    (`Λ = outer(λ_A, λ_G)`).  All scaffolding (eigendecomposition, Λ
    correction, block assembly, damping) lives on `KroneckerEstimator`.
    """

    @staticmethod
    @jax.jit
    def _cov_chunk_sums(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O k"]],
        probs: Float[Array, "N k"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Unnormalized per-chunk EKFAC sums:
        act_sum[l]  = Σ_n a_n a_nᵀ
        grad_sum[l] = Σ_{n,k} p[n,k] g_{n,k} g_{n,k}ᵀ
        """
        act_sums: Dict[str, Array] = {}
        grad_sums: Dict[str, Array] = {}
        for layer in activations_dict.keys():
            a = activations_dict[layer]  # (N, I)
            g = gradients_dict[layer]  # (N, O, k)
            act_sums[layer] = jnp.einsum("ni,nj->ij", a, a)
            grad_sums[layer] = jnp.einsum("nok,npk,nk->op", g, g, probs)
        return {"act": act_sums, "grad": grad_sums}

    @staticmethod
    def _finalize_covariances(
        act_sums: Dict[str, Float[Array, "I I"]],
        grad_sums: Dict[str, Float[Array, "O O"]],
        N: int,
        total_prob: Float,
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """EKFAC normalization: A = act_sum / N, G = grad_sum / Σp."""
        return {
            "activation_cov": {layer: act_sums[layer] / N for layer in act_sums},
            "gradient_cov": {
                layer: grad_sums[layer] / total_prob for layer in grad_sums
            },
        }
