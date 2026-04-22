from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import KroneckerEstimator


@dataclass
class EShampooComputer(KroneckerEstimator):
    """
    Shampoo-style Eigenvalue-Corrected KFAC (EShampoo) Hessian approximation.

    Uses per-sample weight-gradient outer products to form the left and right
    Shampoo preconditioners (Hessian-approximation viewpoint, Gupta et al.;
    identical to paper-TKFAC, Gao et al. 2020):

        L_{ij} = (Σ_{n,k} p[n,k] ||g_{n,k}||^2 a_{ni} a_{nj}) / Σp
        R_{op} = (Σ_{n,k} p[n,k] ||a_n||^2    g_{n,o,k} g_{n,p,k}) / Σp
        F     = (L ⊗ R) / tr(L)           (trace-restricted normalization)

    where L maps to ``activation_cov`` (I x I) and R to ``gradient_cov``
    (O x O).  The ALL_CLASSES probability weights and EF/MCMC's uniform
    weights both drop out of the same formula via ``probs``.

    With `apply_eigenvalue_correction=False` this is plain Shampoo (no Λ
    correction).  All scaffolding lives on `KroneckerEstimator`.
    """

    @staticmethod
    @jax.jit
    def _cov_chunk_sums(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O k"]],
        probs: Float[Array, "N k"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Unnormalized Shampoo-weighted chunk sums:

            act_sum[l]  = Σ_n (Σ_k p[n,k] ||g_{n,k}||²) · a_n a_n^T
            grad_sum[l] = Σ_{n,k} p[n,k] ||a_n||² · g_{n,k} g_{n,k}^T
        """
        act_sums: Dict[str, Array] = {}
        grad_sums: Dict[str, Array] = {}
        for layer in activations_dict.keys():
            a = activations_dict[layer]  # (N, I)
            g = gradients_dict[layer]  # (N, O, k)
            g_norms_sq = jnp.einsum("nok,nok->nk", g, g)  # (N, k)
            a_norms_sq = jnp.einsum("ni,ni->n", a, a)  # (N,)

            w_act = jnp.einsum("nk,nk->n", probs, g_norms_sq)  # (N,)
            act_sums[layer] = jnp.einsum("n,ni,nj->ij", w_act, a, a)

            w_grad = probs * a_norms_sq[:, None]  # (N, k)
            grad_sums[layer] = jnp.einsum("nk,nok,npk->op", w_grad, g, g)
        return {"act": act_sums, "grad": grad_sums}

    @staticmethod
    def _finalize_covariances(
        act_sums: Dict[str, Float[Array, "I I"]],
        grad_sums: Dict[str, Float[Array, "O O"]],
        N: int,
        total_prob: Float,
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Shampoo paper Corollary / TKFAC paper Eq. 11:
            F = (L ⊗ R) / tr(L),   both L and R averaged over total_prob.
        tr(L) = tr(R) here, so dividing activation_cov by its trace is
        equivalent to dividing the Kronecker product by tr(L).
        """
        L = {layer: act_sums[layer] / total_prob for layer in act_sums}
        R = {layer: grad_sums[layer] / total_prob for layer in grad_sums}
        return {
            "activation_cov": {layer: L[layer] / jnp.trace(L[layer]) for layer in L},
            "gradient_cov": R,
        }
