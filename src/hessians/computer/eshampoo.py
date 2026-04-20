from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.ekfac import EKFACComputer


@dataclass
class EShampooComputer(EKFACComputer):
    """
    Shampoo-style Eigenvalue-Corrected KFAC (EShampoo) Hessian approximation.

    Uses per-sample weight-gradient outer products to form the left and right
    Shampoo preconditioners (Hessian-approximation viewpoint, Gupta et al.):

        R_{ij} = (Σ_{n,k} p[n,k] ||g_{n,k}||^2 a_{ni} a_{nj}) / Σp
        L_{op} = (Σ_{n,k} p[n,k] ||a_n||^2    g_{n,o,k} g_{n,p,k}) / Σp

    where R maps to ``activation_cov`` (I x I) and L to ``gradient_cov``
    (O x O).  The ALL_CLASSES probability weights and EF/MCMC's uniform
    weights both drop out of the same formula via ``probs``.
    """

    @staticmethod
    @jax.jit
    def _compute_covariances(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O k"]],
        probs: Float[Array, "N k"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        total_prob = probs.sum()
        activation_cov_dict = {}
        gradient_cov_dict = {}
        for layer in activations_dict.keys():
            a = activations_dict[layer]  # (N, I)
            g = gradients_dict[layer]  # (N, O, k)

            g_norms_sq = jnp.einsum("nok,nok->nk", g, g)  # (N, k)
            a_norms_sq = jnp.einsum("ni,ni->n", a, a)  # (N,)

            # R = (Σ_{n,k} p[n,k] ||g_{n,k}||^2) a_n a_n^T / Σp
            a_weights = jnp.sqrt(jnp.einsum("nk,nk->n", probs, g_norms_sq))  # (N,)
            weighted_a = a * a_weights[:, None]
            activation_cov = (
                jnp.einsum("ni,nj->ij", weighted_a, weighted_a) / total_prob
            )

            # L = Σ_{n,k} (p[n,k] ||a_n||^2) g_{n,k} g_{n,k}^T / Σp
            sqrt_w = jnp.sqrt(probs * a_norms_sq[:, None])  # (N, k)
            weighted_g = g * sqrt_w[:, None, :]  # (N, O, k)
            gradient_cov = (
                jnp.einsum("nok,npk->op", weighted_g, weighted_g) / total_prob
            )

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }
