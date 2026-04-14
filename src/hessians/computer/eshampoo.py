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

        L = E[||a||^2  g g^T],   R = E[||g||^2  a a^T]

    where G_{x,s} = g a^T is the rank-1 per-sample weight gradient, and
    L = E[G G^T], R = E[G^T G] reduces to the cross-norm-weighted forms
    above.  See ``_compute_covariances_batch_averaged`` for the legacy
    batch-averaged version.
    """

    @staticmethod
    @jax.jit
    def _compute_covariances(
        activation_batch_dict: Dict[str, Float[Array, "N I"]],
        gradient_batch_dict: Dict[str, Float[Array, "k N O"]],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Per-sample Shampoo covariances.

        For each sample n and pseudo-target k, the per-sample weight gradient
        is rank-1: Delta_{k,n} = g_{k,n} a_n^T.  The Shampoo factors are:

            R_{ij} = (1/k) sum_{k,n} ||g_{k,n}||^2  a_{ni} a_{nj}
            L_{op} = (1/k) sum_{k,n} ||a_n||^2       g_{kno} g_{knp}

        where R maps to ``activation_cov`` (I x I) and L to ``gradient_cov``
        (O x O).
        """
        activation_cov_dict = {}
        gradient_cov_dict = {}
        for layer in activation_batch_dict.keys():
            activations = activation_batch_dict[layer]  # (N, I)
            gradients = gradient_batch_dict[layer]  # (k, N, O)
            k = gradients.shape[0]

            g_norms_sq = jnp.einsum("kno,kno->kn", gradients, gradients)  # (k, N)
            a_norms_sq = jnp.einsum("ni,ni->n", activations, activations)  # (N,)

            # R = (1/k) sum_{k,n} ||g_{k,n}||^2  a_n a_n^T
            a_weights = jnp.sqrt(g_norms_sq.sum(axis=0) / k)  # (N,)
            weighted_a = activations * a_weights[:, None]  # (N, I)
            activation_cov = jnp.einsum("ni,nj->ij", weighted_a, weighted_a)

            # L = (1/k) sum_{k,n} ||a_n||^2  g_{k,n} g_{k,n}^T
            g_weights = jnp.sqrt(a_norms_sq / k)  # (N,)
            weighted_g = gradients * g_weights[None, :, None]  # (k, N, O)
            gradient_cov = jnp.einsum("kno,knp->op", weighted_g, weighted_g)

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }

    @staticmethod
    @jax.jit
    def _compute_covariances_weighted(
        activation_batch_dict: Dict[str, Float[Array, "N I"]],
        gradient_batch_dict: Dict[str, Float[Array, "K N O"]],
        probabilities: Float[Array, "N K"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Probability-weighted per-sample Shampoo covariances.

        Combines the ALL_CLASSES probability weighting with Shampoo's
        cross-norm reweighting:

            R_{ij} = sum_{k,n} p_{nk} ||g_{k,n}||^2  a_{ni} a_{nj}
            L_{op} = sum_{k,n} p_{nk} ||a_n||^2       g_{kno} g_{knp}
        """
        activation_cov_dict = {}
        gradient_cov_dict = {}
        for layer in activation_batch_dict.keys():
            activations = activation_batch_dict[layer]  # (N, I)
            gradients = gradient_batch_dict[layer]  # (K, N, O)

            g_norms_sq = jnp.einsum("kno,kno->kn", gradients, gradients)  # (K, N)
            a_norms_sq = jnp.einsum("ni,ni->n", activations, activations)  # (N,)

            # R = sum_{k,n} p_{nk} ||g_{k,n}||^2  a_n a_n^T
            # probabilities is (N, K), g_norms_sq is (K, N) -> weight per sample
            a_weights = jnp.sqrt(
                (probabilities.T * g_norms_sq).sum(axis=0)
            )  # (N,)
            weighted_a = activations * a_weights[:, None]  # (N, I)
            activation_cov = jnp.einsum("ni,nj->ij", weighted_a, weighted_a)

            # L = sum_{k,n} p_{nk} ||a_n||^2  g_{kno} g_{knp}
            sqrt_prob = jnp.sqrt(probabilities.T)  # (K, N)
            sqrt_anorm = jnp.sqrt(a_norms_sq)  # (N,)
            weighted_g = gradients * (sqrt_prob * sqrt_anorm[None, :])[:, :, None]  # (K, N, O)
            gradient_cov = jnp.einsum("kno,knp->op", weighted_g, weighted_g)

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }

    # ------------------------------------------------------------------
    # Legacy batch-averaged version (kept for comparison).
    # To use: swap  _compute_covariances = _compute_covariances_batch_averaged
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _compute_covariances_batch_averaged(
        activation_batch_dict: Dict[str, Float[Array, "N I"]],
        gradient_batch_dict: Dict[str, Float[Array, "k N O"]],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Legacy batch-averaged Shampoo covariances.

        Forms the accumulated gradient Delta = (1/k) sum_k sum_n g_{k,n} a_n^T
        and computes A = Delta^T Delta, G = Delta Delta^T.  This corresponds
        to "practical Shampoo" (batch-level gradient outer product) rather than
        the per-sample Hessian-approximation form.
        """
        activation_cov_dict = {}
        gradient_cov_dict = {}
        for layer in activation_batch_dict.keys():
            activations = activation_batch_dict[layer]  # (N, I)
            gradients = gradient_batch_dict[layer]  # (k, N, O)
            k = gradients.shape[0]

            grad = jnp.einsum("kno,ni->oi", gradients, activations) / k  # (O, I)
            activation_cov = jnp.einsum("oi,oj->ij", grad, grad)  # (I, I)
            gradient_cov = jnp.einsum("oi,pi->op", grad, grad)  # (O, O)

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }