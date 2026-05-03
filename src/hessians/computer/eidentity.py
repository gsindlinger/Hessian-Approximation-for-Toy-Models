from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import KroneckerEstimator


@dataclass
class EIdentityComputer(KroneckerEstimator):
    """
    Identity.
    """

    @staticmethod
    @jax.jit
    def _cov_chunk_sums(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O k"]],
        probs: Float[Array, "N k"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Per-chunk sums:
        act_sum[l]  = 0  (placeholder; activation cov is fixed to I at the end)
        grad_sum[l] = 0  (placeholder; gradient cov is fixed to I in the end)
        """
        act_sums: Dict[str, Array] = {}
        grad_sums: Dict[str, Array] = {}
        for layer in activations_dict.keys():
            a = activations_dict[layer]  # (N, I)
            g = gradients_dict[layer]  # (N, O, k)
            O = g.shape[1]
            I = a.shape[1]
            act_sums[layer] = jnp.zeros((I, I), dtype=a.dtype)
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
            "activation_cov": {
                layer: jnp.eye(act_sums[layer].shape[0], dtype=grad_sums[layer].dtype)
                for layer in act_sums
            },
            "gradient_cov": {
                layer: jnp.eye(grad_sums[layer].shape[0], dtype=grad_sums[layer].dtype)
                for layer in grad_sums
            },
        }
