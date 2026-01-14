from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.ekfac import EKFACComputer


@dataclass
class EShampooComputer(EKFACComputer):
    """
    Shampoo-style Eigenvalue-Corrected KFAC (EShampoo) Hessian approximation.
    Same as EKFAC but with different covariance computation.
    """

    @staticmethod
    def _compute_covariances(
        activation_batch_dict: Dict[str, Float[Array, "N I"]],
        gradient_batch_dict: Dict[str, Float[Array, "N O"]],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Compute covariance matrices - Shampoo version."""
        activation_cov_dict = {}
        gradient_cov_dict = {}
        for layer in activation_batch_dict.keys():
            activation_batch = activation_batch_dict[layer]
            gradient_batch = gradient_batch_dict[layer]

            grad = jnp.einsum("no, ni -> oi", gradient_batch, activation_batch)
            # TODO: Double check transpose
            activation_cov = jnp.einsum("oi, oj->ij", grad, grad)
            gradient_cov = jnp.einsum("oi,pi->op", grad, grad)

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }
