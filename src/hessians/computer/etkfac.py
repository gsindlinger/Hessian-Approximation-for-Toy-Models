from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.ekfac import EKFACComputer


@dataclass
class ETKFACComputer(EKFACComputer):
    """
    Trace-normalized Eigenvalue-Corrected KFAC (ETKFAC) Hessian approximation.
    Same as EKFAC but with different covariance computation.
    """

    @staticmethod
    def _compute_covariances(
        activation_batch_dict: Dict[str, Float[Array, "N I"]],
        gradient_batch_dict: Dict[str, Float[Array, "N O"]],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Compute covariance matrices - TKFAC version."""
        activation_cov_dict = {}
        gradient_cov_dict = {}
        trace_dict = {}
        for layer in activation_batch_dict.keys():
            activation_batch = activation_batch_dict[layer]
            gradient_batch = gradient_batch_dict[layer]

            activation_cov = jnp.einsum("ni,nj->ij", activation_batch, activation_batch)
            gradient_cov = jnp.einsum("ni,nj->ij", gradient_batch, gradient_batch)

            activation_trace = jnp.trace(activation_cov)
            gradient_trace = jnp.trace(gradient_cov)

            activation_cov = activation_cov * gradient_trace
            gradient_cov = gradient_cov * activation_trace
            trace_entry = activation_trace * gradient_trace
            # TODO: need to multiply by trace at the end

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov
            trace_dict[layer] = trace_entry

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
            "trace": trace_dict,
        }
