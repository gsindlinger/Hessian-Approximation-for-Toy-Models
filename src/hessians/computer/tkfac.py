from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.etkfac import ETKFACComputer
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class TKFACComputer(ETKFACComputer):
    """
    Trace-normalized KFAC (TKFAC) Hessian approximation.

    Uses ETKFAC's trace-normalized covariance computation, but KFAC's simple
    outer-product Lambda (no eigenvalue correction pass).
    """

    def _compute_lambdas(
        self,
        compute_context: DataActivationsGradients,
        activation_eigvecs: Dict[str, Float[Array, "I I"]],
        gradient_eigvecs: Dict[str, Float[Array, "O O"]],
        activation_eigvals: Dict[str, Float[Array, "I"]],
        gradient_eigvals: Dict[str, Float[Array, "O"]],
    ) -> Dict[str, Float[Array, "I O"]]:
        """KFAC-style Lambda: `outer(λ_A, λ_G)`, no correction pass."""
        return {
            layer: jnp.outer(activation_eigvals[layer], gradient_eigvals[layer])
            for layer in compute_context.layer_names
        }
