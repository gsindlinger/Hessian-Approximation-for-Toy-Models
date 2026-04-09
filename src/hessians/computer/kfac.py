from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class KFACComputer(EKFACComputer):
    """
    Kronecker-Factored Approximate Curvature (KFAC) Hessian approximation.

    Reuses EKFAC's build pipeline and `LayerMatrix`-based estimate methods.
    The only difference is how the per-layer `Lambda` is computed: KFAC uses
    the outer product of the activation and gradient eigenvalues, whereas
    EKFAC uses the eigenvalue correction.
    """

    def _compute_lambdas(
        self,
        compute_context: DataActivationsGradients,
        activation_eigvecs: Dict[str, Float[Array, "I I"]],
        gradient_eigvecs: Dict[str, Float[Array, "O O"]],
        activation_eigvals: Dict[str, Float[Array, "I"]],
        gradient_eigvals: Dict[str, Float[Array, "O"]],
    ) -> Dict[str, Float[Array, "I O"]]:
        """KFAC Lambda is `outer(λ_A, λ_G)` — no correction pass."""
        return {
            layer: jnp.outer(activation_eigvals[layer], gradient_eigvals[layer])
            for layer in compute_context.layer_names
        }
