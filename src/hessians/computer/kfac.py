from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.ekfac import EKFACComputer


@dataclass
class KFACComputer(EKFACComputer):
    """
    Kronecker-Factored Approximate Curvature (KFAC) Hessian approximation.

    Reuses EKFAC's build pipeline and `LayerMatrix`-based estimate methods.
    The only difference is how the per-layer Lambda is computed: KFAC uses
    the outer product of the activation and gradient eigenvalues instead of
    the full eigenvalue correction.
    """

    def _get_lambdas(self) -> Dict[str, Float[Array, "I O"]]:
        """Compute KFAC eigenvalue lambda as `outer(λ_A, λ_G)` per layer."""
        assert self.precomputed_data is not None, (
            "EKFAC data must be built before computing Lambdas."
        )
        activation_eigenvalues = self.precomputed_data.activation_eigenvalues
        gradient_eigenvalues = self.precomputed_data.gradient_eigenvalues
        return {
            layer: jnp.outer(activation_eigenvalues[layer], gradient_eigenvalues[layer])
            for layer in self.compute_context.layer_names
        }
