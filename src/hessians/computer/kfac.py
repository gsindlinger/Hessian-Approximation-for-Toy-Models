from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.utils.data import EKFACData
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class KFACComputer(HessianEstimator):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
    """

    compute_context: EKFACData

    def estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian approximation.
        """
        return self.compute_hessian_or_inverse_hessian_estimate(
            method="normal",
            damping=0.0 if damping is None else damping,
        )

    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        return self.compute_ihvp_or_hvp(
            vectors,
            Lambdas=self._compute_lambdas(),
            method="hvp",
            damping=0.0 if damping is None else damping,
        )

    def compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> float:
        """
        Compare the (E)KFAC Hessian approximation to a given comparison matrix
        """
        return self._compare_hessian_estimates(
            activations_eigenvectors=list(
                self.compute_context.activation_eigenvectors.values()
            ),
            gradients_eigenvectors=list(
                self.compute_context.gradient_eigenvectors.values()
            ),
            Lambdas=list(self._compute_lambdas().values()),
            damping=0.0 if damping is None else damping,
            comparison_matrix=comparison_matrix,
            metric=metric.compute_fn(),
            method="normal",
        )

    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product.
        """
        return self.compute_ihvp_or_hvp(
            vectors=vectors,
            Lambdas=self._compute_lambdas(),
            method="ihvp",
            damping=0.0 if damping is None else damping,
        )

    def estimate_inverse_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full inverse Hessian.
        """
        return self.compute_hessian_or_inverse_hessian_estimate(
            method="inverse",
            damping=0.0 if damping is None else damping,
        )

    def compute_hessian_or_inverse_hessian_estimate(
        self, method: Literal["normal", "inverse"], damping: Float
    ) -> Float[Array, "n_params n_params"]:
        """
        Unified helper method to compute either the full Hessian or its inverse.
        """
        return self._compute_hessian_or_inverse_hessian_estimate(
            eigenvectors_activations=list(
                self.compute_context.activation_eigenvectors.values()
            ),
            eigenvectors_gradients=list(
                self.compute_context.gradient_eigenvectors.values()
            ),
            Lambdas=list(self._compute_lambdas().values()),
            damping=damping,
            method=method,
        )

    def _compute_lambdas(
        self,
    ) -> Dict[str, Float[Array, "I O"]]:
        """Compute eigenvalue lambda for KFAC using the following formula:
        Λ = (Λ_G ⊗ Λ_A) = Λ_A @ Λ_G^T
        where Λ_G and Λ_A are the eigenvalues of the gradient and activation covariances.
        """
        lambdas = {}
        activation_eigenvalues = self.compute_context.activation_eigenvalues
        gradient_eigenvalues = self.compute_context.gradient_eigenvalues
        for layer_name in activation_eigenvalues.keys():
            A_eigvals: Float[Array, "I"] = activation_eigenvalues[layer_name]
            G_eigvals: Float[Array, "O"] = gradient_eigenvalues[layer_name]
            lambdas[layer_name] = jnp.outer(A_eigvals, G_eigvals)
        return lambdas
