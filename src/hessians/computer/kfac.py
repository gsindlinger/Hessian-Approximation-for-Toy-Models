from __future__ import annotations


from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Callable

import jax.numpy as jnp
from jaxtyping import Array, Float
from simple_parsing import field

from src.hessians.computer.computer import CollectorBasedHessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.utils.data import DataActivationsGradients, EKFACData
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class KFACComputer(CollectorBasedHessianEstimator):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
    """

    precomputed_data: EKFACData = field(default_factory=EKFACData)

    @staticmethod
    def _build(
        compute_context: Tuple[DataActivationsGradients, DataActivationsGradients],
    ) -> EKFACData:
        return EKFACComputer._build(compute_context)

    def _estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian approximation.
        """
        return self.compute_hessian_or_inverse_hessian_estimate(
            method="normal",
            damping=0.0 if damping is None else damping,
        )

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        assert self.precomputed_data is not None, (
            "EKFAC data must be built before computing HVP."
        )

        return EKFACComputer.compute_ihvp_or_hvp(
            data=self.precomputed_data,
            vectors=vectors,
            Lambdas=self._compute_lambdas(),
            layer_names=self.compute_context[0].layer_names,
            method="hvp",
            damping=0.0 if damping is None else damping,
        )

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> float:
        """
        Compare the (E)KFAC Hessian approximation to a given comparison matrix
        """
        Lambdas_unordered = self._compute_lambdas()
        return EKFACComputer._compare_hessian_estimates(
            activations_eigenvectors=[
                self.precomputed_data.activation_eigenvectors[layer]
                for layer in self.compute_context[0].layer_names
            ],
            gradients_eigenvectors=[
                self.precomputed_data.gradient_eigenvectors[layer]
                for layer in self.compute_context[0].layer_names
            ],
            Lambdas=[
                Lambdas_unordered[layer]
                for layer in self.compute_context[0].layer_names
            ],
            damping=0.0 if damping is None else damping,
            comparison_matrix=comparison_matrix,
            metric=metric.compute_fn(),
            method="normal",
        )

    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product.
        """
        assert self.precomputed_data is not None, (
            "EKFAC data must be built before computing IHVP."
        )
        return EKFACComputer.compute_ihvp_or_hvp(
            data=self.precomputed_data,
            vectors=vectors,
            Lambdas=self._compute_lambdas(),
            layer_names=self.compute_context[0].layer_names,
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
        assert self.precomputed_data is not None, (
            "EKFAC data must be built before computing Hessian or inverse Hessian."
        )
        Lambdas = self._compute_lambdas()
        return EKFACComputer._compute_hessian_or_inverse_hessian_estimate(
            eigenvectors_activations=[
                self.precomputed_data.activation_eigenvectors[layer]
                for layer in self.compute_context[0].layer_names
            ],
            eigenvectors_gradients=[
                self.precomputed_data.gradient_eigenvectors[layer]
                for layer in self.compute_context[0].layer_names
            ],
            Lambdas=[Lambdas[layer] for layer in self.compute_context[0].layer_names],
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
        assert self.precomputed_data is not None, (
            "EKFAC data must be built before computing Lambdas."
        )
        lambdas = {}
        activation_eigenvalues = self.precomputed_data.activation_eigenvalues
        gradient_eigenvalues = self.precomputed_data.gradient_eigenvalues
        for layer_name in activation_eigenvalues.keys():
            A_eigvals: Float[Array, "I"] = activation_eigenvalues[layer_name]
            G_eigvals: Float[Array, "O"] = gradient_eigenvalues[layer_name]
            lambdas[layer_name] = jnp.outer(A_eigvals, G_eigvals)
        return lambdas
