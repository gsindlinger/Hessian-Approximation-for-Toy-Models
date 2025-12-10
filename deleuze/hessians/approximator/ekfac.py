import logging
from dataclasses import dataclass, field
from typing import Callable, Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from deleuze.hessians.utils.data import EKFACData
from deleuze.models.approximation_model import ApproximationModel
from deleuze.utils.data.data import Dataset

from ..collector import (
    CollectorEKFACEigenvalueCorrections,
    CollectorKFACCovariances,
)
from .approximator import ApproximatorBase
from .kfac import compute_eigenvectors_and_eigenvalues

logger = logging.getLogger(__name__)


@dataclass
class EKFACApproximator(ApproximatorBase):
    """
    Builder for E-KFAC Hessian approximation.
    """

    data: EKFACData = field(default_factory=EKFACData)

    def _build(
        self,
        model: ApproximationModel,
        params: Dict,
        dataset: Dataset,
        loss_fn: Callable,
    ):
        """
        Build the E-KFAC Hessian approximation.
        """

        # Collect covariances by forward and backward passes
        (activations_covs, gradients_covs) = CollectorKFACCovariances(
            model, params
        ).collect(dataset.inputs, dataset.targets, loss_fn)

        logger.info("Finished collecting covariances for EK-FAC approximation.")

        # Compute eigenvectors of the covariances (discard eigenvalues since not needed in EK-FAC)
        (self.data.activation_eigenvectors, self.data.gradient_eigenvectors), _ = (
            compute_eigenvectors_and_eigenvalues(activations_covs, gradients_covs)
        )

        logger.info("Computed eigenvectors for EK-FAC approximation.")

        # Run a second forward and backward pass to collect / compute eigenvalue corrections
        self.data.eigenvalue_corrections = CollectorEKFACEigenvalueCorrections(
            model=model,
            params=params,
            eigenvectors_activations=self.data.activation_eigenvectors,
            eigenvectors_gradients=self.data.gradient_eigenvectors,
        ).collect(dataset.inputs, dataset.targets, loss_fn)

        logger.info("Computed eigenvalue corrections for EK-FAC approximation.")

        # Compute mean eigenvalues and eigenvalue corrections for damping
        self.data.mean_corrections = {}
        self.data.mean_corrections_aggregated = 0.0

        for layer_name in self.data.eigenvalue_corrections.keys():
            mean_correction: Float[Array, ""] = jnp.mean(
                self.data.eigenvalue_corrections[layer_name]
            )
            self.data.mean_corrections[layer_name] = mean_correction
            self.data.mean_corrections_aggregated += mean_correction

        # Divide overall sums by number of layers
        n_layers = len(self.data.eigenvalue_corrections)
        self.data.mean_corrections_aggregated = jnp.array(
            self.data.mean_corrections_aggregated / n_layers if n_layers > 0 else 0.0
        )

        logger.info("Computed mean eigenvalue corrections for EK-FAC approximation.")
