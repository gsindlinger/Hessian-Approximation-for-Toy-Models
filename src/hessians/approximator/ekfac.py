import logging
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.utils.data import EKFACData

from ..collector import (
    CollectorActivationsGradients,
)
from .approximator import ApproximatorBase

logger = logging.getLogger(__name__)


@dataclass
class EKFACApproximator(ApproximatorBase):
    """
    Computes the components required to compute the EK-FAC Hessian approximation,
    i.e., activation and gradient eigenvectors, eigenvalues & eigenvalue corrections.
    """

    collected_data_path_snd: str
    """Path to second set of previously collected data"""

    def _build(
        self,
    ):
        """Method to build the required components to compute"""
        # Load first activations and gradients from path
        if self.collected_data_path is None or self.collected_data_path_snd is None:
            raise ValueError(
                "Both collected_data_path and collected_data_path_snd must be provided to load EK-FAC data."
            )

        logger.info("Loading activations and gradients for EK-FAC approximation.")
        activations, gradients = CollectorActivationsGradients.load(
            directory=self.collected_data_path
        )

        # Compute covariances of activations and gradients
        logger.info("Computing covariances for EK-FAC approximation.")
        activations_covs, gradients_covs = self.compute_covariances(
            activations, gradients
        )

        # Compute eigenvectors & eigenvalues of the covariances (discard eigenvalues since not needed in EK-FAC)
        (
            (activation_eigenvectors, gradient_eigenvectors),
            (activation_eigenvalues, gradient_eigenvalues),
        ) = self.compute_eigenvectors_and_eigenvalues(activations_covs, gradients_covs)
        logger.info("Computed eigenvectors for EK-FAC approximation.")

        # Run data from second forward and backward pass to collect / compute eigenvalue corrections
        logger.info(
            "Loading second activations and gradients for EK-FAC approximation."
        )
        activations, gradients = CollectorActivationsGradients.load(
            directory=self.collected_data_path_snd
        )
        eigenvalue_corrections = self.compute_eigenvalue_corrections(
            activations,
            gradients,
            activation_eigenvectors,
            gradient_eigenvectors,
        )
        logger.info("Computed eigenvalue corrections for EK-FAC approximation.")

        # Compute mean eigenvalues and eigenvalue corrections for damping
        mean_corrections = {}
        mean_eigenvalues = {}
        mean_corrections_aggregated = 0.0
        mean_eigenvalues_aggregated = 0.0

        for layer_name in eigenvalue_corrections.keys():
            mean_correction: Float[Array, ""] = jnp.mean(
                eigenvalue_corrections[layer_name]
            )
            mean_eigenvalue: Float[Array, ""] = jnp.mean(
                activation_eigenvalues[layer_name]
            ) * jnp.mean(gradient_eigenvalues[layer_name])
            mean_eigenvalues[layer_name] = mean_eigenvalue
            mean_corrections[layer_name] = mean_correction
            mean_corrections_aggregated += mean_correction
            mean_eigenvalues_aggregated += mean_eigenvalue

        # Divide overall sums by number of layers
        n_layers = len(eigenvalue_corrections)
        mean_corrections_aggregated = jnp.array(
            mean_corrections_aggregated / n_layers if n_layers > 0 else 0.0
        )
        mean_eigenvalues_aggregated = jnp.array(
            mean_eigenvalues_aggregated / n_layers if n_layers > 0 else 0.0
        )

        logger.info("Computed mean eigenvalue corrections for EK-FAC approximation.")

        return EKFACData(
            activation_eigenvectors=activation_eigenvectors,
            gradient_eigenvectors=gradient_eigenvectors,
            activation_eigenvalues=activation_eigenvalues,
            gradient_eigenvalues=gradient_eigenvalues,
            eigenvalue_corrections=eigenvalue_corrections,
            mean_eigenvalues=mean_eigenvalues,
            mean_eigenvalues_aggregated=mean_eigenvalues_aggregated,
            mean_corrections=mean_corrections,
            mean_corrections_aggregated=mean_corrections_aggregated,
        )
