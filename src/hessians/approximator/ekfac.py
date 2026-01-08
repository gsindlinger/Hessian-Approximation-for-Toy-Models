import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from src.config import DampingStrategy
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

    collected_data_path_second: str
    """Path to second set of previously collected data"""

    def _build(
        self,
    ):
        """Method to build the required components to compute"""
        # Load first activations and gradients from path
        if self.collected_data_path is None or self.collected_data_path_second is None:
            raise ValueError(
                "Both collected_data_path and collected_data_path_snd must be provided to load EK-FAC data."
            )

        logger.info("Loading activations and gradients for EK-FAC approximation.")
        # Note: We approximate the FIM using MCMC and by collecting gradients
        # and activations from two independent runs we try to reduce bias.
        # This is therefore the data from the first run which is used to compute the eigenvectors.
        collected_data = CollectorActivationsGradients.load(
            directory=self.collected_data_path
        )
        activations, gradients, layer_names = (
            collected_data.activations,
            collected_data.gradients,
            collected_data.layer_names,
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
        # Note: For the eigenvalue corrections we have a different estimate which we want to obtain,
        # so we use data from a second independent run.
        collected_data = CollectorActivationsGradients.load(
            directory=self.collected_data_path_second
        )
        (activations, gradients) = (
            collected_data.activations,
            collected_data.gradients,
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
            layer_names=layer_names,
        )

    @staticmethod
    def compute_covariances(
        activations: Dict[str, Float[Array, "N I"]],
        gradients: Dict[str, Float[Array, "N O"]],
    ) -> Tuple[Dict[str, Float[Array, "I I"]], Dict[str, Float[Array, "O O"]]]:
        """
        Compute covariance matrices for activations and gradients for each layer.
        """
        activation_covariances = ApproximatorBase.batched_collector_processing(
            layer_keys=activations.keys(),
            num_samples=list(activations.values())[0].shape[0],
            compute_fn=EKFACApproximator._compute_covariance,
            data={"activations": activations},
        )

        gradient_covariances = ApproximatorBase.batched_collector_processing(
            layer_keys=gradients.keys(),
            num_samples=list(gradients.values())[0].shape[0],
            compute_fn=EKFACApproximator._compute_covariance,
            data={"gradients": gradients},
        )
        return activation_covariances, gradient_covariances

    @staticmethod
    def _compute_covariance(**x: Dict[str, Float[Array, "N D"]]) -> Float[Array, "D D"]:
        """Compute covariance matrix of x.
        Expects x as a dict with a single key, e.g. 'activations' or 'gradients'."""
        x_item = next(iter(x.values()))
        return jnp.einsum("ni,nj->ij", x_item, x_item)

    @staticmethod
    def get_damping(
        ekfac_data: EKFACData,
        damping_strategy: DampingStrategy,
        factor: float,
    ) -> Float:
        """Get damping value for a specific layer based on mean eigenvalues."""
        match damping_strategy:
            case DampingStrategy.FIXED:
                return factor
            case DampingStrategy.AUTO_MEAN_EIGENVALUE:
                return ekfac_data.mean_eigenvalues_aggregated * factor
            case DampingStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION:
                return ekfac_data.mean_corrections_aggregated * factor
            case _:
                raise ValueError(f"Unknown damping strategy: {damping_strategy}")
