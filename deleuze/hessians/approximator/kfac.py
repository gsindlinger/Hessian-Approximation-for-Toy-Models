import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from deleuze.hessians.approximator.approximator import ApproximatorBase
from deleuze.hessians.collector import (
    CollectorKFACCovariances,
)
from deleuze.hessians.utils.data import KFACData
from deleuze.models.approximation_model import ApproximationModel
from deleuze.utils.data.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class KFACApproximator(ApproximatorBase):
    """
    Builder for K-FAC Hessian approximation.
    """

    data: KFACData = field(default_factory=KFACData)

    def _build(
        self,
        model: ApproximationModel,
        params: Dict,
        dataset: Dataset,
        loss_fn: Callable,
    ):
        # Collect covariances by forward and backward passes
        (activations_covs, gradients_covs) = CollectorKFACCovariances(
            model, params
        ).collect(dataset.inputs, dataset.targets, loss_fn)

        logger.info("Finished collecting covariances for K-FAC approximation.")

        # Compute eigenvectors and eigenvalues of the covariances
        (
            (self.data.activation_eigenvectors, self.data.gradient_eigenvectors),
            (self.data.activation_eigenvalues, self.data.gradient_eigenvalues),
        ) = compute_eigenvectors_and_eigenvalues(activations_covs, gradients_covs)

        logger.info("Computed eigenvectors and eigenvalues for K-FAC approximation.")

        # Compute mean eigenvalues and eigenvalue corrections for damping
        self.data.mean_eigenvalues = {}
        self.data.mean_eigenvalues_aggregated = 0.0

        for layer_name in self.data.activation_eigenvalues.keys():
            mean_eigenvalue: Float[Array, ""] = jnp.mean(
                self.data.activation_eigenvalues[layer_name]
            ) * jnp.mean(self.data.gradient_eigenvalues[layer_name])
            self.data.mean_eigenvalues[layer_name] = mean_eigenvalue
            self.data.mean_eigenvalues_aggregated += mean_eigenvalue

        # Divide overall sums by number of layers
        n_layers = len(self.data.activation_eigenvalues)
        self.data.mean_eigenvalues_aggregated = jnp.array(
            self.data.mean_eigenvalues_aggregated / n_layers if n_layers > 0 else 0.0
        )

        logger.info("Computed mean eigenvalues for K-FAC approximation.")


def compute_eigenvectors_and_eigenvalues(
    activations_covs: Dict[str, jnp.ndarray],
    gradients_covs: Dict[str, jnp.ndarray],
) -> Tuple[
    Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
]:
    """Compute eigenvectors of the covariance matrices A and G for each layer."""

    activation_eigvals = {}
    activation_eigvecs = {}
    gradient_eigvals = {}
    gradient_eigvecs = {}

    for layer_name in activations_covs.keys():
        (
            activation_eigvals_layer,
            gradient_eigvals_layer,
            activation_eigvecs_layer,
            gradient_eigvecs_layer,
        ) = compute_layer_eigenvectors(
            activations_covs[layer_name],
            gradients_covs[layer_name],
        )

        activation_eigvals[layer_name] = activation_eigvals_layer
        activation_eigvecs[layer_name] = activation_eigvecs_layer
        gradient_eigvals[layer_name] = gradient_eigvals_layer
        gradient_eigvecs[layer_name] = gradient_eigvecs_layer

    return (activation_eigvecs, gradient_eigvecs), (
        activation_eigvals,
        gradient_eigvals,
    )


@jax.jit
def compute_layer_eigenvectors(
    A: Float[Array, "I I"], G: Float[Array, "O O"]
) -> Tuple[
    Float[Array, "I"], Float[Array, "O"], Float[Array, "I I"], Float[Array, "O O"]
]:
    """
    Compute eigenvectors of covariance matrices A and G for a single layer.

    Returns:
        Tuple containing:
        - Eigenvalues of A (Float[Array, "I"])
        - Eigenvalues of G (Float[Array, "O"])
        - Eigenvectors of A (Float[Array, "I I"])
        - Eigenvectors of G (Float[Array, "O O"])
    """
    # Ensure numerical stability by using float64 for eigen decomposition
    jax.config.update("jax_enable_x64", True)
    A = A.astype(jnp.float64)
    G = G.astype(jnp.float64)

    eigenvals_A: Float[Array, "I"]
    eigvecs_A: Float[Array, "I I"]
    eigenvals_G: Float[Array, "O"]
    eigvecs_G: Float[Array, "O O"]

    eigenvals_A, eigvecs_A = jnp.linalg.eigh(A)
    eigenvals_G, eigvecs_G = jnp.linalg.eigh(G)
    jax.config.update("jax_enable_x64", False)

    return (
        eigenvals_A.astype(jnp.float32),
        eigenvals_G.astype(jnp.float32),
        eigvecs_A.astype(jnp.float32),
        eigvecs_G.astype(jnp.float32),
    )
