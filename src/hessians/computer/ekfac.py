from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy, RegularizationStrategy
from src.hessians.computer.computer import CollectorBasedHessianEstimator
from src.hessians.layer_matrix import KroneckerFactors, LayerMatrix, LayerVector
from src.hessians.utils.data import (
    DataActivationsGradients,
    EKFACData,
)
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric

logger = logging.getLogger(__name__)


@dataclass
class EKFACComputer(CollectorBasedHessianEstimator):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.

    We assume the compute_context as DataActivationsGradients where gradients always have shape (k, N, O):
    - EMPIRICAL_FISHER: k=1, uses same data for covariances and corrections
    - MCMC: k must be even and >=2, split equally between covariances and corrections
    - ALL_CLASSES: k=num_classes, uses all classes for both with probability weighting
    """

    precomputed_data: EKFACData = field(default_factory=EKFACData)

    def __post_init__(self):
        """
        Validate that compute_context meets requirements for each pseudo-target strategy.
        """
        strategy = self.compute_context.pseudo_target_strategy

        if strategy == PseudoTargetGenerationStrategy.MCMC:
            # MCMC requires at least 2 runs and even number to split
            for layer in self.compute_context.layer_names:
                k = self.compute_context.gradients[layer].shape[0]
                if k < 2 or k % 2 != 0:
                    raise ValueError(
                        f"EK-FAC with MCMC pseudo-targets requires at least two collector runs "
                        f"and an even number of collector runs to split the data for eigenvalue corrections. "
                        f"Layer {layer} has k={k}"
                    )

        elif strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            # ALL_CLASSES should have k = num_classes
            if self.compute_context.probabilities is None:
                raise ValueError(
                    "EK-FAC with ALL_CLASSES pseudo-targets requires probabilities to be provided."
                )
            num_classes = self.compute_context.probabilities.shape[1]
            for layer in self.compute_context.layer_names:
                k = self.compute_context.gradients[layer].shape[0]
                if k != num_classes:
                    raise ValueError(
                        f"EK-FAC with ALL_CLASSES pseudo-targets requires k={num_classes} classes. "
                        f"Layer {layer} has k={k}"
                    )

    @classmethod
    def _build(
        cls,
        compute_context: DataActivationsGradients,
    ) -> EKFACData:
        """
        Build the required EK-FAC components to compute the Hessian approximation.

        Handles different pseudo-target strategies:
        - EMPIRICAL_FISHER: Uses same data (k=1) for both covariances and corrections
        - MCMC: Splits k runs equally between covariances and corrections
        - ALL_CLASSES: Uses all k classes for both, with probability weighting
        """
        strategy = compute_context.pseudo_target_strategy
        layer_names = compute_context.layer_names

        logger.info(
            f"Computing covariances for EK-FAC approximation with strategy: {strategy}"
        )

        batch_size = JAXDataLoader.get_batch_size()

        # Prepare data based on strategy
        if strategy == PseudoTargetGenerationStrategy.MCMC:
            # Split MCMC runs: first half for covariances, second half for corrections
            activations_cov = {}
            gradients_cov = {}
            activations_corr = {}
            gradients_corr = {}

            for layer in layer_names:
                k = compute_context.gradients[layer].shape[0]
                split_idx = k // 2

                # First half for covariances
                activations_cov[layer] = compute_context.activations[layer]  # (N, I)
                gradients_cov[layer] = compute_context.gradients[layer][
                    :split_idx
                ]  # (k/2, N, O)

                # Second half for corrections
                activations_corr[layer] = compute_context.activations[layer]  # (N, I)
                gradients_corr[layer] = compute_context.gradients[layer][
                    split_idx:
                ]  # (k/2, N, O)

            # For MCMC, no probabilities are used
            probabilities = None

        elif strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            # Use all classes for both covariances and corrections
            activations_cov = compute_context.activations
            gradients_cov = compute_context.gradients
            activations_corr = compute_context.activations
            gradients_corr = compute_context.gradients

            # Use probabilities for weighting
            probabilities = compute_context.probabilities

        else:  # EMPIRICAL_FISHER
            # Use same data (k=1) for both covariances and corrections
            activations_cov = compute_context.activations
            gradients_cov = compute_context.gradients
            activations_corr = compute_context.activations
            gradients_corr = compute_context.gradients

            # No probabilities for empirical Fisher
            probabilities = None

        # Compute covariances using the first data split
        covariances_dict = cls._batched_covariance_processing(
            activations_dict=activations_cov,
            gradients_dict=gradients_cov,
            probabilities=probabilities,
        )

        activations_covs = covariances_dict["activation_cov"]
        gradients_covs = covariances_dict["gradient_cov"]

        # Compute eigenvectors & eigenvalues
        (
            (activation_eigenvectors, gradient_eigenvectors),
            (activation_eigenvalues, gradient_eigenvalues),
        ) = cls.compute_eigenvectors_and_eigenvalues(activations_covs, gradients_covs)

        logger.info("Computed eigenvectors for EK-FAC approximation.")

        # Compute eigenvalue corrections using the second data split
        data = {}
        if probabilities is not None:
            # Adjust batch size for probabilities
            batch_size = JAXDataLoader.get_batch_size() // probabilities.shape[1]
            compute_fn = cls._compute_eigenvalue_correction_batch_weighted
            data = {
                "activations": activations_corr,
                "gradients": gradients_corr,
                "probabilities": probabilities,
            }
        else:
            batch_size = JAXDataLoader.get_batch_size()
            compute_fn = cls._compute_eigenvalue_correction_batch
            data = {
                "activations": activations_corr,
                "gradients": gradients_corr,
            }
        k_corr = gradients_corr[layer_names[0]].shape[0]
        eigenvalue_corrections = cls.batched_collector_processing(
            layer_keys=layer_names,
            num_samples=activations_corr[layer_names[0]].shape[0],
            compute_fn=compute_fn,
            data=data,
            static_data={
                "Q_A": activation_eigenvectors,
                "Q_G": gradient_eigenvectors,
            },
            batch_size=batch_size,
        )
        logger.info("Computed eigenvalue corrections for EK-FAC approximation.")

        # Compute mean eigenvalues and corrections for damping
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

    def _get_lambdas(self) -> Dict[str, Float[Array, "I O"]]:
        """
        Per-layer eigenvalues used to build the `LayerMatrix`.

        For EKFAC these are the eigenvalue corrections; `KFACComputer`
        overrides this to return `outer(λ_A, λ_G)` instead.
        """
        assert self.precomputed_data is not None, (
            "EKFAC data not computed. Please build the computer first."
        )
        return self.precomputed_data.eigenvalue_corrections

    def _get_layer_matrix(self) -> LayerMatrix:
        """Build a block-diagonal `LayerMatrix` from `precomputed_data`.

        Each diagonal block is a `KroneckerFactors` with `Q_A`, `Q_G`, and
        `Lambda` taken from the precomputed data (or, for KFAC, computed
        via `_get_lambdas`).
        """
        assert self.precomputed_data is not None, (
            "EKFAC data not computed. Please build the computer first."
        )
        layer_names = self.get_layer_names()
        lambdas = self._get_lambdas()
        diag_blocks: Dict[str, KroneckerFactors] = {
            layer: KroneckerFactors.from_eigendecomposition(
                Q_A=self.precomputed_data.activation_eigenvectors[layer],
                Q_G=self.precomputed_data.gradient_eigenvectors[layer],
                Lambda=lambdas[layer],
            )
            for layer in layer_names
        }
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks, param_groups=layer_names
        )

    def _estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """Compute full Hessian approximation."""
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
        damping_f = 0.0 if damping is None else damping
        lmat = self._get_layer_matrix().damped(damping_f)
        lvec = LayerVector.from_flat(
            flat=vectors,
            shapes=lmat.vector_shapes(),
            param_groups=self.get_layer_names(),
        )
        return (lmat @ lvec).to_flat()

    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
        pseudo_inverse_factor: Optional[float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute inverse Hessian-vector product."""
        damping_f = 0.0 if damping is None else damping
        pseudo_f = 0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
        lmat = self._get_layer_matrix().inverse(
            damping=damping_f, pseudo_inverse_factor=pseudo_f
        )
        lvec = LayerVector.from_flat(
            flat=vectors,
            shapes=lmat.vector_shapes(),
            param_groups=self.get_layer_names(),
        )
        return (lmat @ lvec).to_flat()

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> float:
        """Compare the EKFAC Hessian approximation to a given comparison matrix."""
        damping_f = 0.0 if damping is None else damping
        kfac_hessian = self._get_layer_matrix().damped(damping_f).to_dense()
        true_hessian = comparison_matrix + damping_f * jnp.eye(kfac_hessian.shape[0])
        return metric.compute_fn()(true_hessian, kfac_hessian)

    def estimate_inverse_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """Compute full inverse Hessian."""
        return self.compute_hessian_or_inverse_hessian_estimate(
            method="inverse",
            damping=0.0 if damping is None else damping,
        )

    def compute_hessian_or_inverse_hessian_estimate(
        self,
        method: Literal["normal", "inverse"],
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        """Compute the full Hessian estimate (`"normal"`) or its inverse.

        Kept as a public method because the notebook
        `tests/pytorch_jax_nn_geometry_comparison.ipynb` uses it directly.
        """
        lmat = self._get_layer_matrix()
        if method == "inverse":
            lmat = lmat.inverse(damping=damping)
        else:
            lmat = lmat.damped(damping)
        return lmat.to_dense()

    @staticmethod
    def get_damping(
        ekfac_data: EKFACData,
        damping_strategy: RegularizationStrategy,
        factor: float,
    ) -> float:
        """Get damping value based on strategy."""
        match damping_strategy:
            case RegularizationStrategy.FIXED:
                return factor
            case RegularizationStrategy.AUTO_MEAN_EIGENVALUE:
                return float(ekfac_data.mean_eigenvalues_aggregated) * factor
            case RegularizationStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION:
                return float(ekfac_data.mean_corrections_aggregated) * factor
            case _:
                raise ValueError(
                    f"Unsupported regularization strategy: {damping_strategy}"
                )

    @staticmethod
    @jax.jit
    def _compute_eigenvalue_correction_batch(
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        activations: Float[Array, "N I"],
        gradients: Float[Array, "K N O"],
    ) -> Float[Array, "I O"]:
        """
        Compute eigenvalue correction for EMPIRICAL_FISHER and MCMC strategies.
        For each sample n, we compute:
            (Q_G \otimes Q_A)^T vec(a_n g_n^T) = (Q_G \otimes Q_A)^T (g_n \otimes a_n)

        Using the Kronecker product property (A \otimes B)^T = A^T \otimes B^T and
        the mixed-product property, this simplifies to:
            (Q_G^T g_n) \otimes (Q_A^T a_n) = vec((Q_A^T a_n) (Q_G^T g_n)^T)

        where:
        - Q_A, Q_G are the eigenvector matrices of the activation and gradient covariances
        - a_n, g_n are the activation and pre-activation gradient vectors for sample n
        - \otimes denotes the Kronecker product

        Implementation steps:
        1. Transform activations to eigenbasis: a_tilde_n = Q_A^T @ a_n
        2. Transform gradients to eigenbasis: g_tilde_n = Q_G^T @ g_n
        3. Compute outer product / Kronecker product: a_tilde_n \otimes g_tilde_n
        4. Square and sum across samples (averaging is later done by the caller)

        Note:
        The paper of Grosse et al. (2023) misses the transpose of the eigenvector
        basis (Q_A \otimes Q_G) in Equation (20). Refer to George et al. (2018) for the
        correct formulation.

        Handles:
        - EMPIRICAL_FISHER: activations (N, I), gradients (1, N, O) or (N, O)
        - MCMC: activations (N, I), gradients (k/2, N, O)
        """
        # Handle activations - ensure (N, I) format
        if activations.ndim == 3:
            raise ValueError("Activations should be (N, I) format, got (k, N, I)")

        # Handle gradients - ensure proper format
        if gradients.ndim == 2:
            # (N, O) - expand to (1, N, O) for EMPIRICAL_FISHER
            gradients = gradients[None, ...]  # (1, N, O)
        # gradients is now (k, N, O)

        k, N, _ = gradients.shape

        # Transform to eigenbasis
        a_tilde = jnp.einsum("ij,nj->ni", Q_A.T, activations)  # (N, I)
        g_tilde = jnp.einsum("op,knp->kno", Q_G.T, gradients)  # (k, N, O)

        # Expand a_tilde to match g_tilde: (N, I) -> (k, N, I)
        a_tilde_expanded = jnp.broadcast_to(
            a_tilde[None, :, :], (k, N, a_tilde.shape[-1])
        )

        # Compute outer products: (k, N, I) x (k, N, O) -> (k, N, I, O)
        outer = jnp.einsum("kni,kno->knio", a_tilde_expanded, g_tilde)

        # Square the outer products
        squared = outer**2  # (k, N, I, O)

        # EMPIRICAL_FISHER or MCMC: sum over N and average over k
        # Sum over N: (K, N, I, O) -> (K, I, O)
        # Average over k: (K, I, O) -> (I, O)
        correction = jnp.sum(squared, axis=(0, 1))  # (I, O)

        return correction

    @staticmethod
    @jax.jit
    def _compute_eigenvalue_correction_batch_weighted(
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        activations: Float[Array, "N I"],
        gradients: Float[Array, "K N O"],
        probabilities: Float[Array, "N K"],
    ) -> Float[Array, "I O"]:
        """..."""
        if activations.ndim == 3:
            raise ValueError("Activations should be (N, I) format, got (k, N, I)")

        if gradients.ndim == 2:
            gradients = gradients[None, ...]

        K, N, _ = gradients.shape

        # Transform activations to eigenbasis
        a_tilde = jnp.einsum("ij,nj->ni", Q_A.T, activations)  # (N, I)

        # CRITICAL FIX: Weight gradients BEFORE transforming to eigenbasis
        sqrt_weights = jnp.sqrt(probabilities.T)[..., None]  # (K, N, 1)
        weighted_gradients = gradients * sqrt_weights  # (K, N, O)

        # Transform WEIGHTED gradients to eigenbasis
        g_tilde = jnp.einsum("op,knp->kno", Q_G.T, weighted_gradients)  # (K, N, O)

        # Expand a_tilde to match g_tilde: (N, I) -> (K, N, I)
        a_tilde_expanded = jnp.broadcast_to(
            a_tilde[None, :, :], (K, N, a_tilde.shape[-1])
        )

        # Compute outer products: (K, N, I) x (K, N, O) -> (K, N, I, O)
        outer = jnp.einsum("kni,kno->knio", a_tilde_expanded, g_tilde)

        # Square the outer products
        squared = outer**2  # (K, N, I, O)

        # Sum over K and N (weighting already applied through sqrt_weights)
        correction = jnp.sum(squared, axis=(0, 1))  # (I, O)

        return correction

    @staticmethod
    def compute_eigenvectors_and_eigenvalues(
        activations_covariances: Dict[str, jnp.ndarray],
        gradients_covariances: Dict[str, jnp.ndarray],
    ) -> Tuple[
        Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ]:
        """Compute eigenvectors of the covariance matrices A and G for each layer."""

        activation_eigvals = {}
        activation_eigvecs = {}
        gradient_eigvals = {}
        gradient_eigvecs = {}

        for layer_name in activations_covariances.keys():
            (
                activation_eigvals_layer,
                gradient_eigvals_layer,
                activation_eigvecs_layer,
                gradient_eigvecs_layer,
            ) = EKFACComputer.compute_layer_eigenvectors(
                activations_covariances[layer_name],
                gradients_covariances[layer_name],
            )

            activation_eigvals[layer_name] = activation_eigvals_layer
            activation_eigvecs[layer_name] = activation_eigvecs_layer
            gradient_eigvals[layer_name] = gradient_eigvals_layer
            gradient_eigvecs[layer_name] = gradient_eigvecs_layer

        return (activation_eigvecs, gradient_eigvecs), (
            activation_eigvals,
            gradient_eigvals,
        )

    @staticmethod
    @jax.jit
    def compute_layer_eigenvectors(
        A: Float[Array, "I I"], G: Float[Array, "O O"]
    ) -> Tuple[
        Float[Array, "I"],
        Float[Array, "O"],
        Float[Array, "I I"],
        Float[Array, "O O"],
    ]:
        """Compute eigenvectors of covariance matrices A and G for a single layer."""
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
            eigenvals_A.astype(jnp.float64),
            eigenvals_G.astype(jnp.float64),
            eigvecs_A.astype(jnp.float64),
            eigvecs_G.astype(jnp.float64),
        )

    @staticmethod
    @jax.jit
    def _compute_covariances(
        activation_batch_dict: Dict[str, Float[Array, "N I"]],
        gradient_batch_dict: Dict[str, Float[Array, "k N O"]],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """
        Compute covariance matrices for EMPIRICAL_FISHER and MCMC strategies.

        Handles:
        - EMPIRICAL_FISHER: gradients (1, N, O), no weighting
        - MCMC: gradients (k/2, N, O), uniform weighting
        """
        activation_cov_dict = {}
        gradient_cov_dict = {}

        for layer in activation_batch_dict.keys():
            activations = activation_batch_dict[layer]  # (N, I)
            gradients = gradient_batch_dict[layer]  # (k, N, O)

            N, I = activations.shape
            k, _, O = gradients.shape

            # Activation covariance: A^T A
            activation_cov = jnp.einsum("ni,nj->ij", activations, activations)  # (I, I)

            # Gradient covariance: G^T G
            gradient_cov = jnp.einsum("kno,knp->op", gradients, gradients) / k

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }

    @staticmethod
    @jax.jit
    def _compute_covariances_weighted(
        activation_batch_dict: Dict[str, Float[Array, "N I"]],
        gradient_batch_dict: Dict[str, Float[Array, "K N O"]],
        probabilities: Float[Array, "N K"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """
        Compute covariance matrices for ALL_CLASSES strategy with probability weighting.

        Handles:
        - ALL_CLASSES: gradients (K, N, O), probability weighting (N, K)
        """
        activation_cov_dict = {}
        gradient_cov_dict = {}

        for layer in activation_batch_dict.keys():
            activations = activation_batch_dict[layer]  # (N, I)
            gradients = gradient_batch_dict[layer]  # (K, N, O)

            N, I = activations.shape
            K, _, O = gradients.shape

            # Activation covariance: A^T A
            activation_cov = jnp.einsum("ni,nj->ij", activations, activations)  # (I, I)

            # ALL_CLASSES: apply sqrt of weights, then compute covariance
            # This avoids creating the full (K, N, O, O) tensor

            # Weight by sqrt of probabilities: (N, K) -> (K, N, 1)
            sqrt_weights = jnp.sqrt(probabilities.T)[..., None]  # (K, N, 1)

            # Apply sqrt weights to gradients: (K, N, O)
            weighted_grads = gradients * sqrt_weights  # (K, N, O)

            # Reshape to (K*N, O) for efficient matrix multiply
            weighted_grads_flat = weighted_grads.reshape(-1, O)  # (K*N, O)

            # Compute G = weighted_grads_flat^T @ weighted_grads_flat
            gradient_cov = jnp.einsum(
                "ko,kp->op", weighted_grads_flat, weighted_grads_flat
            )

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }

    @staticmethod
    def _batched_covariance_processing(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "k N O"]],
        probabilities: Optional[Float[Array, "N K"]] = None,
        batch_size: int | None = None,
    ) -> Dict[str, Dict[str, Float[Array, "..."]]]:
        """
        Process activations and gradients to compute covariances in batches.

        Works with:
        - activations: (N, I) format
        - gradients: (k, N, O) format where k varies by strategy
        - probabilities: (N, K) format for ALL_CLASSES strategy
        """
        if batch_size is None:
            batch_size = JAXDataLoader.get_batch_size()

        # N is in activations shape
        num_samples = list(activations_dict.values())[0].shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        accumulator: Dict[str, Dict[str, Array]] = {}

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Slice along N dimension
            activation_batch_dict = {
                layer: activations_dict[layer][start_idx:end_idx]  # (batch, I)
                for layer in activations_dict
            }
            gradient_batch_dict = {
                layer: gradients_dict[layer][:, start_idx:end_idx]  # (k, batch, O)
                for layer in gradients_dict
            }

            # Choose the appropriate computation function based on whether probabilities are provided
            if probabilities is not None:
                batch_probs = probabilities[start_idx:end_idx]
                batch_result: Dict[str, Dict[str, Array]] = (
                    EKFACComputer._compute_covariances_weighted(
                        activation_batch_dict=activation_batch_dict,
                        gradient_batch_dict=gradient_batch_dict,
                        probabilities=batch_probs,
                    )
                )
            else:
                batch_result: Dict[str, Dict[str, Array]] = (
                    EKFACComputer._compute_covariances(
                        activation_batch_dict=activation_batch_dict,
                        gradient_batch_dict=gradient_batch_dict,
                    )
                )

            if batch_idx == 0:
                accumulator = batch_result
            else:
                for key in batch_result.keys():
                    for k, v in batch_result[key].items():
                        accumulator[key][k] += v

        for key in accumulator.keys():
            for k in accumulator[key].keys():
                accumulator[key][k] /= num_samples

        return accumulator

    @staticmethod
    def batched_collector_processing(
        layer_keys: Iterable[str],
        num_samples: int,
        compute_fn: Callable,
        data: Dict[str, Dict[str, Float[Array, "..."]]]
        | Dict[str, Float[Array, "..."]],
        static_data: Optional[Dict[str, Dict[str, Float[Array, "..."]]]] = None,
        batch_size: int | None = None,
    ) -> Dict[str, Float[Array, "..."]]:
        """
        Process collected data in batches with optional static data.

        For MCMC, activations are (N, I) and gradients are (k/2, N, O).
        For other strategies, activations are (N, I) and gradients are (k, N, O).
        """
        if batch_size is None:
            batch_size = JAXDataLoader.get_batch_size()
        num_batches = (num_samples + batch_size - 1) // batch_size
        accumulator = {}

        # Loop over batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Loop over layer names
            for layer_name in layer_keys:
                # Extract batch data for this layer
                data_batch = {}
                for key in data:
                    if key == "probabilities":
                        # Special handling: probabilities is global (N, K), not per-layer
                        if data[key] is not None:
                            # Slice along dimension N (sample dimension)
                            assert isinstance(data[key], jnp.ndarray)
                            data_batch[key] = data[key][start_idx:end_idx]  # type: ignore
                        else:
                            data_batch[key] = None
                    elif data[key] is not None and layer_name in data[key]:  # type: ignore
                        # Per-layer data
                        layer_data = data[key][layer_name]  # type: ignore
                        if layer_data is None:
                            data_batch[key] = None
                        else:
                            # Slice along dimension -2 (sample dimension)
                            # For activations (N, I): slices along dim 0 (which is -2)
                            # For gradients (k, N, O): slices along dim 1 (which is -2)
                            ndim = layer_data.ndim
                            slice_obj = tuple(
                                slice(start_idx, end_idx)
                                if i == ndim - 2
                                else slice(None)
                                for i in range(ndim)
                            )
                            data_batch[key] = layer_data[slice_obj]
                    else:
                        data_batch[key] = None

                # Extract static data for this layer if provided
                static_layer_data = (
                    {
                        static_data_key: static_data[static_data_key][layer_name]
                        for static_data_key in static_data
                    }
                    if static_data is not None
                    else {}
                )

                # Compute result for this batch
                batch_result = compute_fn(**data_batch, **static_layer_data)

                # Accumulate results
                if batch_idx == 0:
                    accumulator[layer_name] = batch_result
                else:
                    accumulator[layer_name] += batch_result

        # Average over number of samples
        for layer_name in layer_keys:
            accumulator[layer_name] /= num_samples

        return accumulator

    def get_layer_names(self) -> List[str]:
        """Get the list of layer names from the compute context."""
        return self.compute_context.layer_names
