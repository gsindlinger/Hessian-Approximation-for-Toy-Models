from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.config import DampingStrategy
from src.hessians.computer.computer import CollectorBasedHessianEstimator
from src.hessians.utils.data import DataActivationsGradients, EKFACData
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric

logger = logging.getLogger(__name__)


@dataclass
class EKFACComputer(CollectorBasedHessianEstimator):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
    """

    precomputed_data: EKFACData = field(default_factory=EKFACData)

    @classmethod
    def _build(
        cls,
        compute_context: Tuple[DataActivationsGradients, DataActivationsGradients],
    ) -> EKFACData:
        """Method to build the required EK-FAC components to compute the Hessian approximation."""
        activations, gradients = (
            compute_context[0].activations,
            compute_context[0].gradients,
        )

        # Compute covariances of activations and gradients
        logger.info("Computing covariances for EK-FAC approximation.")

        covariances_dict = cls._batched_covariance_processing(
            activations, gradients, cls._compute_covariances
        )

        activations_covs, gradients_covs = (
            covariances_dict["activation_cov"],
            covariances_dict["gradient_cov"],
        )

        # Compute eigenvectors & eigenvalues of the covariances (discard eigenvalues since not needed in EK-FAC)
        (
            (activation_eigenvectors, gradient_eigenvectors),
            (activation_eigenvalues, gradient_eigenvalues),
        ) = EKFACComputer.compute_eigenvectors_and_eigenvalues(
            activations_covs, gradients_covs
        )
        logger.info("Computed eigenvectors for EK-FAC approximation.")

        (activations, gradients) = (
            compute_context[1].activations,
            compute_context[1].gradients,
        )
        eigenvalue_corrections = EKFACComputer.compute_eigenvalue_corrections(
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
        assert self.precomputed_data is not None, (
            "EKFAC data not computed. Please build the computer first."
        )
        """Compute Hessian-vector product."""
        return self.compute_ihvp_or_hvp(
            data=self.precomputed_data,
            vectors=vectors,
            layer_names=self.compute_context[0].layer_names,
            Lambdas=self.precomputed_data.eigenvalue_corrections,
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
        Compare the EKFAC Hessian approximation to a given comparison matrix.
        Reuses the KFAC comparison implementation.
        """
        assert self.precomputed_data is not None, (
            "EKFAC data not computed. Please build the computer first."
        )
        return self._compare_hessian_estimates(
            activations_eigenvectors=[
                self.precomputed_data.activation_eigenvectors[layer]
                for layer in self.compute_context[0].layer_names
            ],
            gradients_eigenvectors=[
                self.precomputed_data.gradient_eigenvectors[layer]
                for layer in self.compute_context[0].layer_names
            ],
            Lambdas=[
                self.precomputed_data.eigenvalue_corrections[layer]
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
            "EKFAC data not computed. Please build the computer first."
        )
        return self.compute_ihvp_or_hvp(
            data=self.precomputed_data,
            vectors=vectors,
            Lambdas=self.precomputed_data.eigenvalue_corrections,
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
        self,
        method: Literal["normal", "inverse"],
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        """
        Unified helper method to compute either the full Hessian or its inverse.
        Reuses the KFAC implementation.
        """
        assert self.precomputed_data is not None, (
            "EKFAC data not computed. Please build the computer first."
        )
        return self._compute_hessian_or_inverse_hessian_estimate(
            eigenvectors_activations=[
                self.precomputed_data.activation_eigenvectors[layer]
                for layer in self.compute_context[0].layer_names
            ],
            eigenvectors_gradients=[
                self.precomputed_data.gradient_eigenvectors[layer]
                for layer in self.compute_context[0].layer_names
            ],
            Lambdas=[
                self.precomputed_data.eigenvalue_corrections[layer]
                for layer in self.compute_context[0].layer_names
            ],
            damping=damping,
            method=method,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def _compute_hessian_or_inverse_hessian_estimate(
        eigenvectors_activations: List[Float[Array, "I I"]],
        eigenvectors_gradients: List[Float[Array, "O O"]],
        Lambdas: List[Float[Array, "I O"]],
        damping: Float[Array, ""],
        method: Literal["normal", "inverse"],
    ):
        """
        Computes a block-diagonal estimate for the Hessian or its inverse
        using Kronecker-factored eigenvectors and eigenvalues / corrections.

        Depending on the selected method:
        - "normal" computes:
              H ≈ (Q_G ⊗ Q_A) diag(Λ_G ⊗ Λ_A + λ) (Q_G ⊗ Q_A)ᵀ
        - "inverse" computes:
              H⁻¹ ≈ (Q_G ⊗ Q_A) diag(1 / (Λ_G ⊗ Λ_A + λ)) (Q_G ⊗ Q_A)ᵀ
        """

        hessian_list = [
            EKFACComputer._compute_layer_hessian_estimate(
                layer_eigv_activations,
                layer_eigv_gradients,
                EKFACComputer._get_damped_lambda(layer_lambda, damping, method),
            )
            for layer_eigv_activations, layer_eigv_gradients, layer_lambda in zip(
                eigenvectors_activations,
                eigenvectors_gradients,
                Lambdas,
            )
        ]

        return jax.scipy.linalg.block_diag(*hessian_list)

    @staticmethod
    def _get_damped_lambda(
        Lambda: Float[Array, "I O"],
        damping: Float[Array, ""],
        method: Literal["normal", "inverse"],
    ) -> Float[Array, "n_params n_params"]:
        """Compute the damped version of Lambda for the Hessian or its inverse."""
        if method == "inverse":
            return 1.0 / (Lambda + damping)
        else:
            return Lambda + damping

    @staticmethod
    @jax.jit
    def _compute_layer_hessian_estimate(
        eigenvectors_A: Float[Array, "I I"],
        eigenvectors_G: Float[Array, "O O"],
        Lambda: Float[Array, "I O"],
    ) -> Float[Array, "n_params n_params"]:
        """
        Computes the layer Hessian approximation given eigenvectors of activations and gradients,
        and the eigenvalue / correction matrix.

        Note: The formulation in KFAC, etc. assume weights shaped [d_out, d_in]
        with vec(∇W) = a ⊗ ∇s (column-major, similar to Pytorch).
        In contrast, JAX uses [d_in, d_out], which yields vec(∇W') = ∇s ⊗ a
        due to the forward pass formulation y = xW' instead of y = Wx (as in PyTorch).

        Because JAX flattens arrays in row-major (C-style) order, the effective
        vectorization swaps again, giving vec_row(∇W') = a ⊗ ∇s. This matches
        the ordering used when comparing with the true Hessian or constructing
        Kronecker-factored curvature blocks.

        Since we store the eigenvalues and corrections in the shape [input_dim, output_dim],
        we can directly use them here by flattening in JAX-default row-major order without needing to transpose.
        """
        return jnp.einsum(
            "ij,j,jk->ik",
            jnp.kron(eigenvectors_A, eigenvectors_G),
            Lambda.flatten(),
            jnp.kron(eigenvectors_A, eigenvectors_G).T,
        )

    @staticmethod
    def compute_ihvp_or_hvp(
        data: EKFACData,
        vectors: Float[Array, "*batch_size n_params"],
        Lambdas: Dict[str, Float[Array, "I O"]],
        layer_names: List[str],
        method: Literal["ihvp", "hvp"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product or Hessian-vector product.
        Unified method for different approaches, i.e., EK-FAC and K-FAC.
        Uses the respective eigenvectors which are stored in the compute context
        and the provided eigenvalues / corrections (Lambdas)

        Note, that the vector to be multiplied is reshaped in row-major order to
        match the JAX weight layout which is reflected by
        the eigenvalue corrections shape [input_dim, output_dim].
        """

        Q_activations_list = []
        Q_gradients_list = []
        Lambda_list = []
        v_layers = []

        offset = 0
        for layer_name in layer_names:
            Lambda = Lambdas[layer_name]
            input_dim, output_dim = Lambda.shape
            size = input_dim * output_dim

            # Extract and reshape vector for this layer
            v_flat: Float[Array, "*batch_size I*O"] = vectors[
                ..., offset : offset + size
            ]
            v_layer: Float[Array, "*batch_size I O"] = v_flat.reshape(
                v_flat.shape[:-1] + (input_dim, output_dim)
            )

            # Collect all components
            Q_activations_list.append(data.activation_eigenvectors[layer_name])
            Q_gradients_list.append(data.gradient_eigenvectors[layer_name])
            Lambda_list.append(Lambda)
            v_layers.append(v_layer)
            offset += size

        # Compute (I)HVP for all layers
        vp_pieces = EKFACComputer.compute_ihvp_or_hvp_all_layers(
            v_layers=v_layers,
            Q_activations=Q_activations_list,
            Q_gradients=Q_gradients_list,
            Lambdas=Lambda_list,
            damping=damping,
            method=method,
        )

        # Concatenate all layer results
        return jnp.concatenate(vp_pieces, axis=-1)

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def compute_ihvp_or_hvp_all_layers(
        v_layers: list[Float[Array, "*batch_size I O"]],
        Q_activations: list[Float[Array, "I I"]],
        Q_gradients: list[Float[Array, "O O"]],
        Lambdas: list[Float[Array, "I O"]],
        damping: Float[Array, ""],
        method: Literal["ihvp", "hvp"],
    ) -> list[Float[Array, "*batch_size num_params"]]:
        """
        Computes the inverse Hessian-vector product (IHVP) or Hessian-vector product (HVP) for multiple layers.
        Uses the Kronecker-factored eigenvectors and a corresponding eigenvalue / corrections matrix per layer.
        Returns a list of flattened vector products, one per layer.
        """
        vp_pieces = []

        for v_layer, Q_A, Q_G, Lambda in zip(
            v_layers, Q_activations, Q_gradients, Lambdas
        ):
            # Transform to eigenbasis
            V_tilde: Float[Array, "*batch_size I O"] = Q_A.T @ v_layer @ Q_G

            # Apply eigenvalue corrections + damping
            Lambda_damped: Float[Array, "I O"] = Lambda + damping

            if method == "ihvp":
                scaled: Float[Array, "*batch_size I O"] = V_tilde / Lambda_damped
            else:
                scaled: Float[Array, "*batch_size I O"] = V_tilde * Lambda_damped

            # Transform back to original basis
            vector_product: Float[Array, "*batch_size I O"] = Q_A @ scaled @ Q_G.T

            # Flatten last two dimensions: [*batch_size, I, O] -> [*batch_size, I*O]
            # This works for both single vector (I, O) and batched (*batch, I, O)
            batch_shape = vector_product.shape[:-2]
            flat_size = vector_product.shape[-2] * vector_product.shape[-1]
            vp_flat = vector_product.reshape(*batch_shape, flat_size)
            vp_pieces.append(vp_flat)

        return vp_pieces

    @staticmethod
    @partial(jax.jit, static_argnames=["method", "metric"])
    def _compare_hessian_estimates(
        activations_eigenvectors: List[Float[Array, "I I"]],
        gradients_eigenvectors: List[Float[Array, "O O"]],
        Lambdas: List[Float[Array, "I O"]],
        damping: Float[Array, ""],
        comparison_matrix: Float[Array, "n_params n_params"],
        metric: Callable[[jnp.ndarray, jnp.ndarray], float],
        method: Literal["normal", "inverse"] = "normal",
    ) -> float:
        """Compare (E)KFAC Hessian or its inverse to a given comparison matrix and prespecified metric."""
        kfac_hessian = EKFACComputer._compute_hessian_or_inverse_hessian_estimate(
            activations_eigenvectors,
            gradients_eigenvectors,
            Lambdas,
            damping=damping,
            method=method,
        )

        true_hessian = comparison_matrix + damping * jnp.eye(kfac_hessian.shape[0])
        return metric(true_hessian, kfac_hessian)

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

    @staticmethod
    def compute_eigenvalue_corrections(
        activations: Dict[str, Float[Array, "N I"]],
        gradients: Dict[str, Float[Array, "N O"]],
        activation_eigenvectors: Dict[str, Float[Array, "I I"]],
        gradient_eigenvectors: Dict[str, Float[Array, "O O"]],
    ) -> Dict[str, Float[Array, "I O"]]:
        """
        Compute eigenvalue corrections for each layer.
        """
        return EKFACComputer.batched_collector_processing(
            layer_keys=activations.keys(),
            num_samples=list(activations.values())[0].shape[0],
            compute_fn=EKFACComputer._compute_eigenvalue_correction_batch,
            data={
                "activations": activations,
                "gradients": gradients,
            },
            static_data={
                "Q_A": activation_eigenvectors,
                "Q_G": gradient_eigenvectors,
            },
        )

    @staticmethod
    @jax.jit
    def _compute_eigenvalue_correction_batch(
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        activations: Float[Array, "N I"],
        gradients: Float[Array, "N O"],
    ) -> Float[Array, "I O"]:
        r"""
        Compute eigenvalue correction for a given layer.

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
        """
        g_tilde = jnp.einsum("op, np -> no", Q_G.T, gradients)
        a_tilde = jnp.einsum("ij, nj -> ni", Q_A.T, activations)
        outer = jnp.einsum("ni, no -> nio", a_tilde, g_tilde)
        return (outer**2).sum(axis=0)

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

    @staticmethod
    def _compute_covariances(
        activation_batch_dict: Dict[str, Float[Array, "N I"]],
        gradient_batch_dict: Dict[str, Float[Array, "N O"]],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Compute covariance matrix of x.
        Expects x as a dict with a single key, e.g. 'activations' or 'gradients'."""
        activation_cov_dict = {}
        gradient_cov_dict = {}
        for layer in activation_batch_dict.keys():
            activation_batch = activation_batch_dict[layer]
            gradient_batch = gradient_batch_dict[layer]

            activation_cov = jnp.einsum("ni,nj->ij", activation_batch, activation_batch)
            gradient_cov = jnp.einsum("ni,nj->ij", gradient_batch, gradient_batch)

            activation_cov_dict[layer] = activation_cov
            gradient_cov_dict[layer] = gradient_cov

        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }

    @staticmethod
    def _batched_covariance_processing(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O"]],
        compute_fn: Callable,
        batch_size: int | None = None,
    ) -> Dict[str, Dict[str, Float[Array, "..."]]]:
        """Process activations and gradients to compute covariances in batches."""
        if batch_size is None:
            batch_size = JAXDataLoader.get_batch_size()

        num_samples = list(activations_dict.values())[0].shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        accumulator: Dict[str, Dict[str, Array]] = {}

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            activation_batch_dict = {
                layer: activations_dict[layer][start_idx:end_idx]
                for layer in activations_dict
            }
            gradient_batch_dict = {
                layer: gradients_dict[layer][start_idx:end_idx]
                for layer in gradients_dict
            }

            batch_result: Dict[str, Dict[str, Array]] = compute_fn(
                activation_batch_dict=activation_batch_dict,
                gradient_batch_dict=gradient_batch_dict,
            )  # dict with keys typically "activation_..." and "gradient_..."

            if batch_idx == 0:
                accumulator = batch_result
            else:
                for key in batch_result.keys():
                    for k, v in batch_result[key].items():
                        accumulator[key][k] += v

        for key in accumulator.keys():
            for layer_name in activations_dict.keys():
                accumulator[key][layer_name] /= num_samples

        return accumulator

    @staticmethod
    def batched_collector_processing(
        layer_keys: Iterable[str],
        num_samples: int,
        compute_fn: Callable,
        data: Dict[str, Dict[str, Float[Array, "..."]]],
        static_data: Optional[Dict[str, Dict[str, Float[Array, "..."]]]] = None,
        normalize: bool = True,
    ) -> Dict[str, Float[Array, "..."]]:
        """Process collected data (e.g. activations, gradients) with optional additional
        information (e.g. precomputed eigenvectors) in batches to avoid memory issues.

        Since the method should be rather generic, the data is provided as dict with layer names as keys.
        The compute_fn method must accept keyword arguments matching the keys of the data dict.
        """
        batch_size = JAXDataLoader.get_batch_size()
        num_batches = (num_samples + batch_size - 1) // batch_size
        accumulator = {}

        # Loop over batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Loop over layer names (extract data for item in data dict)
            for layer_name in layer_keys:
                # Extract batch data for this layer
                data_batch = {
                    key: data[key][layer_name][start_idx:end_idx] for key in data
                }
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

        # Normalize by number of samples if requested
        if normalize:
            for layer_name in layer_keys:
                accumulator[layer_name] /= num_samples

        return accumulator
