from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy, RegularizationStrategy
from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import KroneckerFactors, LayerMatrix
from src.hessians.utils.data import DataActivationsGradients
from src.utils.data.jax_dataloader import JAXDataLoader

logger = logging.getLogger(__name__)


@dataclass
class EKFACComputer(HessianEstimator):
    compute_context: DataActivationsGradients
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected
    KFAC (EKFAC) Hessian approximation.

    We assume the compute_context as DataActivationsGradients where gradients
    always have shape (k, N, O):
    - EMPIRICAL_FISHER: k=1, uses same data for covariances and corrections
    - MCMC: k must be even and >=2, split equally between covariances and corrections
    - ALL_CLASSES: k=num_classes, uses all classes for both with probability weighting

    `_build` produces a block-diagonal `LayerMatrix` whose per-layer blocks are
    `KroneckerFactors(Q_A, Q_G, Lambda)`.  For EKFAC, `Lambda` is the
    eigenvalue correction; for `KFACComputer`, `Lambda = outer(λ_A, λ_G)`.
    """

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

    # ------------------------------------------------------------------
    # _build — produce the LayerMatrix
    # ------------------------------------------------------------------

    def _build(
        self,
        compute_context: DataActivationsGradients,
    ) -> LayerMatrix:
        """Assemble the block-diagonal LayerMatrix of KroneckerFactors blocks."""
        (
            activation_eigvecs,
            gradient_eigvecs,
            activation_eigvals,
            gradient_eigvals,
        ) = self._compute_eigendecomposition(compute_context)

        lambdas = self._compute_lambdas(
            compute_context=compute_context,
            activation_eigvecs=activation_eigvecs,
            gradient_eigvecs=gradient_eigvecs,
            activation_eigvals=activation_eigvals,
            gradient_eigvals=gradient_eigvals,
        )

        layer_names = list(compute_context.layer_names)
        diag_blocks: Dict[str, KroneckerFactors] = {
            layer: KroneckerFactors.from_eigendecomposition(
                Q_A=activation_eigvecs[layer],
                Q_G=gradient_eigvecs[layer],
                Lambda=lambdas[layer],
            )
            for layer in layer_names
        }
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks, param_groups=layer_names
        )

    def _compute_eigendecomposition(
        self, compute_context: DataActivationsGradients
    ) -> Tuple[
        Dict[str, Float[Array, "I I"]],
        Dict[str, Float[Array, "O O"]],
        Dict[str, Float[Array, "I"]],
        Dict[str, Float[Array, "O"]],
    ]:
        """Compute `(Q_A, Q_G, λ_A, λ_G)` per layer from the collected data.

        Handles the three pseudo-target strategies (MCMC splits the data
        for eigenvalue corrections; EKFAC uses the second split, KFAC
        ignores it).
        """
        strategy = compute_context.pseudo_target_strategy
        layer_names = compute_context.layer_names

        logger.info(
            f"Computing covariances for EK-FAC approximation with strategy: {strategy}"
        )

        # ---- Prepare (covariance) data split ----
        if strategy == PseudoTargetGenerationStrategy.MCMC:
            activations_cov = {}
            gradients_cov = {}
            for layer in layer_names:
                k = compute_context.gradients[layer].shape[0]
                split_idx = k // 2
                activations_cov[layer] = compute_context.activations[layer]
                gradients_cov[layer] = compute_context.gradients[layer][:split_idx]
            probabilities = None
        elif strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            activations_cov = compute_context.activations
            gradients_cov = compute_context.gradients
            probabilities = compute_context.probabilities
        else:  # EMPIRICAL_FISHER
            activations_cov = compute_context.activations
            gradients_cov = compute_context.gradients
            probabilities = None

        covariances_dict = self._batched_covariance_processing(
            activations_dict=activations_cov,
            gradients_dict=gradients_cov,
            probabilities=probabilities,
        )
        activations_covs = covariances_dict["activation_cov"]
        gradients_covs = covariances_dict["gradient_cov"]

        (
            (activation_eigvecs, gradient_eigvecs),
            (activation_eigvals, gradient_eigvals),
        ) = self.compute_eigenvectors_and_eigenvalues(
            activations_covs, gradients_covs
        )
        logger.info("Computed eigenvectors for EK-FAC approximation.")

        return (
            activation_eigvecs,
            gradient_eigvecs,
            activation_eigvals,
            gradient_eigvals,
        )

    def _compute_lambdas(
        self,
        compute_context: DataActivationsGradients,
        activation_eigvecs: Dict[str, Float[Array, "I I"]],
        gradient_eigvecs: Dict[str, Float[Array, "O O"]],
        activation_eigvals: Dict[str, Float[Array, "I"]],
        gradient_eigvals: Dict[str, Float[Array, "O"]],
    ) -> Dict[str, Float[Array, "I O"]]:
        """Per-layer `Λ` used as the `KroneckerFactors.Lambda` field.

        For EKFAC, `Λ` is the eigenvalue correction (computed from the
        second data split for MCMC, or the same data for the other strategies).
        `KFACComputer` overrides this to skip the correction step and return
        `outer(λ_A, λ_G)` directly.
        """
        strategy = compute_context.pseudo_target_strategy
        layer_names = compute_context.layer_names

        # ---- Prepare (correction) data split ----
        if strategy == PseudoTargetGenerationStrategy.MCMC:
            activations_corr = {}
            gradients_corr = {}
            for layer in layer_names:
                k = compute_context.gradients[layer].shape[0]
                split_idx = k // 2
                activations_corr[layer] = compute_context.activations[layer]
                gradients_corr[layer] = compute_context.gradients[layer][split_idx:]
            probabilities = None
        elif strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            activations_corr = compute_context.activations
            gradients_corr = compute_context.gradients
            probabilities = compute_context.probabilities
        else:
            activations_corr = compute_context.activations
            gradients_corr = compute_context.gradients
            probabilities = None

        if probabilities is not None:
            batch_size = JAXDataLoader.get_batch_size() // probabilities.shape[1]
            compute_fn = self._compute_eigenvalue_correction_batch_weighted
            data = {
                "activations": activations_corr,
                "gradients": gradients_corr,
                "probabilities": probabilities,
            }
        else:
            batch_size = JAXDataLoader.get_batch_size()
            compute_fn = self._compute_eigenvalue_correction_batch
            data = {
                "activations": activations_corr,
                "gradients": gradients_corr,
            }

        eigenvalue_corrections = self.batched_collector_processing(
            layer_keys=layer_names,
            num_samples=activations_corr[layer_names[0]].shape[0],
            compute_fn=compute_fn,
            data=data,
            static_data={
                "Q_A": activation_eigvecs,
                "Q_G": gradient_eigvecs,
            },
            batch_size=batch_size,
        )
        logger.info("Computed eigenvalue corrections for EK-FAC approximation.")
        return eigenvalue_corrections

    # ------------------------------------------------------------------
    # Damping helpers
    # ------------------------------------------------------------------

    def get_layer_names(self) -> List[str]:
        """Get the list of layer names from the compute context."""
        return self.compute_context.layer_names

    def get_damping(
        self,
        damping_strategy: RegularizationStrategy,
        factor: float,
    ) -> float:
        """Get damping value based on strategy.

        For non-fixed strategies, reads `mean(Lambda)` across the per-layer
        KroneckerFactors blocks of the built `LayerMatrix`.  (Lambda for
        EKFAC is the eigenvalue correction; for KFAC it is `outer(λ_A, λ_G)`.)

        Both `AUTO_MEAN_EIGENVALUE` and `AUTO_MEAN_EIGENVALUE_CORRECTION`
        collapse to the same computation in this codepath: average over
        layers of `mean(block.Lambda)`.
        """
        if damping_strategy == RegularizationStrategy.FIXED:
            return factor
        if damping_strategy not in (
            RegularizationStrategy.AUTO_MEAN_EIGENVALUE,
            RegularizationStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION,
        ):
            raise ValueError(
                f"Unsupported regularization strategy: {damping_strategy}"
            )
        if self.layer_matrix is None:
            raise RuntimeError(
                "EKFACComputer is not built — call `.build()` before `.get_damping()`."
            )
        means = []
        for layer in self.get_layer_names():
            block = self.layer_matrix.blocks[(layer, layer)]
            assert isinstance(block, KroneckerFactors)
            means.append(jnp.mean(block.Lambda))
        if not means:
            return 0.0
        aggregated = jnp.mean(jnp.stack(means))
        return float(aggregated) * factor

    # ------------------------------------------------------------------
    # Notebook compatibility: keep `compute_hessian_or_inverse_hessian_estimate`
    # so `tests/pytorch_jax_nn_geometry_comparison.ipynb` still works.
    # ------------------------------------------------------------------

    def compute_hessian_or_inverse_hessian_estimate(
        self,
        method: str,
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        """Compute the full Hessian estimate (`"normal"`) or its inverse."""
        if self.layer_matrix is None:
            raise RuntimeError(
                "EKFACComputer is not built — call `.build()` first."
            )
        if method == "inverse":
            lmat = self.layer_matrix.inverse(damping=damping)
        elif method == "normal":
            lmat = self.layer_matrix.damped(damping)
        else:
            raise ValueError(f"Unknown method: {method!r}")
        return lmat.to_dense()

    def estimate_inverse_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """Compute the full inverse Hessian (convenience wrapper)."""
        return self.compute_hessian_or_inverse_hessian_estimate(
            method="inverse",
            damping=0.0 if damping is None else damping,
        )

    # ------------------------------------------------------------------
    # Covariance / eigendecomposition helpers (unchanged)
    # ------------------------------------------------------------------

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
            (Q_G \\otimes Q_A)^T vec(a_n g_n^T) = (Q_G \\otimes Q_A)^T (g_n \\otimes a_n)

        Using the Kronecker product property (A \\otimes B)^T = A^T \\otimes B^T and
        the mixed-product property, this simplifies to:
            (Q_G^T g_n) \\otimes (Q_A^T a_n) = vec((Q_A^T a_n) (Q_G^T g_n)^T)
        """
        if activations.ndim == 3:
            raise ValueError("Activations should be (N, I) format, got (k, N, I)")
        if gradients.ndim == 2:
            gradients = gradients[None, ...]
        k, N, _ = gradients.shape

        a_tilde = jnp.einsum("ij,nj->ni", Q_A.T, activations)  # (N, I)
        g_tilde = jnp.einsum("op,knp->kno", Q_G.T, gradients)  # (k, N, O)

        a_tilde_expanded = jnp.broadcast_to(
            a_tilde[None, :, :], (k, N, a_tilde.shape[-1])
        )
        outer = jnp.einsum("kni,kno->knio", a_tilde_expanded, g_tilde)
        squared = outer**2  # (k, N, I, O)
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
        """Eigenvalue correction for ALL_CLASSES strategy."""
        if activations.ndim == 3:
            raise ValueError("Activations should be (N, I) format, got (k, N, I)")
        if gradients.ndim == 2:
            gradients = gradients[None, ...]
        K, N, _ = gradients.shape

        a_tilde = jnp.einsum("ij,nj->ni", Q_A.T, activations)  # (N, I)
        sqrt_weights = jnp.sqrt(probabilities.T)[..., None]  # (K, N, 1)
        weighted_gradients = gradients * sqrt_weights  # (K, N, O)
        g_tilde = jnp.einsum("op,knp->kno", Q_G.T, weighted_gradients)

        a_tilde_expanded = jnp.broadcast_to(
            a_tilde[None, :, :], (K, N, a_tilde.shape[-1])
        )
        outer = jnp.einsum("kni,kno->knio", a_tilde_expanded, g_tilde)
        squared = outer**2
        correction = jnp.sum(squared, axis=(0, 1))
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
        activation_eigvals: Dict[str, jnp.ndarray] = {}
        activation_eigvecs: Dict[str, jnp.ndarray] = {}
        gradient_eigvals: Dict[str, jnp.ndarray] = {}
        gradient_eigvecs: Dict[str, jnp.ndarray] = {}

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

        return (
            (activation_eigvecs, gradient_eigvecs),
            (activation_eigvals, gradient_eigvals),
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
        jax.config.update("jax_enable_x64", True)
        A = A.astype(jnp.float64)
        G = G.astype(jnp.float64)

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
        """
        activation_cov_dict = {}
        gradient_cov_dict = {}

        for layer in activation_batch_dict.keys():
            activations = activation_batch_dict[layer]  # (N, I)
            gradients = gradient_batch_dict[layer]  # (k, N, O)
            k = gradients.shape[0]

            activation_cov = jnp.einsum("ni,nj->ij", activations, activations)
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
        """
        activation_cov_dict = {}
        gradient_cov_dict = {}

        for layer in activation_batch_dict.keys():
            activations = activation_batch_dict[layer]  # (N, I)
            gradients = gradient_batch_dict[layer]  # (K, N, O)
            N, I = activations.shape
            K, _, O = gradients.shape

            activation_cov = jnp.einsum("ni,nj->ij", activations, activations)

            sqrt_weights = jnp.sqrt(probabilities.T)[..., None]  # (K, N, 1)
            weighted_grads = gradients * sqrt_weights  # (K, N, O)
            weighted_grads_flat = weighted_grads.reshape(-1, O)  # (K*N, O)

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
        """
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
                layer: gradients_dict[layer][:, start_idx:end_idx]
                for layer in gradients_dict
            }

            if probabilities is not None:
                batch_probs = probabilities[start_idx:end_idx]
                batch_result = EKFACComputer._compute_covariances_weighted(
                    activation_batch_dict=activation_batch_dict,
                    gradient_batch_dict=gradient_batch_dict,
                    probabilities=batch_probs,
                )
            else:
                batch_result = EKFACComputer._compute_covariances(
                    activation_batch_dict=activation_batch_dict,
                    gradient_batch_dict=gradient_batch_dict,
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
        """Process collected data in batches with optional static data."""
        if batch_size is None:
            batch_size = JAXDataLoader.get_batch_size()
        num_batches = (num_samples + batch_size - 1) // batch_size
        accumulator: Dict[str, Array] = {}

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            for layer_name in layer_keys:
                data_batch: Dict[str, object] = {}
                for key in data:
                    if key == "probabilities":
                        if data[key] is not None:
                            assert isinstance(data[key], jnp.ndarray)
                            data_batch[key] = data[key][start_idx:end_idx]  # type: ignore
                        else:
                            data_batch[key] = None
                    elif data[key] is not None and layer_name in data[key]:  # type: ignore
                        layer_data = data[key][layer_name]  # type: ignore
                        if layer_data is None:
                            data_batch[key] = None
                        else:
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

                static_layer_data = (
                    {
                        static_data_key: static_data[static_data_key][layer_name]
                        for static_data_key in static_data
                    }
                    if static_data is not None
                    else {}
                )

                batch_result = compute_fn(**data_batch, **static_layer_data)
                if batch_idx == 0:
                    accumulator[layer_name] = batch_result
                else:
                    accumulator[layer_name] += batch_result

        for layer_name in layer_keys:
            accumulator[layer_name] /= num_samples

        return accumulator
