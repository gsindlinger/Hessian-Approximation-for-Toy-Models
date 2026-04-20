from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.config import RegularizationStrategy
from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import KroneckerFactors, LayerMatrix
from src.hessians.utils.data import DataActivationsGradients

logger = logging.getLogger(__name__)


@dataclass
class EKFACComputer(HessianEstimator):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected
    KFAC (EKFAC) Hessian approximation.

    Takes two `DataActivationsGradients` contexts:
    - `compute_context` (inherited): data for the covariance factors A, G.
    - `corr_context`: data for the eigenvalue correction Λ.

    For deterministic collection (EMPIRICAL_FISHER, ALL_CLASSES) the caller
    should pass the same object for both.  For MCMC, two independent
    collector runs (different rng keys) should be passed so the eigenvalue
    correction isn't fit on the same samples as the covariances.

    `_build` produces a block-diagonal `LayerMatrix` whose per-layer blocks
    are `KroneckerFactors(Q_A, Q_G, Lambda)`.  For EKFAC, `Lambda` is the
    eigenvalue correction; `KFACComputer` overrides `_compute_lambdas` to
    return `outer(λ_A, λ_G)` instead.

    Gradient layout: `(N, O, k)` where k is the number of pseudo-target draws.
    `probs` shape `(N, k)` carries per-draw weights (ones for EF/MCMC,
    softmax(logits) for ALL_CLASSES).
    """

    corr_context: Optional[DataActivationsGradients] = field(default=None)

    def __post_init__(self):
        # For deterministic strategies (EF, ALL_CLASSES) there is no sampling
        # noise to protect against, so reusing compute_context for the
        # correction stage is fine.  MCMC callers must pass two collected
        # datasets with different rng keys.
        if self.corr_context is None:
            self.corr_context = self.compute_context

    # ------------------------------------------------------------------
    # _build — produce the LayerMatrix
    # ------------------------------------------------------------------

    def _build(self, compute_context: DataActivationsGradients) -> LayerMatrix:
        """Assemble the block-diagonal LayerMatrix of KroneckerFactors blocks."""
        cov_data = self.compute_context
        corr_data = self.corr_context
        assert isinstance(cov_data, DataActivationsGradients)
        assert isinstance(corr_data, DataActivationsGradients)

        (
            activation_eigvecs,
            gradient_eigvecs,
            activation_eigvals,
            gradient_eigvals,
        ) = self._compute_eigendecomposition(cov_data)

        lambdas = self._compute_lambdas(
            compute_context=corr_data,
            activation_eigvecs=activation_eigvecs,
            gradient_eigvecs=gradient_eigvecs,
            activation_eigvals=activation_eigvals,
            gradient_eigvals=gradient_eigvals,
        )

        layer_names = list(cov_data.layer_names)
        diag_blocks: Dict[str, KroneckerFactors] = {
            layer: KroneckerFactors.from_eigendecomposition(
                Q_A=activation_eigvecs[layer],
                Q_G=gradient_eigvecs[layer],
                Lambda=lambdas[layer],
                lambda_A=activation_eigvals[layer],
                lambda_G=gradient_eigvals[layer],
            )
            for layer in layer_names
        }
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks, param_groups=layer_names
        )

    def _compute_eigendecomposition(
        self, data: DataActivationsGradients
    ) -> Tuple[
        Dict[str, Float[Array, "I I"]],
        Dict[str, Float[Array, "O O"]],
        Dict[str, Float[Array, "I"]],
        Dict[str, Float[Array, "O"]],
    ]:
        """Compute `(Q_A, Q_G, λ_A, λ_G)` per layer from the covariance data."""
        logger.info("Computing covariances for EK-FAC approximation.")
        covariances = self._compute_covariances(
            activations_dict=data.activations,
            gradients_dict=data.gradients,
            probs=data.probs,
        )
        (
            (activation_eigvecs, gradient_eigvecs),
            (activation_eigvals, gradient_eigvals),
        ) = self.compute_eigenvectors_and_eigenvalues(
            covariances["activation_cov"], covariances["gradient_cov"]
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

        For EKFAC, `Λ[i, o] = (Σ p_nk · (Q_A^T a)_{n,i}^2 (Q_G^T g)_{n,o,k}^2) / Σp`.
        `KFACComputer` overrides this to skip the correction and return
        `outer(λ_A, λ_G)`.
        """
        probs = compute_context.probs
        total_prob = probs.sum()
        lambdas: Dict[str, Array] = {}
        for layer in compute_context.layer_names:
            a = compute_context.activations[layer]  # (N, I)
            g = compute_context.gradients[layer]  # (N, O, k)
            Q_A = activation_eigvecs[layer]
            Q_G = gradient_eigvecs[layer]
            lambdas[layer] = self._compute_layer_lambda(
                a, g, probs, Q_A, Q_G, total_prob
            )
        logger.info("Computed eigenvalue corrections for EK-FAC approximation.")
        return lambdas

    @staticmethod
    @jax.jit
    def _compute_layer_lambda(
        a: Float[Array, "N I"],
        g: Float[Array, "N O k"],
        probs: Float[Array, "N k"],
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        total_prob: Float,
    ) -> Float[Array, "I O"]:
        """Λ = (Σ_n Σ_k p[n,k] · a_tilde_n,i^2 · g_tilde_{n,o,k}^2) / Σp.

        Since ((Q_A^T a)_i * (Q_G^T g)_o)^2 = a_tilde[i]^2 * g_tilde[o]^2,
        we can square first and contract with an einsum.
        """
        a_tilde_sq = (
            a @ Q_A
        ) ** 2  # (N, I)  — Q_A^T is applied via a @ Q_A since Q_A is orthogonal
        g_tilde_sq = jnp.einsum("op,npk->nok", Q_G.T, g) ** 2  # (N, O, k)
        return jnp.einsum("ni,nok,nk->io", a_tilde_sq, g_tilde_sq, probs) / total_prob

    # ------------------------------------------------------------------
    # Covariance / eigendecomposition helpers
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _compute_covariances(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O k"]],
        probs: Float[Array, "N k"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Unified weighted covariance:

            A_l = (1/N) Σ_n a_n a_n^T
            G_l = (Σ_n Σ_k p[n,k] g_{n,k} g_{n,k}^T) / Σp

        Activations don't depend on k so they use the uniform (1/N) average.
        """
        total_prob = probs.sum()
        activation_cov_dict = {}
        gradient_cov_dict = {}
        for layer in activations_dict.keys():
            a = activations_dict[layer]  # (N, I)
            g = gradients_dict[layer]  # (N, O, k)
            N = a.shape[0]
            activation_cov_dict[layer] = jnp.einsum("ni,nj->ij", a, a) / N
            gradient_cov_dict[layer] = (
                jnp.einsum("nok,npk,nk->op", g, g, probs) / total_prob
            )
        return {
            "activation_cov": activation_cov_dict,
            "gradient_cov": gradient_cov_dict,
        }

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
        """Compute eigenvectors of covariance matrices A and G for a single layer.

        eigh on near-rank-deficient covariances is unreliable in float32;
        promote to float64 for the decomposition and cast back to orig_dtype.
        """
        orig_dtype = A.dtype
        eigenvals_A, eigvecs_A = jnp.linalg.eigh(A.astype(jnp.float64))
        eigenvals_G, eigvecs_G = jnp.linalg.eigh(G.astype(jnp.float64))
        return (
            eigenvals_A.astype(orig_dtype),
            eigenvals_G.astype(orig_dtype),
            eigvecs_A.astype(orig_dtype),
            eigvecs_G.astype(orig_dtype),
        )

    # ------------------------------------------------------------------
    # Damping helpers
    # ------------------------------------------------------------------

    def get_layer_names(self) -> List[str]:
        """Get the list of layer names from the covariance context."""
        return self.compute_context.layer_names

    def get_damping(
        self,
        damping_strategy: RegularizationStrategy,
        factor: float,
    ) -> float:
        """Get damping value based on strategy.

        Reads per-layer statistics from the `KroneckerFactors` blocks of the
        built `LayerMatrix`:

        * `AUTO_MEAN_EIGENVALUE` → average over layers of
          `mean(λ_A) * mean(λ_G)`, the product of the raw activation and
          gradient covariance eigenvalue means.  For KFAC this equals
          `mean(Lambda)` because `Lambda = outer(λ_A, λ_G)`, but for EKFAC
          the eigenvalue correction diverges and this strategy reflects the
          *uncorrected* basis.
        * `AUTO_MEAN_EIGENVALUE_CORRECTION` → average over layers of
          `mean(block.Lambda)`, i.e. the eigenvalue-corrected Λ (equivalent
          to `AUTO_MEAN_EIGENVALUE` for KFAC, but the correction-aware
          version for EKFAC).
        """
        if damping_strategy == RegularizationStrategy.FIXED:
            return factor
        if damping_strategy not in (
            RegularizationStrategy.AUTO_MEAN_EIGENVALUE,
            RegularizationStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION,
        ):
            raise ValueError(f"Unsupported regularization strategy: {damping_strategy}")
        if self.layer_matrix is None:
            raise RuntimeError(
                "EKFACComputer is not built — call `.build()` before `.get_damping()`."
            )
        means = []
        for layer in self.get_layer_names():
            block = self.layer_matrix.blocks[(layer, layer)]
            assert isinstance(block, KroneckerFactors)
            if damping_strategy == RegularizationStrategy.AUTO_MEAN_EIGENVALUE:
                if block.lambda_A is None or block.lambda_G is None:
                    raise RuntimeError(
                        f"Layer '{layer}' KroneckerFactors block is missing "
                        f"`lambda_A` / `lambda_G` — AUTO_MEAN_EIGENVALUE "
                        f"requires the raw covariance eigenvalues.  Rebuild "
                        f"with the EKFAC pipeline, which populates them."
                    )
                means.append(jnp.mean(block.lambda_A) * jnp.mean(block.lambda_G))
            else:  # AUTO_MEAN_EIGENVALUE_CORRECTION
                means.append(jnp.mean(block.Lambda))
        if not means:
            return 0.0
        aggregated = jnp.mean(jnp.stack(means))
        return float(aggregated) * factor
