from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy
from src.hessians.computer.computer import CollectorBasedHessianEstimator
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


class FIMBlockComputer(CollectorBasedHessianEstimator):
    """
    Block-diagonal Fisher Information Matrix approximation.

    Computes F = block_diag(F_1, ..., F_L) where each F_l is the FIM for layer l.

    Supports different pseudo-target strategies:
    - EMPIRICAL_FISHER: Uses ground truth labels (k=1)
    - MCMC: Uses sampled pseudo-targets (k=num_samples)
    - ALL_CLASSES: Uses all classes with probability weighting (k=num_classes)
    """

    def _estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute the Fisher Information Matrix block approximation.
        """
        damping = 0.0 if damping is None else damping
        strategy = self.compute_context.pseudo_target_strategy

        # Extract activations and gradients
        activations = [
            self.compute_context.activations[layer]
            for layer in self.compute_context.layer_names
        ]
        gradients = [
            self.compute_context.gradients[layer]
            for layer in self.compute_context.layer_names
        ]

        if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            return self._compute_fim_block_weighted(
                activations=activations,
                gradients=gradients,
                probabilities=self.compute_context.probabilities,
                damping=damping,
            )
        else:
            return self._compute_fim_block_unweighted(
                activations=activations,
                gradients=gradients,
                damping=damping,
            )

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the Fisher Information Matrix block-vector product.
        """
        damping = 0.0 if damping is None else damping
        strategy = self.compute_context.pseudo_target_strategy

        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        activations = [
            self.compute_context.activations[layer]
            for layer in self.compute_context.layer_names
        ]
        gradients = [
            self.compute_context.gradients[layer]
            for layer in self.compute_context.layer_names
        ]

        if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            result_2D = self._compute_fim_block_hvp_weighted(
                activations=activations,
                gradients=gradients,
                probabilities=self.compute_context.probabilities,
                vectors=vectors_2D,
                damping=damping,
            )
        else:
            result_2D = self._compute_fim_block_hvp_unweighted(
                activations=activations,
                gradients=gradients,
                vectors=vectors_2D,
                damping=damping,
            )
        if is_single:
            return result_2D.squeeze(0)
        else:
            return result_2D

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare the FIM block approximation with another Hessian matrix.
        """
        damping = 0.0 if damping is None else damping
        fim = self._estimate_hessian(damping)
        return metric.compute_fn()(comparison_matrix, fim)

    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
        pseudo_inverse_factor: Optional[float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the inverse Fisher Information Matrix block-vector product.
        """
        damping = 0.0 if damping is None else damping
        pseudo_inverse_factor = (
            0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
        )
        strategy = self.compute_context.pseudo_target_strategy
        activations = [
            self.compute_context.activations[layer]
            for layer in self.compute_context.layer_names
        ]
        gradients = [
            self.compute_context.gradients[layer]
            for layer in self.compute_context.layer_names
        ]

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            result_2D = self._compute_fim_block_ihvp_weighted(
                activations=activations,
                gradients=gradients,
                probabilities=self.compute_context.probabilities,
                vectors=vectors_2D,
                damping=damping,
                pseudo_inverse_factor=pseudo_inverse_factor,
            )
        else:
            result_2D = self._compute_fim_block_ihvp_unweighted(
                activations=activations,
                gradients=gradients,
                vectors=vectors_2D,
                damping=damping,
                pseudo_inverse_factor=pseudo_inverse_factor,
            )
        if is_single:
            return result_2D.squeeze(0)
        else:
            return result_2D

    # -------------------------------------------------------------------------
    # Unweighted methods (EMPIRICAL_FISHER and MCMC)
    # -------------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _compute_fim_block_unweighted(
        activations: list[Float[Array, "N I"]],
        gradients: list[Float[Array, "K N O"]],
        damping: float,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute block-diagonal FIM for EMPIRICAL_FISHER and MCMC.

        For each layer block: F_l = (1/(K*N)) Σ_k Σ_n w^(k,n) (w^(k,n))^T
        where w^(k,n) = g_l^(k,n) ⊗ a_{l-1}^(n)
        """
        fim_blocks = []

        for act, grad in zip(activations, gradients):
            N, I = act.shape
            K, _, O = grad.shape

            # Expand activations: (N, I) -> (K, N, I)
            act_expanded = jnp.broadcast_to(act[None, :, :], (K, N, I))

            # Compute per-sample parameter gradients and flatten: (K, N, I*O)
            per_sample_vecs = jnp.einsum("kni,kno->knio", act_expanded, grad).reshape(
                K, N, -1
            )

            # Flatten to (K*N, I*O)
            vecs_flat = per_sample_vecs.reshape(K * N, -1)

            # Compute FIM block: (1/(K*N)) * vecs_flat^T @ vecs_flat
            fim_block = (vecs_flat.T @ vecs_flat) / (K * N)

            # Add damping
            fim_block = fim_block + damping * jnp.eye(fim_block.shape[0])

            fim_blocks.append(fim_block)

        return jax.scipy.linalg.block_diag(*fim_blocks)

    @staticmethod
    @jax.jit
    def _compute_fim_block_hvp_unweighted(
        activations: list[Float[Array, "N I"]],
        gradients: list[Float[Array, "K N O"]],
        vectors: Float[Array, "*batch_size n_params"],
        damping: float,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute (F + λI)v for block-diagonal FIM (unweighted).

        Uses efficient formulation: F_l v = (1/(K*N)) Σ_k Σ_n <v, w^(k,n)> w^(k,n)
        """
        results = []
        offset = 0

        for act, grad in zip(activations, gradients):
            N, I = act.shape
            K, _, O = grad.shape
            D = I * O

            v_block = vectors[..., offset : offset + D]

            # Expand activations and compute per-sample vecs
            act_expanded = jnp.broadcast_to(act[None, :, :], (K, N, I))
            per_sample_vecs = jnp.einsum("kni,kno->knio", act_expanded, grad).reshape(
                K, N, D
            )
            vecs_flat = per_sample_vecs.reshape(K * N, D)

            # Compute coefficients: <v, w^(k,n)>
            coeffs = jnp.einsum("...d,kd->...k", v_block, vecs_flat)

            # Compute result: (1/(K*N)) Σ <v, w^(k,n)> w^(k,n)
            y_block = jnp.einsum("...k,kd->...d", coeffs, vecs_flat) / (K * N)

            # Add damping
            y_block = y_block + damping * v_block

            results.append(y_block)
            offset += D

        return jnp.concatenate(results, axis=-1)

    @staticmethod
    @partial(jax.jit, static_argnames=["pseudo_inverse_factor"])
    def _compute_fim_block_ihvp_unweighted(
        activations: list[Float[Array, "N I"]],
        gradients: list[Float[Array, "K N O"]],
        vectors: Float[Array, "*batch_size n_params"],
        damping: float,
        pseudo_inverse_factor: float,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute (F + λI)^(-1)v for block-diagonal FIM (unweighted).
        """
        results = []
        offset = 0

        for act, grad in zip(activations, gradients):
            N, I = act.shape
            K, _, O = grad.shape
            D = I * O

            v_block = vectors[..., offset : offset + D]

            # Compute per-sample vecs and FIM block
            act_expanded = jnp.broadcast_to(act[None, :, :], (K, N, I))
            per_sample_vecs = jnp.einsum("kni,kno->knio", act_expanded, grad).reshape(
                K, N, D
            )
            vecs_flat = per_sample_vecs.reshape(K * N, D)
            fim_block = (vecs_flat.T @ vecs_flat) / (K * N)

            if pseudo_inverse_factor > 0.0:
                jax.config.update("jax_enable_x64", True)
                eigvals, eigvecs = jnp.linalg.eigh(0.5 * (fim_block + fim_block.T))
                eigvals_inv = jnp.where(
                    jnp.abs(eigvals) > pseudo_inverse_factor, 1.0 / eigvals, 0.0
                )
                jax.config.update("jax_enable_x64", False)
                y_block = jnp.einsum(
                    "ij,j,jk,nk->ni", eigvecs, eigvals_inv, eigvecs.T, v_block
                )
            else:
                fim_block = fim_block + damping * jnp.eye(D)
                y_block = jnp.linalg.solve(fim_block, v_block.T).T

            results.append(y_block)
            offset += D

        return jnp.concatenate(results, axis=-1)

    # -------------------------------------------------------------------------
    # Weighted methods (ALL_CLASSES)
    # -------------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _compute_fim_block_weighted(
        activations: list[Float[Array, "N I"]],
        gradients: list[Float[Array, "K N O"]],
        probabilities: Float[Array, "N K"],
        damping: float,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute block-diagonal FIM for ALL_CLASSES strategy.

        For each layer block: F_l = (1/N) Σ_n Σ_k p(k|n) * w^(k,n) (w^(k,n))^T
        """
        fim_blocks = []

        for act, grad in zip(activations, gradients):
            N, I = act.shape
            K, _, _ = grad.shape

            # Expand activations: (N, I) -> (K, N, I)
            act_expanded = jnp.broadcast_to(act[None, :, :], (K, N, I))

            # Compute per-sample parameter gradients: (K, N, I*O)
            per_sample_vecs = jnp.einsum("kni,kno->knio", act_expanded, grad).reshape(
                K, N, -1
            )

            # Apply sqrt of probabilities: (N, K) -> (K, N, 1)
            sqrt_probs = jnp.sqrt(probabilities.T)[..., None]  # (K, N, 1)

            # Weight vectors: (K, N, D)
            weighted_vecs = per_sample_vecs * sqrt_probs

            # Flatten to (K*N, D)
            weighted_vecs_flat = weighted_vecs.reshape(K * N, -1)

            # Compute FIM block: (1/N) * weighted_vecs_flat^T @ weighted_vecs_flat
            fim_block = (weighted_vecs_flat.T @ weighted_vecs_flat) / N

            # Add damping
            fim_block = fim_block + damping * jnp.eye(fim_block.shape[0])

            fim_blocks.append(fim_block)

        return jax.scipy.linalg.block_diag(*fim_blocks)

    @staticmethod
    @jax.jit
    def _compute_fim_block_hvp_weighted(
        activations: list[Float[Array, "N I"]],
        gradients: list[Float[Array, "K N O"]],
        probabilities: Float[Array, "N K"],
        vectors: Float[Array, "*batch_size n_params"],
        damping: float,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute (F + λI)v for block-diagonal FIM (weighted).
        """
        results = []
        offset = 0

        for act, grad in zip(activations, gradients):
            N, I = act.shape
            K, _, O = grad.shape
            D = I * O

            v_block = vectors[..., offset : offset + D]

            # Compute per-sample vecs
            act_expanded = jnp.broadcast_to(act[None, :, :], (K, N, I))
            per_sample_vecs = jnp.einsum("kni,kno->knio", act_expanded, grad).reshape(
                K, N, D
            )

            # Apply sqrt of probabilities
            sqrt_probs = jnp.sqrt(probabilities.T)[..., None]  # (K, N, 1)
            weighted_vecs = per_sample_vecs * sqrt_probs
            weighted_vecs_flat = weighted_vecs.reshape(K * N, D)

            # Compute coefficients
            coeffs = jnp.einsum("...d,kd->...k", v_block, weighted_vecs_flat)

            # Compute result: (1/N) Σ <v, w^(k,n)> w^(k,n)
            y_block = jnp.einsum("...k,kd->...d", coeffs, weighted_vecs_flat) / N

            # Add damping
            y_block = y_block + damping * v_block

            results.append(y_block)
            offset += D

        return jnp.concatenate(results, axis=-1)

    @staticmethod
    @partial(jax.jit, static_argnames=["pseudo_inverse_factor"])
    def _compute_fim_block_ihvp_weighted(
        activations: list[Float[Array, "N I"]],
        gradients: list[Float[Array, "K N O"]],
        probabilities: Float[Array, "N K"],
        vectors: Float[Array, "*batch_size n_params"],
        damping: float,
        pseudo_inverse_factor: float,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute (F + λI)^(-1)v for block-diagonal FIM (weighted).
        """
        results = []
        offset = 0

        for act, grad in zip(activations, gradients):
            N, I = act.shape
            K, _, O = grad.shape
            D = I * O

            v_block = vectors[..., offset : offset + D]

            # Compute per-sample vecs
            act_expanded = jnp.broadcast_to(act[None, :, :], (K, N, I))
            per_sample_vecs = jnp.einsum("kni,kno->knio", act_expanded, grad).reshape(
                K, N, D
            )

            # Apply sqrt of probabilities and compute FIM block
            sqrt_probs = jnp.sqrt(probabilities.T)[..., None]
            weighted_vecs = per_sample_vecs * sqrt_probs
            weighted_vecs_flat = weighted_vecs.reshape(K * N, D)
            fim_block = (weighted_vecs_flat.T @ weighted_vecs_flat) / N

            if pseudo_inverse_factor > 0.0:
                jax.config.update("jax_enable_x64", True)
                eigvals, eigvecs = jnp.linalg.eigh(0.5 * (fim_block + fim_block.T))
                eigvals_inv = jnp.where(
                    jnp.abs(eigvals) > pseudo_inverse_factor, 1.0 / eigvals, 0.0
                )
                jax.config.update("jax_enable_x64", False)
                y_block = jnp.einsum(
                    "ij,j,jk,nk->ni", eigvecs, eigvals_inv, eigvecs.T, v_block
                )
            else:
                fim_block = fim_block + damping * jnp.eye(D)
                y_block = jnp.linalg.solve(fim_block, v_block.T).T

            results.append(y_block)
            offset += D

        return jnp.concatenate(results, axis=-1)
