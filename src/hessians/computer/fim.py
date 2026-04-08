from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy
from src.hessians.computer.computer import CollectorBasedHessianEstimator
from src.hessians.layer_matrix import LayerMatrix, LayerVector
from src.hessians.utils.data import DataActivationsGradients, FIMData
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class FIMComputer(CollectorBasedHessianEstimator):
    """
    Fisher Information Matrix approximation.

    The Fisher Information Matrix is defined as:
    FIM = E[∇log p(y|x) ∇log p(y|x)^T]

    Use previously collected gradients to compute the FIM using its outer product.

    Supports different pseudo-target strategies:
    - EMPIRICAL_FISHER: Uses ground truth labels (k=1)
    - MCMC: Uses sampled pseudo-targets (k=num_samples)
    - ALL_CLASSES: Uses all classes with probability weighting (k=num_classes)
    """

    precomputed_data: FIMData = field(default_factory=FIMData)

    @classmethod
    def _build(
        cls,
        compute_context: DataActivationsGradients,
    ) -> FIMData:
        """Build FIMData from activations and gradients."""
        strategy = compute_context.pseudo_target_strategy
        layer_names = compute_context.layer_names

        n_samples = compute_context.activations[layer_names[0]].shape[0]

        grads_per_layer = []

        for layer in layer_names:
            a = compute_context.activations[layer]  # (N, I_l)
            g = compute_context.gradients[layer]  # (K, N, O_l)

            k = g.shape[0]

            # Compute parameter gradients: a ⊗ g
            # Expand activations from (N, I) to (k, N, I)
            a_expanded = jnp.broadcast_to(a[None, :, :], (k, n_samples, a.shape[-1]))

            # Compute outer product and flatten: (k, N, I) ⊗ (k, N, O) -> (k, N, I*O)
            G_l = jnp.einsum("kni,kno->knio", a_expanded, g).reshape(k, n_samples, -1)

            grads_per_layer.append(G_l)

        grads_all = jnp.concatenate(grads_per_layer, axis=2)

        # Store probabilities separately for ALL_CLASSES strategy
        probabilities = (
            compute_context.probabilities
            if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES
            else None
        )

        return FIMData(per_sample_grads=grads_all, probabilities=probabilities)

    # ------------------------------------------------------------------
    # LayerMatrix construction
    # ------------------------------------------------------------------

    def get_layer_names(self) -> List[str]:
        return self.compute_context.layer_names

    def _layer_shapes(self) -> Dict[str, Tuple[int, int]]:
        return {
            l: (
                int(self.compute_context.activations[l].shape[-1]),
                int(self.compute_context.gradients[l].shape[-1]),
            )
            for l in self.get_layer_names()
        }

    def _get_layer_matrix(self) -> LayerMatrix:
        """Materialize the FIM and slice it into per-layer DenseBlocks."""
        gradients = self.precomputed_data.per_sample_grads  # (k, N, n_params)
        strategy = self.compute_context.pseudo_target_strategy
        if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            dense = self._compute_fim_all_classes(
                gradients=gradients,
                probabilities=self.precomputed_data.probabilities,
                damping=0.0,
            )
        else:
            dense = self._compute_fim_unweighted(gradients, 0.0)
        return LayerMatrix.from_dense(
            dense,
            param_groups=self.get_layer_names(),
            layer_shapes=self._layer_shapes(),
        )

    # ------------------------------------------------------------------
    # HessianEstimator interface (thin wrappers over LayerMatrix)
    # ------------------------------------------------------------------

    def _estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """Materialized FIM (optionally damped)."""
        d = 0.0 if damping is None else damping
        return self._get_layer_matrix().damped(d).to_dense()

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """Compare the FIM against `comparison_matrix` under the given metric."""
        d = 0.0 if damping is None else damping
        fim = self._estimate_hessian(d)
        return metric.compute_fn()(comparison_matrix, fim)

    @staticmethod
    @jax.jit
    def _compute_fim_unweighted(
        gradients: Float[Array, "k N n_params"],
        damping: Float = 0.0,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute FIM for EMPIRICAL_FISHER and MCMC strategies.

        FIM = (1/N) * (1/k) * sum_k sum_n g_{k,n} g_{k,n}^T + damping * I

        For EMPIRICAL_FISHER: k=1, so this reduces to standard empirical FIM
        For MCMC: k>1, averages over sampled pseudo-targets
        """
        k, n_samples, n_params = gradients.shape

        def body_fn(accum, idx):
            k_idx = idx // n_samples
            n_idx = idx % n_samples
            g = gradients[k_idx, n_idx]  # (n_params,)
            return accum + jnp.outer(g, g), None

        fim0 = jnp.zeros((n_params, n_params), dtype=gradients.dtype)

        fim_sum, _ = jax.lax.scan(
            body_fn,
            fim0,
            jnp.arange(k * n_samples),
        )

        # Average over both k and N
        fim = fim_sum / (k * n_samples)
        fim += damping * jnp.eye(n_params, dtype=fim.dtype)
        return fim

    @staticmethod
    @jax.jit
    def _compute_fim_all_classes(
        gradients: Float[Array, "K N n_params"],
        probabilities: Float[Array, "N K"],
        damping: Float = 0.0,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute FIM for ALL_CLASSES strategy.

        FIM = (1/N) * sum_n sum_k p(k|n) * g_{k,n} g_{k,n}^T + damping * I

        This matches the reference implementation exactly.
        """
        K, n_samples, n_params = gradients.shape

        def per_sample_fim(n_idx):
            """Compute FIM contribution for sample n_idx"""
            fim_n = jnp.zeros((n_params, n_params), dtype=gradients.dtype)

            for k_idx in range(K):
                g = gradients[k_idx, n_idx]  # (n_params,)
                p = probabilities[n_idx, k_idx]  # scalar probability
                # Accumulate: p(k|n) * g_k g_k^T
                fim_n += p * jnp.outer(g, g)

            return fim_n

        # Average FIM over all samples
        fim_per_sample = jax.vmap(per_sample_fim)(jnp.arange(n_samples))
        fim = jnp.mean(fim_per_sample, axis=0)
        fim += damping * jnp.eye(n_params, dtype=fim.dtype)
        return fim

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Fisher-vector product via the materialized `LayerMatrix`."""
        d = 0.0 if damping is None else damping
        lmat = self._get_layer_matrix().damped(d)
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
        """Inverse Fisher-vector product via the materialized `LayerMatrix`."""
        d = 0.0 if damping is None else damping
        p = 0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
        lmat = self._get_layer_matrix().inverse(
            damping=d, pseudo_inverse_factor=p
        )
        lvec = LayerVector.from_flat(
            flat=vectors,
            shapes=lmat.vector_shapes(),
            param_groups=self.get_layer_names(),
        )
        return (lmat @ lvec).to_flat()

    @staticmethod
    @partial(jax.jit, static_argnames=["pseudo_inverse_factor"])
    def _solve_fim_inv(
        fim: Float[Array, "n_params n_params"],
        vectors: Float[Array, "batch_size n_params"],
        pseudo_inverse_factor: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Solve FIM^{-1} @ v for each vector v using direct linear solve.
        """
        if pseudo_inverse_factor > 0.0:
            jax.config.update("jax_enable_x64", True)
            eigvals, eigvecs = jnp.linalg.eigh(0.5 * (fim + fim.T))
            jax.config.update("jax_enable_x64", False)
            eigvals_inv = jnp.where(
                jnp.abs(eigvals) > pseudo_inverse_factor, 1.0 / eigvals, 0.0
            )
            ifvp = (eigvecs * eigvals_inv) @ (eigvecs.T @ vectors.T)
            return ifvp.T
        else:
            return jnp.linalg.solve(fim, vectors.T).T

    @staticmethod
    @partial(jax.jit, static_argnames=["sample_batch_size"])
    def _compute_fvp_unweighted(
        gradients: Float[Array, "k N n_params"],
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
        sample_batch_size: int = 264,
    ) -> Float[Array, "batch_size n_params"]:
        """
        JIT-compatible memory-efficient FVP with batching.
        """
        k, n_samples, n_params = gradients.shape
        total_samples = k * n_samples
        batch_size = vectors.shape[0]

        gradients_flat = gradients.reshape(total_samples, n_params)
        n_batches = (total_samples + sample_batch_size - 1) // sample_batch_size

        # Pad to make even batches
        pad_size = n_batches * sample_batch_size - total_samples
        if pad_size > 0:
            gradients_flat = jnp.pad(gradients_flat, ((0, pad_size), (0, 0)))

        gradients_batched = gradients_flat.reshape(
            n_batches, sample_batch_size, n_params
        )

        def body_fn(fvp, g_batch):
            # g_batch: (sample_batch_size, n_params)
            projections = g_batch @ vectors.T  # (sample_batch_size, batch_size)
            return fvp + (projections.T @ g_batch), None

        fvp, _ = jax.lax.scan(
            body_fn, jnp.zeros((batch_size, n_params)), gradients_batched
        )

        return fvp / total_samples + damping * vectors

    @staticmethod
    @partial(jax.jit, static_argnames=["sample_batch_size"])
    def _compute_fvp_all_classes(
        gradients: Float[Array, "K N n_params"],
        probabilities: Float[Array, "N K"],
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
        sample_batch_size: int = 264,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Fisher-vector product for ALL_CLASSES strategy with batching over samples.

        Computes: (1/N) * sum_n sum_k p(k|n) * (g_{k,n} @ v) * g_{k,n} + damping * v

        Batches over the N dimension to control memory usage.
        Memory per batch: O(K * sample_batch_size * batch_size) for projections
        """
        K, n_samples, n_params = gradients.shape
        batch_size = vectors.shape[0]

        n_batches = (n_samples + sample_batch_size - 1) // sample_batch_size

        # Pad gradients and probabilities to make even batches
        pad_size = n_batches * sample_batch_size - n_samples
        if pad_size > 0:
            # Pad along N dimension: gradients (K, N, n_params) -> (K, N + pad, n_params)
            gradients_padded = jnp.pad(gradients, ((0, 0), (0, pad_size), (0, 0)))
            # Pad probabilities (N, K) -> (N + pad, K) with zeros (won't contribute to sum)
            probabilities_padded = jnp.pad(probabilities, ((0, pad_size), (0, 0)))
        else:
            gradients_padded = gradients
            probabilities_padded = probabilities

        # Reshape for scanning: (K, n_batches, sample_batch_size, n_params)
        gradients_batched = gradients_padded.reshape(
            K, n_batches, sample_batch_size, n_params
        )
        # (n_batches, sample_batch_size, K)
        probabilities_batched = probabilities_padded.reshape(
            n_batches, sample_batch_size, K
        )

        def body_fn(fvp_accum, batch_data):
            g_batch, p_batch = batch_data
            # g_batch: (K, sample_batch_size, n_params)
            # p_batch: (sample_batch_size, K)

            # Compute projections: (K, sample_batch_size, n_params) @ (batch_size, n_params).T
            # -> (K, sample_batch_size, batch_size)
            projections = jnp.einsum("ksp,bp->ksb", g_batch, vectors)

            # Weight by probabilities: (sample_batch_size, K) -> (K, sample_batch_size, 1)
            prob_weights = p_batch.T[:, :, None]  # (K, sample_batch_size, 1)
            weighted_projections = (
                projections * prob_weights
            )  # (K, sample_batch_size, batch_size)

            # Accumulate: sum_k (p * (g^T v) * g)
            # (K, sample_batch_size, batch_size) with (K, sample_batch_size, n_params)
            # -> (batch_size, n_params)
            fvp_batch = jnp.einsum("ksb,ksp->bp", weighted_projections, g_batch)

            return fvp_accum + fvp_batch, None

        # Transpose gradients_batched for scan: (n_batches, K, sample_batch_size, n_params)
        gradients_scan = jnp.transpose(gradients_batched, (1, 0, 2, 3))

        fvp, _ = jax.lax.scan(
            body_fn,
            jnp.zeros((batch_size, n_params), dtype=gradients.dtype),
            (gradients_scan, probabilities_batched),
        )

        # Average over N (not over padded samples - but padding has zero probabilities)
        return fvp / n_samples + damping * vectors
