from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator, _accumulate_chunks
from src.hessians.layer_matrix import LayerMatrix
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class FIMComputer(HessianEstimator):
    compute_context: DataActivationsGradients
    batch_size: Optional[int] = field(default=None)
    """
    Fisher Information Matrix approximation.

    FIM = E[∇log p(y|x) ∇log p(y|x)^T]

    Unified weighted sum across pseudo-target draws:

        FIM = (Σ_n Σ_k p[n,k] · g_{n,k} g_{n,k}^T) / Σ_n Σ_k p[n,k]

    which recovers the three supported regimes:
    - EMPIRICAL_FISHER (k=1, p=ones):    (1/N) Σ_n g_n g_n^T
    - MCMC            (p=ones):          (1/(Nk)) Σ_n Σ_k g_{n,k} g_{n,k}^T
    - ALL_CLASSES     (p=softmax logits): (1/N) Σ_n Σ_k p(k|n) g_{n,k} g_{n,k}^T
    """

    def get_layer_names(self) -> List[str]:
        return self.compute_context.layer_names

    def _layer_shapes_from_context(
        self, compute_context: DataActivationsGradients
    ) -> Dict[str, Tuple[int, int]]:
        return {
            l: (
                int(compute_context.activations[l].shape[-1]),
                int(compute_context.gradients[l].shape[-2]),
            )
            for l in compute_context.layer_names
        }

    def _build(self) -> LayerMatrix:
        """Stream per-chunk FIM contributions and assemble the LayerMatrix.

        Each chunk reconstructs per-sample weight gradients from the
        collector's per-layer `(a, g)` and accumulates the outer-product
        sum — so we never materialize the full `(N, n_params, k)` tensor.
        """
        ctx = self.compute_context
        layer_names = list(ctx.layer_names)
        layer_shapes = self._layer_shapes_from_context(ctx)
        N = ctx.probs.shape[0]

        def _chunk(sl: slice):
            return self._fim_chunk_sum(
                {l: ctx.activations[l][sl] for l in layer_names},
                {l: ctx.gradients[l][sl] for l in layer_names},
                ctx.probs[sl],
            )

        fim_sum = _accumulate_chunks(N, self.batch_size, _chunk)
        fim = fim_sum / ctx.probs.sum()
        fim = 0.5 * (fim + fim.T)

        return LayerMatrix.from_dense(
            fim,
            param_groups=layer_names,
            layer_shapes=layer_shapes,
        )

    @staticmethod
    @jax.jit
    def _fim_chunk_sum(
        activations_dict: Dict[str, Float[Array, "n I"]],
        gradients_dict: Dict[str, Float[Array, "n O k"]],
        probs: Float[Array, "n k"],
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
        """
        Compute the Fisher-vector product (FVP) from pre-collected gradients.
        """
        gradients = self.precomputed_data.per_sample_grads  # (k, N, n_params)
        damping = 0.0 if damping is None else damping
        strategy = self.compute_context.pseudo_target_strategy

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            # For ALL_CLASSES, use special computation that averages over N only
            result = self._compute_fvp_all_classes(
                gradients=gradients,
                probabilities=self.precomputed_data.probabilities,
                damping=damping,
                vectors=vectors_2D,
            )
        else:
            # For EMPIRICAL_FISHER and MCMC, average over k*N
            result = self._compute_fvp_unweighted(
                gradients=gradients, vectors=vectors_2D, damping=damping
            )

        return result.squeeze(0) if is_single else result

    def _estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
        pseudo_inverse_factor: Optional[float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute the inverse Fisher-vector product (IFVP) using direct solve.
        """
        damping = 0.0 if damping is None else damping
        pseudo_inverse_factor = (
            0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
        )

        # Normalize to 2D: add batch dimension if needed
        is_single = vectors.ndim == 1
        vectors_2D: Float[Array, "batch_size n_params"] = (
            vectors[None, :] if is_single else vectors
        )

        # Compute FIM
        fim = self._estimate_hessian(damping)

        # Solve FIM^{-1} @ v
        result_2D = self._solve_fim_inv(fim, vectors_2D, pseudo_inverse_factor)

        return result_2D.squeeze(0) if is_single else result_2D

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
            eigvals, eigvecs = jnp.linalg.eigh(0.5 * (fim + fim.T))
            eigvals_inv = jnp.where(eigvals > pseudo_inverse_factor, 1.0 / eigvals, 0.0)
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
