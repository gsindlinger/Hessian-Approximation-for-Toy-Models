from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy
from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import LayerMatrix
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class FIMComputer(HessianEstimator):
    compute_context: DataActivationsGradients
    """
    Fisher Information Matrix approximation.

    The Fisher Information Matrix is defined as:
    FIM = E[∇log p(y|x) ∇log p(y|x)^T]

    Uses previously collected gradients to compute the FIM as the outer
    product of per-sample parameter gradients, then slices the materialized
    `(n_params, n_params)` matrix into per-layer `DenseBlock`s via
    `LayerMatrix.from_dense`.

    Supports different pseudo-target strategies:
    - EMPIRICAL_FISHER: Uses ground truth labels (k=1)
    - MCMC: Uses sampled pseudo-targets (k=num_samples)
    - ALL_CLASSES: Uses all classes with probability weighting (k=num_classes)
    """

    def get_layer_names(self) -> List[str]:
        return self.compute_context.layer_names

    def _layer_shapes_from_context(
        self, compute_context: DataActivationsGradients
    ) -> Dict[str, Tuple[int, int]]:
        return {
            l: (
                int(compute_context.activations[l].shape[-1]),
                int(compute_context.gradients[l].shape[-1]),
            )
            for l in compute_context.layer_names
        }

    def _build(self, compute_context: DataActivationsGradients) -> LayerMatrix:
        """Assemble per-sample parameter gradients, materialize the FIM, and slice."""
        strategy = compute_context.pseudo_target_strategy
        layer_names = list(compute_context.layer_names)
        layer_shapes = self._layer_shapes_from_context(compute_context)

        n_samples = compute_context.activations[layer_names[0]].shape[0]
        grads_per_layer = []
        for layer in layer_names:
            a = compute_context.activations[layer]  # (N, I_l)
            g = compute_context.gradients[layer]  # (K, N, O_l)
            k = g.shape[0]
            a_expanded = jnp.broadcast_to(a[None, :, :], (k, n_samples, a.shape[-1]))
            G_l = jnp.einsum("kni,kno->knio", a_expanded, g).reshape(
                k, n_samples, -1
            )
            grads_per_layer.append(G_l)
        grads_all = jnp.concatenate(grads_per_layer, axis=2)

        if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
            dense = self._compute_fim_all_classes(
                gradients=grads_all,
                probabilities=compute_context.probabilities,
                damping=0.0,
            )
        else:
            dense = self._compute_fim_unweighted(grads_all, 0.0)

        return LayerMatrix.from_dense(
            dense,
            param_groups=layer_names,
            layer_shapes=layer_shapes,
        )

    # ------------------------------------------------------------------
    # Materialization helpers (used by _build)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Lazy HVP escape hatches (retained for future big-model overrides;
    # not used by the refactored `estimate_hvp` path).
    # ------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnames=["pseudo_inverse_factor"])
    def _solve_fim_inv(
        fim: Float[Array, "n_params n_params"],
        vectors: Float[Array, "batch_size n_params"],
        pseudo_inverse_factor: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Solve FIM^{-1} @ v for each vector v using direct linear solve.

        Currently unused — retained as the lazy IHVP escape hatch.
        """
        if pseudo_inverse_factor > 0.0:
            # eigh needs float64 for a near-rank-deficient FIM; cast back to
            # input dtype after the decomposition.
            orig_dtype = fim.dtype
            sym = (0.5 * (fim + fim.T)).astype(jnp.float64)
            eigvals, eigvecs = jnp.linalg.eigh(sym)
            eigvals = eigvals.astype(orig_dtype)
            eigvecs = eigvecs.astype(orig_dtype)
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

        Currently unused — retained as the lazy HVP escape hatch for a
        future big-model subclass that overrides `estimate_hvp` to bypass
        `LayerMatrix`.
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

        Currently unused — retained as the lazy HVP escape hatch.
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
