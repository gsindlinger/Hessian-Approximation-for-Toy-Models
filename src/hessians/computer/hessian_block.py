from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_leaves
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.utils.data import ModelContext
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class BlockHessianComputer(HessianEstimator):
    """
    Block-Diagonal Hessian approximation.
    Optimized version using full Hessian computation with masking.
    """

    compute_context: ModelContext

    def __post_init__(self):
        test = self.compute_context.unravel_fn(self.compute_context.params_flat)
        leaves = tree_leaves(test)

        # Build flat index ranges
        blocks = []
        idx = 0
        for leaf in leaves:
            size = leaf.size
            blocks.append((idx, idx + size))
            idx += size

        self.blocks = blocks
        self.n_params = idx

        # Precompute block mask for efficient extraction
        self.block_mask = self._create_block_mask()

    def _create_block_mask(self) -> Float[Array, "n_params n_params"]:
        """Create a binary mask for block-diagonal structure."""
        mask = jnp.zeros((self.n_params, self.n_params), dtype=bool)
        for start, end in self.blocks:
            mask = mask.at[start:end, start:end].set(True)
        return mask

    def estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """
        Explicit block-diagonal Hessian matrix.
        Uses full Hessian computation and masks to block-diagonal.
        """
        damping = 0.0 if damping is None else damping

        # Compute full Hessian (same as HessianComputer)
        H_full = self._compute_full_hessian(
            compute_context=self.compute_context,
            damping=damping,
        )

        # Zero out off-block-diagonal entries
        H_block = jnp.where(self.block_mask, H_full, 0.0)
        assert isinstance(H_block, jnp.ndarray)
        return H_block

    def compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare block-diagonal Hessian against a full Hessian.
        """
        damping = 0.0 if damping is None else damping
        H_block = self.estimate_hessian(damping)
        return metric.compute_fn()(comparison_matrix, H_block)

    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Block-diagonal Hessian-vector product.
        More efficient than old implementation.
        """
        damping = 0.0 if damping is None else damping
        is_single = vectors.ndim == 1
        vectors = vectors[None, :] if is_single else vectors

        # Apply block structure by computing block-wise
        results = []
        for start, end in self.blocks:
            # For block-diagonal, HVP only depends on corresponding block of vector
            v_block = vectors[..., start:end]
            hvp_block = self._compute_block_hvp_fast(start, end, v_block, damping)
            results.append(hvp_block)

        out = jnp.concatenate(results, axis=-1)
        return out.squeeze(0) if is_single else out

    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Inverse block-diagonal Hessian-vector product.
        Solves each block independently.
        """
        damping = 0.0 if damping is None else damping
        is_single = vectors.ndim == 1
        vectors = vectors[None, :] if is_single else vectors

        # Get block-diagonal Hessian
        H_block = self.estimate_hessian(damping)

        results = []
        for start, end in self.blocks:
            # Extract block
            H_i = H_block[start:end, start:end]
            v_block = vectors[..., start:end]

            # Solve for this block
            y_block = jnp.linalg.solve(H_i, v_block.T).T
            results.append(y_block)

        out = jnp.concatenate(results, axis=-1)
        return out.squeeze(0) if is_single else out

    def _compute_block_hvp_fast(
        self,
        start: int,
        end: int,
        v_block: Float[Array, "... d"],
        damping: Float,
    ) -> Float[Array, "... d"]:
        """
        Fast block HVP using the scan-based approach from HessianComputer.
        """
        return self._compute_block_hvp_fast_static(
            compute_context=self.compute_context,
            start=start,
            end=end,
            v_block=v_block,
            damping=damping,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["start", "end"])
    def _compute_block_hvp_fast_static(
        compute_context: ModelContext,
        start: int,
        end: int,
        v_block: Float[Array, "... d"],
        damping: Float,
    ) -> Float[Array, "... d"]:
        """
        Static version for JIT compilation.
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), y)

        # Embed block vector into full parameter space
        def embed(v):
            full = jnp.zeros_like(p_flat)
            return lax.dynamic_update_slice(full, v, (start,))

        # Extract block from full vector
        def extract(full):
            return lax.dynamic_slice(full, (start,), (end - start,))

        @jax.jit
        def hvp_single(v):
            """Compute H @ v for a single vector"""
            v_full = embed(v)

            def body_fn(accum, xy):
                x_i, y_i = xy

                def grad_fn(p):
                    return jax.grad(lambda p_: loss_single(p_, x_i, y_i))(p)

                _, hvp_i = jax.jvp(grad_fn, (p_flat,), (v_full,))
                return accum + extract(hvp_i), None

            summed, _ = jax.lax.scan(body_fn, jnp.zeros_like(v), (X, Y))
            hvp = summed / X.shape[0]
            return hvp + damping * v

        # Handle batching with chunking
        if v_block.ndim == 1:
            return hvp_single(v_block)

        CHUNK_SIZE = 32
        n_vectors = v_block.shape[0]

        if n_vectors <= CHUNK_SIZE:
            return jax.vmap(hvp_single)(v_block)
        else:
            outs = []
            for i in range(0, n_vectors, CHUNK_SIZE):
                chunk = v_block[i : i + CHUNK_SIZE]
                outs.append(jax.vmap(hvp_single)(chunk))
            return jnp.concatenate(outs, axis=0)

    @staticmethod
    @jax.jit
    def _compute_full_hessian(
        compute_context: ModelContext,
        damping: Float,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian (reuses HessianComputer logic).
        """

        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), y)

        @jax.jit
        def compute_sample_hessian(p_flat, x, y):
            return jax.hessian(lambda p: loss_single(p, x, y))(p_flat)

        def scan_body(carry, xy):
            p_flat, H = carry
            x_i, y_i = xy
            H_i = compute_sample_hessian(p_flat, x_i, y_i)
            return (p_flat, H + H_i), None

        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        H0 = jnp.zeros((p_flat.size, p_flat.size))
        (_, H_full), _ = jax.lax.scan(scan_body, init=(p_flat, H0), xs=(X, Y))

        H_full = H_full / X.shape[0]
        H_full = 0.5 * (H_full + H_full.T)

        return H_full + damping * jnp.eye(H_full.shape[0])

    @staticmethod
    @jax.jit
    def _compute_hvp(
        compute_context: ModelContext,
        vectors: Float[Array, "batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "batch_size n_params"]:
        """
        Full HVP computation (reuses HessianComputer logic).
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        def loss_single(p, x, y):
            params = compute_context.unravel_fn(p)
            z = compute_context.model_apply_fn(params, x[None, ...]).squeeze(0)
            return compute_context.loss_fn(z, y)

        @jax.jit
        def hvp_single(v):
            def body_fn(accum, xy):
                x_i, y_i = xy

                def grad_fn(p):
                    return jax.grad(lambda p_: loss_single(p_, x_i, y_i))(p)

                _, hvp_i = jax.jvp(grad_fn, (p_flat,), (v,))
                return accum + hvp_i, None

            summed, _ = jax.lax.scan(body_fn, jnp.zeros_like(v), (X, Y))
            hvp = summed / X.shape[0]
            return hvp + damping * v

        CHUNK_SIZE = 32
        n_vectors = vectors.shape[0]

        if n_vectors <= CHUNK_SIZE:
            return jax.vmap(hvp_single)(vectors)
        else:
            outs = []
            for i in range(0, n_vectors, CHUNK_SIZE):
                chunk = vectors[i : i + CHUNK_SIZE]
                outs.append(jax.vmap(hvp_single)(chunk))
            return jnp.concatenate(outs, axis=0)
