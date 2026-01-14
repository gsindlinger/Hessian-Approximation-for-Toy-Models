from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_leaves
from jaxtyping import Array, Float

from src.hessians.computer.computer import ModelBasedHessianEstimator
from src.hessians.utils.data import BlockHessianData, ModelContext
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class BlockHessianComputer(ModelBasedHessianEstimator):
    """
    Block-Diagonal Hessian approximation.
    Each block corresponds to a parameter leaf in the model.
    """

    precomputed_data: BlockHessianData = field(default_factory=BlockHessianData)

    @staticmethod
    def _build(compute_context: ModelContext) -> BlockHessianData:
        test = compute_context.unravel_fn(compute_context.params_flat)
        leaves = tree_leaves(test)

        # Build flat index ranges
        blocks = []
        idx = 0
        for leaf in leaves:
            size = leaf.size
            blocks.append((idx, idx + size))
            idx += size

        return BlockHessianData(blocks=blocks, n_params=idx)

    def _estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """
        Explicit block-diagonal Hessian matrix.
        Uses full Hessian computation and masks to block-diagonal.
        """
        damping = 0.0 if damping is None else damping
        blocks = self._compute_blocks(damping)
        return jax.scipy.linalg.block_diag(*blocks)

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """
        Compare block-diagonal Hessian against a full Hessian.
        """
        damping = 0.0 if damping is None else damping
        H_block = self._estimate_hessian(damping)
        return metric.compute_fn()(comparison_matrix, H_block)

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Block-diagonal Hessian-vector product.
        Computes each block independently.
        """
        damping = 0.0 if damping is None else damping
        is_single = vectors.ndim == 1
        vectors = vectors[None, :] if is_single else vectors

        # Apply block structure by computing block-wise
        results = []

        for block in self.precomputed_data.blocks:
            start, end = block
            v_block = vectors[..., start:end]
            hvp_block = self.compute_block_hvp(start, end, v_block, damping)
            results.append(hvp_block)

        out = jnp.concatenate(results, axis=-1)
        return out.squeeze(0) if is_single else out

    def _estimate_ihvp(
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

        results = []
        H_blocks = self._compute_blocks(damping)
        for i, (start, end) in enumerate(self.precomputed_data.blocks):
            v_block = vectors[..., start:end]

            # Solve for this block
            y_block = jnp.linalg.solve(H_blocks[i], v_block.T).T
            results.append(y_block)

        out = jnp.concatenate(results, axis=-1)
        return out.squeeze(0) if is_single else out

    def compute_block_hvp(
        self,
        start: int,
        end: int,
        v_block: Float[Array, "... d"],
        damping: Float,
    ) -> Float[Array, "... d"]:
        """
        Fast block HVP using scan-based approach.
        """
        return self._compute_block_hvp(
            compute_context=self.compute_context,
            start=start,
            end=end,
            v_block=v_block,
            damping=damping,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["start", "end"])
    def _compute_block_hvp(
        compute_context: ModelContext,
        start: int,
        end: int,
        v_block: Float[Array, "... d"],
        damping: Float,
    ) -> Float[Array, "... d"]:
        """
        Fast block HVP using scan-based approach.
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

    def _compute_blocks(self, damping: Float):
        blocks = []
        for start, end in self.precomputed_data.blocks:
            d = end - start
            eye = jnp.eye(d)
            H_block = self.compute_block_hvp(
                start=start,
                end=end,
                v_block=eye,
                damping=damping,
            )
            blocks.append(H_block)

        return blocks
