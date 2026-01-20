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
        Single dataset pass for all blocks.
        """
        damping = 0.0 if damping is None else damping
        blocks = self.compute_blocks(damping)
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
        """
        damping = 0.0 if damping is None else damping
        is_single = vectors.ndim == 1
        vectors = vectors[None, :] if is_single else vectors

        results = []
        for block in self.precomputed_data.blocks:
            start, end = block
            start = int(start)
            end = int(end)
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
        """
        damping = 0.0 if damping is None else damping
        is_single = vectors.ndim == 1
        vectors = vectors[None, :] if is_single else vectors

        results = []
        H_blocks = self.compute_blocks(damping)
        for i, (start, end) in enumerate(self.precomputed_data.blocks):
            start = int(start)
            end = int(end)
            v_block = vectors[..., start:end]
            y_block = jnp.linalg.solve(H_blocks[i], v_block.T).T
            results.append(y_block)

        out = jnp.concatenate(results, axis=-1)
        return out.squeeze(0) if is_single else out

    def compute_blocks(self, damping: Float):
        """
        Creates identity matrices for all blocks, concatenates them,
        then does one scan over the data computing all HVPs simultaneously.
        """
        blocks_tuple = tuple(self.precomputed_data.blocks)
        return self._compute_blocks(
            compute_context=self.compute_context,
            blocks=blocks_tuple,
            damping=damping,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["blocks"])
    def _compute_blocks(
        compute_context: ModelContext,
        blocks: tuple,
        damping: Float,
    ):
        """
        Compute all blocks in one dataset pass by batching all identity vectors.
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets
        n_samples = X.shape[0]

        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), y)

        # Create all identity vectors for all blocks at once
        all_identity_vectors = []
        block_sizes = []

        for start, end in blocks:
            start = int(start)
            end = int(end)
            block_size = end - start
            block_sizes.append(block_size)

            # Identity matrix for this block
            I_block = jnp.eye(block_size)

            # Embed each identity vector into full parameter space
            for i in range(block_size):
                full = jnp.zeros_like(p_flat)
                full = lax.dynamic_update_slice(full, I_block[i], (start,))
                all_identity_vectors.append(full)

        # Stack all identity vectors: shape (total_params, n_params)
        all_vectors = jnp.stack(all_identity_vectors)
        total_vectors = all_vectors.shape[0]

        # The key: Single scan over dataset computing all HVPs at once
        def compute_all_hvps(vectors):
            """Compute H @ vectors for all vectors in one dataset pass"""

            def body_fn(accum, xy):
                x_i, y_i = xy

                # Compute HVP for all vectors simultaneously
                def grad_fn(p):
                    return jax.grad(lambda p_: loss_single(p_, x_i, y_i))(p)

                # vmap over all vectors to compute all HVPs at once
                def single_jvp(v):
                    _, hvp = jax.jvp(grad_fn, (p_flat,), (v,))
                    return hvp

                hvps = jax.vmap(single_jvp)(vectors)
                return accum + hvps, None

            summed, _ = jax.lax.scan(body_fn, jnp.zeros_like(vectors), (X, Y))
            return summed / n_samples

        # Compute in chunks to avoid OOM
        CHUNK_SIZE = 256  # Adjust based on memory
        all_hvps = []

        for i in range(0, total_vectors, CHUNK_SIZE):
            chunk = all_vectors[i : min(i + CHUNK_SIZE, total_vectors)]
            hvps_chunk = compute_all_hvps(chunk)
            all_hvps.append(hvps_chunk)

        all_hvps = jnp.concatenate(all_hvps, axis=0)

        # Extract blocks from the concatenated HVPs
        result_blocks = []
        idx = 0

        for block_idx, (start, end) in enumerate(blocks):
            start = int(start)
            end = int(end)
            block_size = block_sizes[block_idx]

            # Extract the HVPs for this block
            block_hvps = all_hvps[idx : idx + block_size]

            # Extract only the block-diagonal portion
            H_block = block_hvps[:, start:end]

            # Add damping
            H_block = H_block + damping * jnp.eye(block_size)
            result_blocks.append(H_block)

            idx += block_size

        return result_blocks

    def compute_block_hvp(
        self,
        start: int,
        end: int,
        v_block: Float[Array, "... d"],
        damping: Float,
    ) -> Float[Array, "... d"]:
        """
        Fast block HVP using JVP-based approach.
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
        Fast block HVP using scan-based approach with JVP.
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets

        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), y)

        def embed(v):
            full = jnp.zeros_like(p_flat)
            return lax.dynamic_update_slice(full, v, (start,))

        def extract(full):
            return lax.dynamic_slice(full, (start,), (end - start,))

        @jax.jit
        def hvp_single(v):
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
