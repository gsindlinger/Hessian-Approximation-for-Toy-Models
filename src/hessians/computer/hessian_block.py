from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import DenseBlock, LayerMatrix
from src.hessians.utils.data import (
    ModelContext,
    layer_shapes_from_model_context,
)


@dataclass
class BlockHessianComputer(HessianEstimator):
    compute_context: ModelContext
    """
    Block-Diagonal Hessian approximation.

    Each block corresponds to a *layer* (kernel + optional bias merged) of
    the model, matching the KFAC layer-dict convention.  `_build` materializes
    one `DenseBlock` per layer and wraps the collection as a block-diagonal
    `LayerMatrix`.  The lazy per-block HVP helper `_compute_blocks` is kept
    as the escape hatch a future big-model subclass can use by overriding
    `estimate_hvp`.
    """

    def _build(self) -> LayerMatrix:
        """Materialize each per-layer Hessian block and return a block-diagonal LayerMatrix."""
        ctx = self.compute_context
        layer_shapes = layer_shapes_from_model_context(ctx)
        layer_names: List[str] = list(ctx.model.get_layer_names())

        # layer_shapes_from_model_context already validated that each layer's
        # leaves are contiguous and that declared order matches flat layout,
        # so cumulative I*O offsets give the correct per-block ranges.
        block_ranges: List[Tuple[int, int]] = []
        idx = 0
        for name in layer_names:
            I, O = layer_shapes[name]
            block_ranges.append((idx, idx + I * O))
            idx += I * O

        block_arrays = self._compute_blocks(
            compute_context=ctx,
            blocks=tuple(block_ranges),
            damping=0.0,
        )
        diag_blocks: Dict[str, DenseBlock] = {
            name: DenseBlock(
                matrix=block_arrays[i],
                row_shape=layer_shapes[name],
                col_shape=layer_shapes[name],
            )
            for i, name in enumerate(layer_names)
        }
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks,
            param_groups=layer_names,
            layer_shapes=layer_shapes,
        )

    def get_layer_names(self) -> List[str]:
        return list(self.compute_context.model.get_layer_names())

    # ------------------------------------------------------------------
    # Block materialization helper (also retained as the lazy-HVP escape
    # hatch for a future big-model subclass)
    # ------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnames=["blocks"])
    def _compute_blocks(
        compute_context: ModelContext,
        blocks: Tuple[Tuple[int, int], ...],
        damping: Float,
    ):
        """
        Compute all blocks in parallel.
        Uses list comprehension with static block sizes for parallelization.
        """
        p_flat = compute_context.params_flat
        X = compute_context.inputs
        Y = compute_context.targets
        n_samples = X.shape[0]

        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), y)

        def compute_single_block_static(start: int, end: int):
            """Compute Hessian for a single block with static dimensions."""
            block_size = end - start
            I_block = jnp.eye(block_size, dtype=p_flat.dtype)

            def compute_block_hvps(identity_vectors):
                n_vectors = identity_vectors.shape[0]

                def body_fn(accum, xy):
                    x_i, y_i = xy

                    def grad_fn(p):
                        return jax.grad(lambda p_: loss_single(p_, x_i, y_i))(p)

                    def single_jvp(v):
                        full = jnp.zeros_like(p_flat)
                        full = lax.dynamic_update_slice(full, v, (start,))
                        _, hvp = jax.jvp(grad_fn, (p_flat,), (full,))
                        return lax.dynamic_slice(hvp, (start,), (block_size,))

                    hvps = jax.vmap(single_jvp)(identity_vectors)
                    return accum + hvps, None

                summed, _ = jax.lax.scan(
                    body_fn,
                    jnp.zeros((n_vectors, block_size), dtype=p_flat.dtype),
                    (X, Y),
                )
                return summed / n_samples

            n_params = p_flat.shape[0]
            if n_params < 1000:
                chunk_size = block_size
            elif n_params < 10000:
                chunk_size = min(2048, block_size)
            elif n_params < 100000:
                chunk_size = min(1024, block_size)
            else:
                chunk_size = min(512, block_size)

            if block_size <= chunk_size:
                H_block = compute_block_hvps(I_block)
            else:
                chunks = []
                for i in range(0, block_size, chunk_size):
                    chunk = I_block[i : min(i + chunk_size, block_size)]
                    chunk_result = compute_block_hvps(chunk)
                    chunks.append(chunk_result)
                H_block = jnp.concatenate(chunks, axis=0)

            H_block = H_block + damping * jnp.eye(block_size, dtype=p_flat.dtype)
            return H_block

        result_blocks = [
            compute_single_block_static(int(start), int(end))
            for start, end in blocks
        ]
        return result_blocks
