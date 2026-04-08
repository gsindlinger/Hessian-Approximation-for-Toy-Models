from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_flatten_with_path
from jaxtyping import Array, Float

from src.hessians.computer.computer import ModelBasedHessianEstimator
from src.hessians.layer_matrix import DenseBlock, LayerMatrix, LayerVector
from src.hessians.utils.data import (
    BlockHessianData,
    ModelContext,
    layer_shapes_from_model_context,
)
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric


@dataclass
class BlockHessianComputer(ModelBasedHessianEstimator):
    """
    Block-Diagonal Hessian approximation.

    Each block corresponds to a *layer* (kernel + optional bias merged) of
    the model, matching the KFAC layer-dict convention.
    """

    precomputed_data: BlockHessianData = field(default_factory=BlockHessianData)

    @staticmethod
    def _build(compute_context: ModelContext) -> BlockHessianData:
        """Compute per-layer flat-index ranges and shapes from the Flax param tree."""
        layer_shapes = layer_shapes_from_model_context(compute_context)
        layer_names = compute_context.model.get_layer_names()

        # Verify each layer's leaves are contiguous in params_flat by walking
        # the flat tree leaves and grouping by layer.
        params = compute_context.unravel_fn(compute_context.params_flat)
        params_root = params["params"] if "params" in params else params
        leaves_with_paths, _ = tree_flatten_with_path(params_root)

        leaf_layer_of: List[str] = []
        leaf_sizes: List[int] = []
        for path, leaf in leaves_with_paths:
            layer = "/".join(k.key for k in path[:-1])
            leaf_layer_of.append(layer)
            leaf_sizes.append(int(leaf.size))

        blocks: List[Tuple[int, int]] = []
        idx = 0
        for layer in layer_names:
            start = idx
            for leaf_layer, size in zip(leaf_layer_of, leaf_sizes):
                if leaf_layer == layer:
                    idx += size
            end = idx
            blocks.append((start, end))

        return BlockHessianData(
            blocks=blocks,
            layer_names=list(layer_names),
            layer_shapes=layer_shapes,
            n_params=int(compute_context.params_flat.size),
        )

    # ------------------------------------------------------------------
    # LayerMatrix construction
    # ------------------------------------------------------------------

    def get_layer_names(self) -> List[str]:
        return list(self.precomputed_data.layer_names)

    def _layer_shapes(self) -> Dict[str, Tuple[int, int]]:
        # Re-derive from compute_context (cheap; avoids load-time pytree-roundtrip
        # quirks for Dict[str, Tuple[int, int]] persistence).
        return layer_shapes_from_model_context(self.compute_context)

    def _get_layer_matrix(self) -> LayerMatrix:
        """Materialize each per-layer Hessian block and wrap as `LayerMatrix.block_diagonal`."""
        layer_names = self.get_layer_names()
        shapes = self._layer_shapes()
        block_ranges = tuple(
            (int(s), int(e)) for (s, e) in self.precomputed_data.blocks
        )
        block_arrays = self._compute_blocks(
            compute_context=self.compute_context,
            blocks=block_ranges,
            damping=0.0,
        )
        diag_blocks: Dict[str, DenseBlock] = {
            name: DenseBlock(
                matrix=block_arrays[i],
                row_shape=shapes[name],
                col_shape=shapes[name],
            )
            for i, name in enumerate(layer_names)
        }
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks,
            param_groups=layer_names,
            layer_shapes=shapes,
        )

    # ------------------------------------------------------------------
    # HessianEstimator interface (thin wrappers over LayerMatrix)
    # ------------------------------------------------------------------

    def _estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        d = 0.0 if damping is None else damping
        return self._get_layer_matrix().damped(d).to_dense()

    def _compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        d = 0.0 if damping is None else damping
        H_block = self._estimate_hessian(d)
        return metric.compute_fn()(comparison_matrix, H_block)

    def _estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
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

    # ------------------------------------------------------------------
    # Block materialization helper (unchanged from prior implementation)
    # ------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnames=["blocks"])
    def _compute_blocks(
        compute_context: ModelContext,
        blocks: Tuple[Tuple[int, int]],
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
            I_block = jnp.eye(block_size)

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
                    jnp.zeros((n_vectors, block_size)),
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

            H_block = H_block + damping * jnp.eye(block_size)
            return H_block

        result_blocks = [
            compute_single_block_static(int(start), int(end))
            for start, end in blocks
        ]
        return result_blocks
