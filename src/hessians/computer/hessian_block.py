from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

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

    def estimate_hessian(
        self,
        damping: Optional[Float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """
        Explicit block-diagonal Hessian matrix.
        """
        damping = 0.0 if damping is None else damping
        blocks = self._compute_blocks(damping)
        return jax.scipy.linalg.block_diag(*blocks)

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
        """
        damping = 0.0 if damping is None else damping
        is_single = vectors.ndim == 1
        vectors = vectors[None, :] if is_single else vectors

        results = []

        for block in self.blocks:
            start, end = block
            v_block = vectors[..., start:end]
            y_block = self._block_hvp(
                compute_context=self.compute_context,
                block=block,
                v_block=v_block,
                damping=damping,
            )
            results.append(y_block)

        out = jnp.concatenate(results, axis=-1)
        return out.squeeze(0) if is_single else out

    def estimate_ihvp(
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
        H_blocks = self._compute_blocks(damping)
        for i, (start, end) in enumerate(self.blocks):
            v_block = vectors[..., start:end]
            y_block = jnp.linalg.solve(H_blocks[i], v_block.T).T
            results.append(y_block)

        out = jnp.concatenate(results, axis=-1)
        return out.squeeze(0) if is_single else out

    @staticmethod
    @partial(jax.jit, static_argnames=["block"])
    def _block_hvp(
        compute_context: ModelContext,
        block: Tuple[int, int],
        v_block: Float[Array, "... d"],
        damping: Float,
    ) -> Float[Array, "... d"]:
        """
        Computes (E_i[H_ii^(i)] + λI) v for a single block,
        averaged over samples (x_i, y_i).
        """
        start, end = block

        X = compute_context.inputs
        Y = compute_context.targets
        assert Y is not None, (
            "Targets must be provided in ModelContext for HVP computation."
        )

        p_flat = compute_context.params_flat

        # Per-sample loss
        def loss_single(p, x, y):
            params_unflat = compute_context.unravel_fn(p)
            preds = compute_context.model_apply_fn(params_unflat, x[None, ...])
            return compute_context.loss_fn(preds.squeeze(0), y)

        grad_loss_single = jax.grad(loss_single)

        # Embed block vector into full parameter vector
        def embed(v):
            full = jnp.zeros((p_flat.shape[0],), dtype=v.dtype)
            return lax.dynamic_update_slice(full, v, (start,))

        # Per-sample HVP for a single v
        def hvp_single_sample(v, x, y):
            _, hv = jax.jvp(
                lambda p: grad_loss_single(p, x, y),
                (p_flat,),
                (embed(v),),
            )
            return lax.dynamic_slice(hv, (start,), (end - start,))

        # Average HVP over batch
        def hvp_avg(v):
            def scan_body(acc, xy):
                x_i, y_i = xy
                hv_i = hvp_single_sample(v, x_i, y_i)
                return acc + hv_i, None

            hv0 = jnp.zeros_like(v)
            hv_sum, _ = lax.scan(scan_body, hv0, (X, Y))
            hv_mean = hv_sum / X.shape[0]

            return hv_mean + damping * v

        # Support batched v_block
        if v_block.ndim == 1:
            return hvp_avg(v_block)
        else:
            return jax.vmap(hvp_avg)(v_block)

    def _compute_blocks(self, damping: Float):
        blocks = []
        for start, end in self.blocks:
            d = end - start
            eye = jnp.eye(d)
            H_block = self._block_hvp(
                compute_context=self.compute_context,
                block=(start, end),
                v_block=eye,
                damping=damping,
            )
            blocks.append(H_block)

        return blocks
