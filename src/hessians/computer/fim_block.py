from functools import partial
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy
from src.hessians.computer.computer import CollectorBasedHessianEstimator
from src.hessians.layer_matrix import DenseBlock, LayerMatrix, LayerVector
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
        """Build a block-diagonal `LayerMatrix` of per-layer dense FIM blocks."""
        strategy = self.compute_context.pseudo_target_strategy
        shapes = self._layer_shapes()

        diag_blocks: Dict[str, DenseBlock] = {}
        for layer in self.get_layer_names():
            act = self.compute_context.activations[layer]
            grad = self.compute_context.gradients[layer]
            if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
                block = self._compute_layer_block_weighted(
                    act, grad, self.compute_context.probabilities
                )
            else:
                block = self._compute_layer_block_unweighted(act, grad)
            diag_blocks[layer] = DenseBlock(
                matrix=block,
                row_shape=shapes[layer],
                col_shape=shapes[layer],
            )
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks,
            param_groups=self.get_layer_names(),
            layer_shapes=shapes,
        )

    @staticmethod
    @jax.jit
    def _compute_layer_block_unweighted(
        act: Float[Array, "N I"],
        grad: Float[Array, "K N O"],
    ) -> Float[Array, "n n"]:
        """Single-layer FIM block for EMPIRICAL_FISHER / MCMC strategies."""
        N, I = act.shape
        K, _, O = grad.shape
        act_expanded = jnp.broadcast_to(act[None, :, :], (K, N, I))
        per_sample_vecs = jnp.einsum(
            "kni,kno->knio", act_expanded, grad
        ).reshape(K, N, -1)
        vecs_flat = per_sample_vecs.reshape(K * N, -1)
        return (vecs_flat.T @ vecs_flat) / (K * N)

    @staticmethod
    @jax.jit
    def _compute_layer_block_weighted(
        act: Float[Array, "N I"],
        grad: Float[Array, "K N O"],
        probabilities: Float[Array, "N K"],
    ) -> Float[Array, "n n"]:
        """Single-layer FIM block for ALL_CLASSES strategy."""
        N, I = act.shape
        K, _, _ = grad.shape
        act_expanded = jnp.broadcast_to(act[None, :, :], (K, N, I))
        per_sample_vecs = jnp.einsum(
            "kni,kno->knio", act_expanded, grad
        ).reshape(K, N, -1)
        sqrt_probs = jnp.sqrt(probabilities.T)[..., None]  # (K, N, 1)
        weighted_vecs = per_sample_vecs * sqrt_probs
        weighted_vecs_flat = weighted_vecs.reshape(K * N, -1)
        return (weighted_vecs_flat.T @ weighted_vecs_flat) / N

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
        fim = self._estimate_hessian(d)
        return metric.compute_fn()(comparison_matrix, fim)

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
