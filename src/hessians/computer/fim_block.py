from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.config import PseudoTargetGenerationStrategy
from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import DenseBlock, LayerMatrix
from src.hessians.utils.data import DataActivationsGradients


@dataclass
class FIMBlockComputer(HessianEstimator):
    compute_context: DataActivationsGradients
    """
    Block-diagonal Fisher Information Matrix approximation.

    Computes F = block_diag(F_1, ..., F_L) where each F_l is the FIM for layer l.

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
        """Build a block-diagonal `LayerMatrix` of per-layer dense FIM blocks."""
        strategy = compute_context.pseudo_target_strategy
        layer_names = list(compute_context.layer_names)
        shapes = self._layer_shapes_from_context(compute_context)

        diag_blocks: Dict[str, DenseBlock] = {}
        for layer in layer_names:
            act = compute_context.activations[layer]
            grad = compute_context.gradients[layer]
            if strategy == PseudoTargetGenerationStrategy.ALL_CLASSES:
                block = self._compute_layer_block_weighted(
                    act, grad, compute_context.probabilities
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
            param_groups=layer_names,
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
