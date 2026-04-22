from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import jax.numpy as jnp

from src.hessians.computer.computer import HessianEstimator
from src.hessians.layer_matrix import DenseBlock, LayerMatrix
from src.hessians.utils.data import ModelContext, layer_shapes_from_model_context


@dataclass
class IdentityComputer(HessianEstimator):
    """Identity approximation: block-diagonal identity matrix matching the
    per-layer layout used by the other estimators."""

    compute_context: ModelContext

    def get_layer_names(self) -> List[str]:
        return list(self.compute_context.model.get_layer_names())

    def _build(self) -> LayerMatrix:
        layer_shapes = layer_shapes_from_model_context(self.compute_context)
        layer_names = self.get_layer_names()
        dtype = self.compute_context.params_flat.dtype
        diag_blocks: Dict[str, DenseBlock] = {
            name: DenseBlock(
                matrix=jnp.eye(
                    layer_shapes[name][0] * layer_shapes[name][1], dtype=dtype
                ),
                row_shape=layer_shapes[name],
                col_shape=layer_shapes[name],
            )
            for name in layer_names
        }
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks,
            param_groups=layer_names,
            layer_shapes=layer_shapes,
        )
