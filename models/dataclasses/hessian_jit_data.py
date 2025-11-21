from __future__ import annotations

from typing import Callable

import flax.struct as struct
from jax import flatten_util
from jaxtyping import Array, Float

from models.dataclasses.model_data import ModelData


@struct.dataclass
class HessianJITData:
    """Container for the model, dataset, parameters, and loss function."""

    training_data: Float[Array, "..."]
    training_targets: Float[Array, "..."]
    params_flat: Float[Array, "..."]
    unravel_fn: Callable = struct.field(pytree_node=False)
    model_apply_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)

    @staticmethod
    def get_data_and_params_for_hessian(
        model_data: ModelData,
    ) -> HessianJITData:
        # Important: Flattening structure for linear modules with bias is the following: b, w
        # So for output dim 2, input dim 3, the order is: b1, b2, w1
        training_data, training_targets = model_data.dataset.get_train_data()
        params_flat, unravel_fn = flatten_util.ravel_pytree(model_data.params)
        return HessianJITData(
            training_data=training_data,
            training_targets=training_targets,
            params_flat=params_flat,
            unravel_fn=unravel_fn,
            model_apply_fn=model_data.model.apply,
            loss_fn=model_data.loss,
        )

    @property
    def num_params(self) -> int:
        return self.params_flat.shape[0]
