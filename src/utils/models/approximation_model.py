from __future__ import annotations

from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, List

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.tree_util import tree_flatten_with_path


@dataclass
class ApproximationModel(nn.Module):
    input_dim: int
    output_dim: int
    use_bias: bool = False
    seed: int = 42

    @classmethod
    def get_activation(cls, act_str: str) -> Callable:
        activations = {
            "relu": nn.relu,
            "tanh": nn.tanh,
        }
        if act_str not in activations:
            raise ValueError(f"Unknown activation: {act_str}")
        return activations[act_str]

    def get_layer_names(self) -> List[str]:
        """
        Get the names of all layers in the model.
        Avoids needing to initialize the model parameters and uses eval_shape instead.
        """
        shapes = jax.eval_shape(
            self.init, jax.random.PRNGKey(0), jnp.zeros((1, self.input_dim))
        )
        flatted_shapes, _ = tree_flatten_with_path(shapes["params"])
        return [path[0][0].key for path in flatted_shapes]

    @abstractmethod
    def collector_apply(self, x, collector) -> Any:
        """Special apply method which enables the custom forward and backward passes in order to collect information."""
        pass

    def serialize(self):
        return {
            "class": self.__class__.__name__,
            **self.to_dict_with_exluded_fields(["parent", "name"]),
        }

    def to_dict_with_exluded_fields(self, excluded_fields: list[str]) -> dict:
        """Convert the dataclass to a dictionary excluding specified fields."""
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in excluded_fields
        }
