from __future__ import annotations

from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, List

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.tree_util import tree_flatten_with_path

from src.config import ModelArchitecture, ModelConfig
from src.utils.models.linear_model import LinearModel
from src.utils.models.mlp import MLP
from src.utils.models.mlp_swiglu import MLPSwiGLU


@dataclass
class ApproximationModel(nn.Module):
    input_dim: int
    output_dim: int
    seed: int = 42

    @staticmethod
    def get_model(
        model_config: ModelConfig, input_dim: int, output_dim: int, seed: int = 42
    ) -> ApproximationModel:
        model_cls = MODEL_REGISTRY[model_config.architecture]
        hidden_layers_dict = asdict(model_config).get("hidden_dims", {})
        return model_cls(
            input_dim=input_dim, output_dim=output_dim, seed=seed, **hidden_layers_dict
        )

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
        Get the names of all unique layers in the model.
        Avoids needing to initialize the model parameters and uses eval_shape instead.

        If a model has multiple parts of a single layer (e.g., kernel and bias),
        only the layer name is returned once.
        """
        shapes = jax.eval_shape(
            self.init, jax.random.PRNGKey(0), jnp.zeros((1, self.input_dim))
        )
        flatted_shapes, _ = tree_flatten_with_path(shapes["params"])

        # ensuring unique layer names by using a set
        seen_names = set()
        layer_names = []
        for path, _ in flatted_shapes:
            name = "/".join(key.key for key in path[:-1])
            if name not in seen_names:
                seen_names.add(name)
                layer_names.append(name)
        return layer_names

    @property
    def num_params(self) -> int:
        """Count the total number of parameters in the model."""
        shapes = jax.eval_shape(
            self.init, jax.random.PRNGKey(0), jnp.zeros((1, self.input_dim))
        )
        flattened_shapes, _ = tree_flatten_with_path(shapes["params"])
        total_params = 0
        for _, leaf in flattened_shapes:
            total_params += int(jnp.prod(jnp.array(leaf.shape)))

        return total_params

    @abstractmethod
    def collector_apply(self, x, collector) -> Any:
        """Special apply method which enables the custom forward and backward passes in order to collect information."""
        pass

    def serialize(self):
        return {
            "class": self.__class__.__name__,
            **self.to_dict_with_excluded_fields(["parent", "name"]),
        }

    def to_dict_with_excluded_fields(self, excluded_fields: list[str]) -> dict:
        """Convert the dataclass to a dictionary excluding specified fields."""
        test_dict = {
            key: value
            for key, value in asdict(self).items()
            if key not in excluded_fields
        }

        return test_dict


MODEL_REGISTRY: dict[ModelArchitecture, type[ApproximationModel]] = {
    ModelArchitecture.MLP: MLP,
    ModelArchitecture.MLPSWIGLU: MLPSwiGLU,
    ModelArchitecture.LINEAR: LinearModel,
}
