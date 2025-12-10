from __future__ import annotations

from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable

from flax import linen as nn


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

    @abstractmethod
    def collector_apply(self, x, collector) -> Any:
        """Special apply method which enables the custom forward and backward passes in order to collect information."""
        pass

    def get_num_params(self, params: dict) -> int:
        """Get total number of parameters in the model."""
        from jax.flatten_util import ravel_pytree

        flat_params, _ = ravel_pytree(params)
        return flat_params.shape[0]

    def serialize(self):
        return {"class": self.__class__.__name__, **asdict(self)}
