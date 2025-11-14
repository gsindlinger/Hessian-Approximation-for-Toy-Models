from __future__ import annotations

from abc import abstractmethod
from typing import Any

from flax import linen as nn


class ApproximationModel(nn.Module):
    input_dim: int
    output_dim: int
    use_bias: bool = False

    @staticmethod
    def get_activation(act_str: str):
        activations = {
            "relu": nn.relu,
            "tanh": nn.tanh,
        }
        if act_str not in activations:
            raise ValueError(f"Unknown activation: {act_str}")
        return activations[act_str]

    @abstractmethod
    def kfac_apply(self, x, collector) -> Any:
        """Special apply method for K-FAC that wraps layers by custom VJP."""
        pass

    def get_num_params(self, params: dict) -> int:
        """Get total number of parameters in the model."""
        from jax.flatten_util import ravel_pytree

        _, unravel_fn = ravel_pytree(params)
        flat_params, _ = ravel_pytree(params)
        return flat_params.shape[0]
