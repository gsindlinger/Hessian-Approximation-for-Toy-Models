from dataclasses import dataclass
from typing import Dict, Tuple

from jax import numpy as jnp


@dataclass
class LayerComponents:
    """Container for layer-wise activations and gradients."""

    activations: Dict[str, jnp.ndarray]
    gradients: Dict[str, jnp.ndarray]

    def __init__(self, activations=None, gradients=None):
        self.activations = activations if activations is not None else {}
        self.gradients = gradients if gradients is not None else {}

    def __bool__(self):
        return bool(self.activations) and bool(self.gradients)

    def items(self):
        for layer_name in self.activations.keys():
            yield layer_name, (self.activations[layer_name], self.gradients[layer_name])

    def keys(self):
        return self.activations.keys()

    def values(self):
        for layer_name in self.activations.keys():
            yield (self.activations[layer_name], self.gradients[layer_name])

    def __getitem__(self, layer_name: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.activations[layer_name], self.gradients[layer_name]

    def __setitem__(
        self, layer_name: str, data: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> None:
        self.activations[layer_name] = data[0]
        self.gradients[layer_name] = data[1]

    def __len__(self):
        return len(self.activations)
