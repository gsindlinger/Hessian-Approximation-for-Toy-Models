from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    name: Literal["linear", "mlp"]
    loss: Literal["mse", "cross_entropy"]


@dataclass
class LinearModelConfig(ModelConfig):
    name: str = field(default="linear", init=False)
    loss: str
    hidden_dim: List[int] = field(default_factory=list)
    use_bias: bool = field(default=False)


@dataclass
class MLPModelConfig(ModelConfig):
    name: str = field(default="mlp", init=False)
    loss: str
    hidden_dim: List[int] = field(default_factory=list)
    use_bias: bool = field(default=False)
