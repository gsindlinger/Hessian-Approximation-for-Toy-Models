from __future__ import annotations

from dataclasses import field

from flax import linen as nn

from models.train import ApproximationModel


class LinearModel(ApproximationModel):
    """Linear model with optional hidden layers."""

    hidden_dim: list[int] = field(default_factory=list)

    def _get_layer_config(self):
        """Returns list of (name, dim, use_bias) tuples for all layers."""
        config = []
        for i, h in enumerate(self.hidden_dim):
            config.append((f"linear_{i}", h, self.use_bias))
        config.append(("output", self.output_dim, self.use_bias))
        return config

    @nn.compact
    def __call__(self, x):
        config = self._get_layer_config()
        for name, dim, use_bias in config:
            x = nn.Dense(dim, use_bias=use_bias, name=name)(x)
        return x
