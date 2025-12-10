from __future__ import annotations

from dataclasses import dataclass, field

from flax import linen as nn

from deleuze.hessians.collector import layer_wrapper_vjp

from .approximation_model import ApproximationModel


@dataclass
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

    @nn.compact
    def collector_apply(self, x, collector):
        """A special apply method for K-FAC that wraps layers by custom VJP."""
        activations = x
        config = self._get_layer_config()

        for name, dim, use_bias in config:
            layer_module = nn.Dense(features=dim, use_bias=use_bias, name=name)
            layer_params = self.variables["params"][name]

            def pure_apply_fn(p, a, mod=layer_module):
                return mod.apply({"params": p}, a)

            activations = layer_wrapper_vjp(
                pure_apply_fn, layer_params, activations, name, collector
            )
        return activations
