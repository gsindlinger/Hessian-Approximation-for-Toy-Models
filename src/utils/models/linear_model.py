from __future__ import annotations

from dataclasses import field

from flax import linen as nn
from jaxtyping import Array, Float

from src.hessians.collector import CollectorBase, layer_wrapper_vjp

from .approximation_model import ApproximationModel


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

    def get_linear_layer_names(self) -> list[str]:
        result = [name for name, _, _ in self._get_layer_config()]
        return result

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch_size input_dim"]
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass of the Linear model.
        Returns the logits of the model.
        """
        config = self._get_layer_config()
        for name, dim, use_bias in config:
            x = nn.Dense(dim, use_bias=use_bias, name=name)(x)
        return x

    @nn.compact
    def collector_apply(
        self, x: Float[Array, "batch_size input_dim"], collector: CollectorBase
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass with hooks for collecting activations and gradients.
        This method uses a custom VJP wrapper around each layer to enable
        collection of necessary data during forward and backward passes.
        Data is colleected by the provided `collector` instance.

        Returns the logits of the model.
        """
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

        final_logits = activations
        return final_logits
