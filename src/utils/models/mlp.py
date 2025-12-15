from __future__ import annotations

from dataclasses import field

from flax import linen as nn
from jaxtyping import Array, Float

from src.hessians.collector import CollectorBase, layer_wrapper_vjp
from src.utils.models.approximation_model import ApproximationModel


class MLP(ApproximationModel):
    """Multi-layer Perceptron model with configurable hidden layers and activations."""

    hidden_dim: list[int] = field(default_factory=list)
    activation: str = "relu"

    def _get_layer_config(self):
        """Returns list of (name, dim, use_bias) tuples for all layers."""
        config = []
        for i, h in enumerate(self.hidden_dim):
            config.append((f"linear_{i}", h, self.use_bias))
        config.append(("output", self.output_dim, self.use_bias))
        return config

    def get_linear_layer_names(self) -> list[str]:
        return [name for name, _, _ in self._get_layer_config()]

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch_size input_dim"]
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass of the MLP model.
        Returns the logits of the model.
        """
        act_fn = self.get_activation(self.activation)
        for i, h in enumerate(self.hidden_dim):
            x = nn.Dense(h, use_bias=self.use_bias, name=f"linear_{i}")(x)
            x = act_fn(x)
        final_logits = nn.Dense(self.output_dim, use_bias=self.use_bias, name="output")(
            x
        )
        return final_logits

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
        act_fn = self.get_activation(self.activation)

        for i, h in enumerate(self.hidden_dim):
            layer_module = nn.Dense(
                features=h, use_bias=self.use_bias, name=f"linear_{i}"
            )
            layer_params = self.variables["params"][f"linear_{i}"]

            def pure_apply_fn(p, a, mod=layer_module):
                return mod.apply({"params": p}, a)

            activations = layer_wrapper_vjp(
                pure_apply_fn, layer_params, activations, f"linear_{i}", collector
            )
            activations = act_fn(activations)

        output_module = nn.Dense(
            features=self.output_dim, use_bias=self.use_bias, name="output"
        )
        output_params = self.variables["params"]["output"]

        def pure_apply_fn_output(p, a, mod=output_module):
            return mod.apply({"params": p}, a)

        final_logits = layer_wrapper_vjp(
            pure_apply_fn_output, output_params, activations, "output", collector
        )

        return final_logits
