from __future__ import annotations

from dataclasses import field

from flax import linen as nn
from jaxtyping import Array, Float

from src.hessians.collector import CollectorBase, layer_wrapper_vjp

from .approximation_model import ApproximationModel


class LinearModel(ApproximationModel):
    """Linear model with optional hidden layers.

    Each hidden layer consists of a Dense layer without activation.

    Note: Assumes for simplicity no bias in the layers.
    """

    hidden_dim: list[int] | None = field(default_factory=list)

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch_size input_dim"]
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass of the Linear model.
        Returns the logits of the model.
        """
        if self.hidden_dim is not None:
            for i, h in enumerate(self.hidden_dim):
                x = nn.Dense(h, use_bias=False, name=f"linear_{i}")(x)
        final_logits = nn.Dense(self.output_dim, use_bias=False, name="output")(x)
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

        def pure_apply_fn(module, params, activations):
            """Helper to apply a module with given parameters."""
            return module.apply({"params": params}, activations)

        activations = x

        if self.hidden_dim is not None:
            for i, h in enumerate(self.hidden_dim):
                layer_module = nn.Dense(features=h, use_bias=False, name=f"linear_{i}")
                layer_params = self.variables["params"][f"linear_{i}"]

            activations = layer_wrapper_vjp(
                lambda p, a: pure_apply_fn(layer_module, p, a),
                layer_params,
                activations,
                f"linear_{i}",
                collector,
            )

        final_layer_module = nn.Dense(
            features=self.output_dim, use_bias=False, name="output"
        )
        final_layer_params = self.variables["params"]["output"]
        final_logits = layer_wrapper_vjp(
            lambda p, a: pure_apply_fn(final_layer_module, p, a),
            final_layer_params,
            activations,
            "output",
            collector,
        )
        return final_logits
