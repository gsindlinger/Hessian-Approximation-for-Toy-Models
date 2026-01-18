from __future__ import annotations

from dataclasses import field

from flax import linen as nn
from jaxtyping import Array, Float

from src.config import ActivationFunction
from src.hessians.collector import CollectorBase, layer_wrapper_vjp
from src.utils.models.approximation_model import ApproximationModel


class ResNetMLP(ApproximationModel):
    """ResNet-style MLP with residual connections.

    Each layer computes: output = input + activation(Dense(input))

    Note: Assumes for simplicity no bias in the layers.
    """

    hidden_dim: list[int] | None = field(default_factory=list)
    activation: ActivationFunction = ActivationFunction.TANH

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.activation in {
            ActivationFunction.RELU,
            ActivationFunction.TANH,
        }, "ResNetMLP only supports ReLU and Tanh activations."

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch_size input_dim"]
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass of the ResNet MLP model.

        Returns the logits of the model.
        """
        act_fn = self.get_activation_fn(self.activation)

        if self.hidden_dim is not None:
            for i, h in enumerate(self.hidden_dim):
                residual = x

                # Project residual if dimensions don't match
                if x.shape[-1] != h:
                    residual = nn.Dense(h, use_bias=False, name=f"residual_proj_{i}")(
                        residual
                    )

                # output = input + act(Wx)
                x = nn.Dense(h, use_bias=False, name=f"linear_{i}")(x)
                x = act_fn(x)
                x = x + residual

        final_logits = nn.Dense(self.output_dim, use_bias=False, name="output")(x)
        return final_logits

    @nn.compact
    def collector_apply(
        self, x: Float[Array, "batch_size input_dim"], collector: CollectorBase
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass with hooks for collecting activations and gradients.

        This method uses a custom VJP wrapper around each layer to enable
        collection of necessary data during forward and backward passes.
        Data is collected by the provided `collector` instance.

        Returns the logits of the model.
        """

        def pure_apply_fn(module, params, activations):
            """Helper to apply a module with given parameters."""
            return module.apply({"params": params}, activations)

        activations = x
        act_fn = self.get_activation_fn(self.activation)

        if self.hidden_dim is not None:
            for i, h in enumerate(self.hidden_dim):
                residual = activations

                # Project residual if dimensions don't match
                if activations.shape[-1] != h:
                    proj_module = nn.Dense(
                        features=h, use_bias=False, name=f"residual_proj_{i}"
                    )
                    proj_params = self.variables["params"][f"residual_proj_{i}"]
                    residual = layer_wrapper_vjp(
                        lambda p, a: pure_apply_fn(proj_module, p, a),
                        proj_params,
                        residual,
                        f"residual_proj_{i}",
                        collector,
                    )

                # output = input + act(Wx)
                layer_module = nn.Dense(features=h, use_bias=False, name=f"linear_{i}")
                layer_params = self.variables["params"][f"linear_{i}"]
                activations = layer_wrapper_vjp(
                    lambda p, a: pure_apply_fn(layer_module, p, a),
                    layer_params,
                    activations,
                    f"linear_{i}",
                    collector,
                )
                activations = act_fn(activations)
                activations = activations + residual

        output_module = nn.Dense(
            features=self.output_dim, use_bias=False, name="output"
        )
        output_params = self.variables["params"]["output"]
        final_logits = layer_wrapper_vjp(
            lambda p, a: pure_apply_fn(output_module, p, a),
            output_params,
            activations,
            "output",
            collector,
        )

        return final_logits
