from __future__ import annotations

from dataclasses import field
from typing import List, Tuple

from flax import linen as nn
from jaxtyping import Array, Float

from src.hessians.collector import CollectorBase, layer_wrapper_vjp
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.swiglu import SwiGLU


class MLPSwiGLU(ApproximationModel):
    """Multi-layer Perceptron model with SwiGLU activation blocks.

    Each hidden layer consists of a SwiGLU block with configurable dimensions.

    Note: Assumes for simplicity no bias in the layers.
    """

    hidden_dim: List[Tuple[int, int, int]] | None = field(default_factory=list)
    activation: str = "swiglu"

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.activation == "swiglu", "Only 'swiglu' activation is supported."

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch_size input_dim"]
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass of the MLP model with SwiGLU activations.
        Returns the logits of the model.
        """
        if self.hidden_dim is not None:
            for i, (up_dim, gate_dim, down_dim) in enumerate(self.hidden_dim):
                x = SwiGLU(
                    up_dim=up_dim,
                    gate_dim=gate_dim,
                    down_dim=down_dim,
                    name=f"swiglu_{i}",
                )(x)

        # Final output layer
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

        if self.hidden_dim is not None:
            for i, (up_dim, gate_dim, down_dim) in enumerate(self.hidden_dim):
                swiglu_module = SwiGLU(
                    up_dim=up_dim,
                    gate_dim=gate_dim,
                    down_dim=down_dim,
                    name=f"swiglu_{i}",
                )
                swiglu_params = self.variables["params"][f"swiglu_{i}"]

                # Use the SwiGLU's collector_apply method
                activations = swiglu_module.apply(
                    {"params": swiglu_params},
                    activations,
                    collector,
                    prefix=f"swiglu_{i}",
                    method=swiglu_module.collector_apply,
                )

        # Final output layer
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
