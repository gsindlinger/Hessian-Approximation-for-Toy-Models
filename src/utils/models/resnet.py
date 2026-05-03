from __future__ import annotations

from dataclasses import field
from typing import List, Tuple

import jax.nn as jnn
from flax import linen as nn
from jaxtyping import Array, Float

from src.config import ActivationFunction
from src.hessians.collector import CollectorBase, layer_wrapper_vjp
from src.utils.models.approximation_model import ApproximationModel


class ResNetMLP(ApproximationModel):
    """ReLU residual MLP with 2-linear up/down blocks.

    Each block computes: h_out = h_in + W_down(ReLU(W_up(h_in))).

    `hidden_dim` is a list of (up_dim, down_dim) per block. The residual stream
    dim is the first block's down_dim; the embedding projects the input to that
    dim, and a residual_proj is inserted whenever consecutive down_dims differ.

    No bias on any Linear; Kaiming/He uniform init throughout.
    """

    hidden_dim: List[Tuple[int, int]] | None = field(default_factory=list)
    activation: ActivationFunction = ActivationFunction.RELU

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.activation == ActivationFunction.RELU, (
            "ResNetMLP only supports ReLU activation."
        )

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch_size input_dim"]
    ) -> Float[Array, "batch_size output_dim"]:
        init_fn = nn.initializers.he_uniform()

        if self.hidden_dim is not None and len(self.hidden_dim) > 0:
            _, first_down_dim = self.hidden_dim[0]
            x = nn.Dense(
                first_down_dim, use_bias=False, kernel_init=init_fn, name="embedding"
            )(x)

            for i, (up_dim, down_dim) in enumerate(self.hidden_dim):
                residual = x

                if x.shape[-1] != down_dim:
                    residual = nn.Dense(
                        down_dim,
                        use_bias=False,
                        kernel_init=init_fn,
                        name=f"residual_proj_{i}",
                    )(residual)

                x = nn.Dense(
                    up_dim, use_bias=False, kernel_init=init_fn, name=f"up_{i}"
                )(x)
                x = jnn.relu(x)
                x = nn.Dense(
                    down_dim, use_bias=False, kernel_init=init_fn, name=f"down_{i}"
                )(x)
                x = x + residual

        final_logits = nn.Dense(
            self.output_dim, use_bias=False, kernel_init=init_fn, name="output"
        )(x)
        return final_logits

    @nn.compact
    def collector_apply(
        self, x: Float[Array, "batch_size input_dim"], collector: CollectorBase
    ) -> Float[Array, "batch_size output_dim"]:
        def pure_apply_fn(module, params, activations):
            return module.apply({"params": params}, activations)

        init_fn = nn.initializers.he_uniform()
        activations = x

        if self.hidden_dim is not None and len(self.hidden_dim) > 0:
            _, first_down_dim = self.hidden_dim[0]
            embed_module = nn.Dense(
                features=first_down_dim,
                use_bias=False,
                kernel_init=init_fn,
                name="embedding",
            )
            embed_params = self.variables["params"]["embedding"]
            activations = layer_wrapper_vjp(
                lambda p, a: pure_apply_fn(embed_module, p, a),
                embed_params,
                activations,
                "embedding",
                collector,
            )

            for i, (up_dim, down_dim) in enumerate(self.hidden_dim):
                residual = activations

                if activations.shape[-1] != down_dim:
                    proj_module = nn.Dense(
                        features=down_dim,
                        use_bias=False,
                        kernel_init=init_fn,
                        name=f"residual_proj_{i}",
                    )
                    proj_params = self.variables["params"][f"residual_proj_{i}"]
                    residual = layer_wrapper_vjp(
                        lambda p, a: pure_apply_fn(proj_module, p, a),
                        proj_params,
                        residual,
                        f"residual_proj_{i}",
                        collector,
                    )

                up_module = nn.Dense(
                    features=up_dim,
                    use_bias=False,
                    kernel_init=init_fn,
                    name=f"up_{i}",
                )
                up_params = self.variables["params"][f"up_{i}"]
                activations = layer_wrapper_vjp(
                    lambda p, a: pure_apply_fn(up_module, p, a),
                    up_params,
                    activations,
                    f"up_{i}",
                    collector,
                )
                activations = jnn.relu(activations)

                down_module = nn.Dense(
                    features=down_dim,
                    use_bias=False,
                    kernel_init=init_fn,
                    name=f"down_{i}",
                )
                down_params = self.variables["params"][f"down_{i}"]
                activations = layer_wrapper_vjp(
                    lambda p, a: pure_apply_fn(down_module, p, a),
                    down_params,
                    activations,
                    f"down_{i}",
                    collector,
                )
                activations = activations + residual

        output_module = nn.Dense(
            features=self.output_dim,
            use_bias=False,
            kernel_init=init_fn,
            name="output",
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