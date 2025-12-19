from __future__ import annotations

import jax.nn as jnn
from flax import linen as nn
from jaxtyping import Array, Float

from src.hessians.collector import CollectorBase, layer_wrapper_vjp


class SwiGLU(nn.Module):
    """
    SwiGLU block with explicit W_gate, W_up, W_down.

    Gated(x) = silu(W_gate x) ⊙ (W_up x)
    Output   = W_down(Gated(x))
    """

    up_dim: int
    gate_dim: int
    down_dim: int

    def __post_init__(self) -> None:
        assert self.up_dim == self.gate_dim, (
            "For SwiGLU, up_dim must be equal to gate_dim."
        )

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch_size input_dim"]
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass of the SwiGLU block.
        Returns the output of the block.
        """
        gate = nn.Dense(
            self.gate_dim,
            use_bias=False,
            name="W_gate",
        )(x)

        up = nn.Dense(
            self.up_dim,
            use_bias=False,
            name="W_up",
        )(x)

        gated = jnn.silu(gate) * up

        out = nn.Dense(
            self.down_dim,
            use_bias=False,
            name="W_down",
        )(gated)

        return out

    @nn.compact
    def collector_apply(
        self,
        x: Float[Array, "batch_size input_dim"],
        collector: CollectorBase,
        prefix: str,
    ) -> Float[Array, "batch_size output_dim"]:
        """Forward pass with hooks for collecting activations and gradients.
        This method uses a custom VJP wrapper around each layer to enable
        collection of necessary data during forward and backward passes.

        Data is collected by the provided `collector` instance.

        Returns the output of the SwiGLU block.
        """

        def pure_apply_fn(module, params, activations):
            """Helper to apply a module with given parameters."""
            return module.apply({"params": params}, activations)

        # Gate branch
        gate_module = nn.Dense(self.gate_dim, use_bias=False, name="W_gate")
        gate_params = self.variables["params"]["W_gate"]
        gate = layer_wrapper_vjp(
            lambda p, a: pure_apply_fn(gate_module, p, a),
            gate_params,
            x,
            f"{prefix}/W_gate",
            collector,
        )

        # Up branch
        up_module = nn.Dense(self.up_dim, use_bias=False, name="W_up")
        up_params = self.variables["params"]["W_up"]
        up = layer_wrapper_vjp(
            lambda p, a: pure_apply_fn(up_module, p, a),
            up_params,
            x,
            f"{prefix}/W_up",
            collector,
        )

        # Gated activation
        gated = jnn.silu(gate) * up

        # Down projection
        down_module = nn.Dense(self.down_dim, use_bias=False, name="W_down")
        down_params = self.variables["params"]["W_down"]
        out = layer_wrapper_vjp(
            lambda p, a: pure_apply_fn(down_module, p, a),
            down_params,
            gated,
            f"{prefix}/W_down",
            collector,
        )

        return out
