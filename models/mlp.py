from __future__ import annotations

from dataclasses import field

from flax import linen as nn

from models.train import ApproximationModel


class MLPModel(ApproximationModel):
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

    @nn.compact
    def __call__(self, x):
        act_fn = self.get_activation(self.activation)
        for i, h in enumerate(self.hidden_dim):
            x = nn.Dense(h, use_bias=self.use_bias, name=f"linear_{i}")(x)
            x = act_fn(x)
        x = nn.Dense(self.output_dim, use_bias=self.use_bias, name="output")(x)
        return x

    @nn.compact
    def kfac_apply(self, x, collector):
        from hessian_approximations.kfac.activation_gradient_collector import (
            layer_wrapper_vjp,
        )

        """A special apply method for K-FAC that wraps layers by custom VJP."""
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

        activations = layer_wrapper_vjp(
            pure_apply_fn_output, output_params, activations, "output", collector
        )

        return activations
