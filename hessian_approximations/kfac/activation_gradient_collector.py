import os
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from hessian_approximations.kfac.layer_components import LayerComponents
from models.train import ApproximationModel


@dataclass
class ActivationGradientCollector:
    """
    Captures layer inputs (activations) and output gradients during a forward-backward pass.

    This is a simple container that stores (activation, gradient) pairs for each layer.
    """

    def __init__(self):
        self.captured_data: LayerComponents = LayerComponents()

    def add(self, layer_name: str, data: Tuple[jnp.ndarray, jnp.ndarray]) -> None:
        """
        Store captured data for a layer.

        Args:
            layer_name: Name of the layer
            data: Tuple of (activations, gradients)
        """
        self.captured_data[layer_name] = data

    def load_from_disk(self, model: ApproximationModel) -> None:
        """
        Load previously captured data from disk.
        """

        model_name = model.__class__.__name__
        data_path = f"data/{model_name}/activations_gradients.pkl"
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                self.captured_data = pickle.load(f)
            print(f"Loaded captured data from {data_path}")
        else:
            raise FileNotFoundError(f"No captured data file found at {data_path}")

    def save_to_disk(self, model: ApproximationModel) -> None:
        """
        Save captured data to disk for future use.
        """

        model_name = model.__class__.__name__
        os.makedirs(f"data/{model_name}", exist_ok=True)
        data_path = f"data/{model_name}/activations_gradients.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(self.captured_data, f)
        print(f"Saved captured data to {data_path}")


### Hooks for collecting activations and gradients of specific layers
# See application in kfac_apply method of ApproximationModel
# Most important part establishing hook in JAX's autodiff
@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def layer_wrapper_vjp(
    pure_apply_fn: Callable,
    params: Dict,
    x: jnp.ndarray,
    name: str,
    collector: ActivationGradientCollector,
):
    """
    Custom VJP wrapper for capturing activations and gradients.

    Args:
        pure_apply_fn: Pure function with signature fn(params, x) -> output
        params: Layer parameters
        x: Layer input (activations)
        name: Layer name for identification
        collector: KFACCollector instance to store captured data

    Returns:
        Layer output
    """
    return pure_apply_fn(params, x)


def layer_wrapper_fwd(
    pure_apply_fn: Callable,
    params: Dict,
    x: jnp.ndarray,
    name: str,
    collector: ActivationGradientCollector,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
    """
    Forward pass: compute output and save residuals for backward pass.

    Residuals are the values needed for gradient computation.
    """
    output = pure_apply_fn(params, x)
    residuals = (x, params)
    return output, residuals


def layer_wrapper_bwd(
    pure_apply_fn: Callable,
    name: str,
    collector: ActivationGradientCollector,
    residuals: Tuple[jnp.ndarray, Dict],
    g: jnp.ndarray,
) -> Tuple[Dict, jnp.ndarray]:
    """
    Backward pass: capture data and compute gradients.

    This is where the magic happens:
    - Capture input activations (a) and output gradients (g)
    - Store them in the collector
    - Compute and return actual gradients via VJP
    """
    a, params = residuals

    # Capture the data for K-FAC
    collector.add(name, (a, g))

    # Compute actual gradients using JAX's VJP
    _primals, vjp_fn = jax.vjp(pure_apply_fn, params, a)
    param_grads, input_grads = vjp_fn(g)

    return (param_grads, input_grads)


# Register forward and backward functions
layer_wrapper_vjp.defvjp(layer_wrapper_fwd, layer_wrapper_bwd)
