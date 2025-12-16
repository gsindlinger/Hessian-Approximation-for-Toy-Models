import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.models.approximation_model import ApproximationModel

logger = logging.getLogger(__name__)


@dataclass
class CollectorBase(ABC):
    """
    Base class for collecting activations and gradients via hooks.
    """

    model: ApproximationModel
    params: Dict

    def collect(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        loss_fn: Callable,
        batch_size: Optional[int] = None,
        save_directory: Optional[str] = None,
    ) -> Any:
        """
        Run the model with hooks to collect activations and gradients.

        Args:
            model: The neural network model (Flax Module).
            params: The model parameters.
            inputs: Input data.
            targets: Target data.
            loss_fn: Loss function.
            save_path: Optional path to save collected data.

        Returns:
            Collected data from teardown().
        """

        def loss_fn_for_grad(p, inputs, targets):
            predictions = self.model.apply(
                p,
                inputs,
                self,
                method=self.model.collector_apply,
            )
            # Use sum reduction to avoid prematurely averaging gradients
            return loss_fn(predictions, targets, reduction="sum")

        dataloader_batch_size = (
            JAXDataLoader.get_batch_size() if batch_size is None else batch_size
        )
        dataloader = JAXDataLoader(
            inputs=inputs,
            targets=targets,
            shuffle=False,
            batch_size=dataloader_batch_size,
        )

        logger.info(f"Start collecting data for Collector: {self.__class__.__name__}")

        for batch_data, batch_targets in dataloader:
            # JIT-compiled gradient computation
            _ = jax.value_and_grad(loss_fn_for_grad)(
                self.params, batch_data, batch_targets
            )

        logger.info(
            f"Finished collecting data for Collector: {self.__class__.__name__}"
        )

        # Get collected data
        collected_data = self.teardown()

        if save_directory is not None:
            logger.info(f"Saving collected data to: {save_directory}")
            self.save(save_directory, collected_data)

        return collected_data

    @abstractmethod
    def save(self, save_path: str, data: Any):
        """
        Save collected data to the specified path.
        """
        pass

    @staticmethod
    @abstractmethod
    def load(load_path: str) -> Dict[str, Any]:
        """
        Load collected data from the specified path.
        """
        pass

    @abstractmethod
    def teardown(self) -> Dict[str, Any]:
        """
        Function to be called after all data has been collected.

        Returns:
            Dictionary containing collected data
        """
        pass

    @abstractmethod
    def backward_collector_fn(self, name: str, a: jnp.ndarray, g: jnp.ndarray):
        """Function to be called during backward pass to collect information."""
        pass


@dataclass
class CollectorActivationsGradients(CollectorBase):
    """
    Collector specifically for storing activations and gradients for K-FAC.
    Inherits from CollectorBase.
    """

    activations: Dict[str, List[Float[Array, "N I"]]] = field(default_factory=dict)
    gradients: Dict[str, List[Float[Array, "N O"]]] = field(default_factory=dict)

    FILENAME: str = "collected_activations_gradients.npz"

    def backward_collector_fn(
        self, name: str, a: Float[Array, "N I"], g: Float[Array, "N O"]
    ):
        """Store activations and gradients for this layer."""
        if name not in self.activations:
            self.activations[name] = [a]
            self.gradients[name] = [g]
        else:
            self.activations[name].append(a)
            self.gradients[name].append(g)

    def teardown(
        self,
    ) -> Tuple[
        Dict[str, Float[Array, "N I"]], Dict[str, Float[Array, "N O"]], List[str]
    ]:
        """
        Concatenate all collected activations and gradients.

        Returns:
            Dictionary with 'activations' and 'gradients' keys and the layer names as a list (in order).
        """
        return (
            {k: jnp.concatenate(v, axis=0) for k, v in self.activations.items()},
            {k: jnp.concatenate(v, axis=0) for k, v in self.gradients.items()},
            self.model.get_layer_names(),
        )

    def save(
        self,
        directory: str,
        data: Tuple[
            Dict[str, Float[Array, "N I"]], Dict[str, Float[Array, "N O"]], List[str]
        ],
    ):
        """
        Save collected activations and gradients to the specified directory.
        Additionally stores the layer names which can be used for correct ordering later.
        """

        # if directory is a file path, use its directory instead
        if os.path.isfile(directory):
            logger.warning(
                f"Provided save_path {directory} is a file. Using its directory instead."
            )
            directory = os.path.dirname(directory)

        # check if directory exists
        if not os.path.exists(directory):
            logger.info(f"Directory {directory} does not exist. Creating it.")
            os.makedirs(directory)

        file_path = os.path.join(directory, self.FILENAME)

        activations, gradients, layer_names = data
        data_dict = {
            **{f"activations_{k}": np.array(v) for k, v in activations.items()},
            **{f"gradients_{k}": np.array(v) for k, v in gradients.items()},
            "layer_names": np.array(",".join(layer_names), dtype="U"),
        }
        np.savez_compressed(file=file_path, allow_pickle=False, **data_dict)

    @staticmethod
    def load(
        directory: str,
    ) -> Tuple[
        Dict[str, Float[Array, "N I"]], Dict[str, Float[Array, "N O"]], List[str]
    ]:
        """
        Load collected activations and gradients from the specified directory.
        Tries with or without file ending, both for .npz and .npy formats.

        Returns tuple of: (activations, gradients, layer_names)
        """
        load_path = f"{directory}/{CollectorActivationsGradients.FILENAME}"
        # Try loading without .npz extension
        try:
            loaded = np.load(load_path)
            assert isinstance(loaded, np.lib.npyio.NpzFile)
        except FileNotFoundError:
            raise ValueError(f"File not found: {load_path} or its variants")

        activations = {}
        gradients = {}
        layer_names = loaded["layer_names"].item().split(",")

        for k in loaded.files:
            if k.startswith("activations_"):
                layer_name = k[len("activations_") :]
                activations[layer_name] = jnp.array(loaded[k])
            elif k.startswith("gradients_"):
                layer_name = k[len("gradients_") :]
                gradients[layer_name] = jnp.array(loaded[k])

        return activations, gradients, layer_names


### Hooks for collecting activations and gradients of specific layers
# See application in kfac_apply method of ApproximationModel
# Most important part establishing hook in JAX's autodiff
@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def layer_wrapper_vjp(
    pure_apply_fn: Callable,
    params: Dict,
    x: jnp.ndarray,
    name: str,
    collector: CollectorBase,
):
    """
    Custom VJP wrapper for capturing activations and gradients.

    Args:
        pure_apply_fn: Pure function with signature fn(params, x) -> output
        params: Layer parameters
        x: Layer input (activations)
        name: Layer name for identification
        collector: CollectorBase instance to store captured data

    Returns:
        Layer output
    """
    return pure_apply_fn(params, x)


def layer_wrapper_fwd(
    pure_apply_fn: Callable,
    params: Dict,
    x: jnp.ndarray,
    name: str,
    collector: CollectorBase,
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
    collector: CollectorBase,
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

    # Store the captured activations and gradients for this layer
    collector.backward_collector_fn(name, a, g)

    # Compute actual gradients using JAX's VJP
    _primals, vjp_fn = jax.vjp(pure_apply_fn, params, a)
    param_grads, input_grads = vjp_fn(g)

    return (param_grads, input_grads)


# Register forward and backward functions
layer_wrapper_vjp.defvjp(layer_wrapper_fwd, layer_wrapper_bwd)
