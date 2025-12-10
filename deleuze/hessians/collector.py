import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from deleuze.models.approximation_model import ApproximationModel
from deleuze.utils.data.jax_dataloader import JAXDataLoader

logger = logging.getLogger(__name__)


@dataclass
class CollectorBase(ABC):
    """
    Base class for collecting activations and gradients via hooks.
    """

    model: ApproximationModel
    params: Dict
    total_samples: int = field(default=0)

    def collect(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        loss_fn: Callable,
    ) -> Any:
        """
        Run the model with hooks to collect activations and gradients.

        Args:
            model: The neural network model (Flax Module).
            params: The model parameters.
            inputs: Input data.
            targets: Target data.
            loss_fn: Loss function.

        Returns:
            Model output.
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

        dataloader_batch_size = JAXDataLoader.get_batch_size()
        dataloader = JAXDataLoader(
            inputs=inputs,
            targets=targets,
            shuffle=False,
            batch_size=dataloader_batch_size,
        )

        logger.info(f"Start collecting data for Collector: {self.__class__.__name__}")

        self.total_samples = 0
        for batch_data, batch_targets in dataloader:
            # JIT-compiled gradient computation
            _ = jax.value_and_grad(loss_fn_for_grad)(
                self.params, batch_data, batch_targets
            )
            self.total_samples += batch_data.shape[0]

        logger.info(
            f"Finished collecting data for Collector: {self.__class__.__name__}"
        )
        return self.collected_data()

    @abstractmethod
    def collected_data(self) -> Any:
        pass

    @abstractmethod
    def backward_collector_fn(self, name: str, a: jnp.ndarray, g: jnp.ndarray):
        pass


@dataclass
class CollectorActivationsGradients(CollectorBase):
    """
    Collector specifically for storing activations and gradients for K-FAC.
    Inherits from HookCollectorBase.
    """

    activations: Dict[str, List[Float[Array, "N I"]]] = field(default_factory=dict)
    gradients: Dict[str, List[Float[Array, "N O"]]] = field(default_factory=dict)

    def backward_collector_fn(
        self, name: str, a: Float[Array, "N I"], g: Float[Array, "N O"]
    ):
        # Store activations and gradients for this layer
        # Note: We store the full batch data; so we have to extend the arrays for each batch
        if name not in self.activations:
            self.activations[name] = [a]
            self.gradients[name] = [g]
        else:
            self.activations[name].append(a)
            self.gradients[name].append(g)

    def collected_data(self):
        return (
            {k: jnp.concatenate(v, axis=0) for k, v in self.activations.items()},
            {k: jnp.concatenate(v, axis=0) for k, v in self.gradients.items()},
        )


@dataclass
class CollectorKFACCovariances(CollectorBase):
    """
    Collector specifically for storing covariances for EK-FAC.
    Inherits from HookCollectorBase.
    """

    activations_covs: Dict[str, Float[Array, "I I"]] = field(default_factory=dict)
    gradients_covs: Dict[str, Float[Array, "O O"]] = field(default_factory=dict)

    def backward_collector_fn(
        self, name: str, a: Float[Array, "N I"], g: Float[Array, "N O"]
    ):
        if name not in self.activations_covs:
            self.activations_covs[name] = jnp.zeros((a.shape[1], a.shape[1]))
            self.gradients_covs[name] = jnp.zeros((g.shape[1], g.shape[1]))

        self.activations_covs[name] += a.T @ a
        self.gradients_covs[name] += g.T @ g

    def collected_data(self):
        return {k: v / self.total_samples for k, v in self.activations_covs.items()}, {
            k: v / self.total_samples for k, v in self.gradients_covs.items()
        }


@dataclass
class CollectorEKFACEigenvalueCorrections(CollectorBase):
    """
    Collector specifically for storing eigenvalue corrections for EK-FAC.
    Inherits from HookCollectorBase.
    """

    eigenvectors_gradients: Dict[str, Float[Array, "O O"]] = field(default_factory=dict)
    eigenvectors_activations: Dict[str, Float[Array, "I I"]] = field(
        default_factory=dict
    )
    eigenvalue_corrections: Dict[str, Float[Array, "I O"]] = field(default_factory=dict)

    def backward_collector_fn(
        self, name: str, a: Float[Array, "N I"], g: Float[Array, "N O"]
    ):
        if name not in self.eigenvalue_corrections:
            self.eigenvalue_corrections[name] = jnp.zeros(
                (
                    self.eigenvectors_activations[name].shape[0],
                    self.eigenvectors_gradients[name].shape[0],
                )
            )

        g_tilde = jnp.einsum("op, np -> no", self.eigenvectors_gradients[name].T, g)
        a_tilde = jnp.einsum("ij, nj -> ni", self.eigenvectors_activations[name].T, a)
        outer = jnp.einsum("ni, no -> nio", a_tilde, g_tilde)
        self.eigenvalue_corrections[name] += (outer**2).sum(axis=0)

    def collected_data(self):
        return {
            k: v / self.total_samples for k, v in self.eigenvalue_corrections.items()
        }


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
