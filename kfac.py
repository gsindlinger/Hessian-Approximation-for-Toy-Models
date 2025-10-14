import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Tuple, Any
from dataclasses import dataclass, field
import flax.linen as nn


from models.models import ApproximationModel

# ============================================================================
# SECTION 1: Data Collector
# ============================================================================


class KFACCollector:
    """
    Captures layer inputs (activations) and output gradients during a forward-backward pass.

    This is a simple container that stores (activation, gradient) pairs for each layer.
    """

    def __init__(self):
        self.captured_data: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]] = {}

    def add(self, layer_name: str, data: Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Store captured data for a layer.

        Args:
            layer_name: Name of the layer
            data: Tuple of (activations, gradients)
        """
        self.captured_data[layer_name] = data


# ============================================================================
# SECTION 2: Custom VJP Wrapper
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def layer_wrapper_vjp(pure_apply_fn, params, x, name, collector):
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


def layer_wrapper_fwd(pure_apply_fn, params, x, name, collector):
    """
    Forward pass: compute output and save residuals for backward pass.

    Residuals are the values needed for gradient computation.
    """
    output = pure_apply_fn(params, x)
    residuals = (x, params)
    return output, residuals


def layer_wrapper_bwd(pure_apply_fn, name, collector, residuals, g):
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


# ============================================================================
# SECTION 3: KFAC State
# ============================================================================


@dataclass
class KFACState:
    """
    Immutable state for the K-FAC optimizer.

    Attributes:
        step: Current optimization step
        A_matrices: Dictionary of A covariance matrices (one per layer)
        G_matrices: Dictionary of G covariance matrices (one per layer)
    """

    step: int = 0
    A_matrices: Dict[str, jnp.ndarray] = field(default_factory=dict)
    G_matrices: Dict[str, jnp.ndarray] = field(default_factory=dict)


# ============================================================================
# SECTION 4: KFAC Optimizer
# ============================================================================


class KFACOptimizer:
    """
    K-FAC (Kronecker-Factored Approximate Curvature) optimizer.

    This optimizer approximates the Fisher Information Matrix using Kronecker products
    of smaller matrices, making second-order optimization tractable.
    """

    def __init__(self, cov_update_freq: int = 1, use_bias: bool = False):
        """
        Initialize the K-FAC optimizer.

        Args:
            cov_update_freq: How often to update covariance statistics (every N steps)
        """
        self.cov_update_freq = cov_update_freq
        self.use_bias = use_bias

    def init(self, params: Any) -> KFACState:
        """
        Initialize optimizer state.

        Args:
            params: Model parameters (not used, but included for consistency)

        Returns:
            Empty KFACState
        """
        return KFACState()

    def _compute_covariances(
        self, captured_data: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """
        Convert raw captured (activations, gradients) to A and G covariance matrices.

        For each layer:
        - A = (1/batch_size) * a^T @ a  where a includes bias term
        - G = (1/batch_size) * g^T @ g

        Args:
            captured_data: Dictionary mapping layer_name -> (a, g) where:
                - a: input activations with shape [batch_size, in_features]
                - g: output gradients with shape [batch_size, out_features]

        Returns:
            A_matrices: Dictionary mapping layer_name -> A covariance matrix
                       with shape [in_features + 1, in_features + 1]
            G_matrices: Dictionary mapping layer_name -> G covariance matrix
                       with shape [out_features, out_features]
        """
        A_matrices = {}
        G_matrices = {}

        for layer_name, (a, g) in captured_data.items():
            batch_size = a.shape[0]

            if self.use_bias:
                # Augment activations with bias term
                a = jnp.concatenate([a, jnp.ones((batch_size, 1))], axis=1)

            # Compute covariance matrices
            A = (a.T @ a) / batch_size
            G = (g.T @ g) / batch_size

            A_matrices[layer_name] = A
            G_matrices[layer_name] = G

        return A_matrices, G_matrices

    def update_statistics(
        self,
        state: KFACState,
        captured_data: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    ) -> KFACState:
        """
        Update covariance statistics with new captured data.

        Args:
            state: Current optimizer state
            captured_data: Newly captured (activations, gradients) from this batch

        Returns:
            New state with updated A and G matrices
        """
        A_matrices, G_matrices = self._compute_covariances(captured_data)

        return KFACState(step=state.step, A_matrices=A_matrices, G_matrices=G_matrices)

    def step(
        self,
        params: Any,
        grads: Any,
        state: KFACState,
        captured_data: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    ) -> Tuple[Any, KFACState]:
        """
        Perform one K-FAC optimization step.

        Currently just updates statistics. Returns unmodified gradients.

        Args:
            params: Model parameters (not used yet)
            grads: Gradients from backward pass (returned unmodified for now)
            state: Current optimizer state
            captured_data: Captured activations and gradients from this step

        Returns:
            grads: Unmodified gradients (placeholder for future preconditioning)
            new_state: Updated optimizer state
        """
        new_state = state

        # Update covariance statistics

        new_state = self.update_statistics(state, captured_data)

        # Increment step counter
        new_state = KFACState(
            step=new_state.step + 1,
            A_matrices=new_state.A_matrices,
            G_matrices=new_state.G_matrices,
        )

        return params, new_state


class LinearModel(ApproximationModel):
    """
    A refactored LinearModel using @nn.compact for robust scoping.
    """

    hidden_dim: list[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        """Standard forward pass for inference or non-K-FAC training."""
        activations = x
        # Define and apply hidden layers
        for i, h_dim in enumerate(self.hidden_dim):
            activations = nn.Dense(features=h_dim, name=f"linear_{i}", use_bias=False)(
                activations
            )

        # Define and apply the output layer
        activations = nn.Dense(features=self.output_dim, name="output")(activations)
        return activations

    @nn.compact
    def kfac_apply(self, x, collector):
        """A special apply method for K-FAC that wraps layers."""
        activations = x
        all_dims = self.hidden_dim + (self.output_dim,)
        layer_names = [f"linear_{i}" for i in range(len(self.hidden_dim))] + ["output"]

        for name, dim in zip(layer_names, all_dims):
            # Define the layer for this scope
            layer_module = nn.Dense(
                features=dim, name=name, use_bias=False
            )  # TODO: Change this
            layer_params = self.variables["params"][name]

            # THE FIX: Capture `layer_module` by value using a default argument.
            pure_apply_fn = lambda p, a, mod=layer_module: mod.apply({"params": p}, a)

            # Pass the pure function to the VJP wrapper
            activations = layer_wrapper_vjp(
                pure_apply_fn, layer_params, activations, name, collector
            )
        return activations
