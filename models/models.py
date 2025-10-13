from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import field
import os
from typing import Callable, Literal, Tuple, Any
from typing_extensions import override
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm
from config.config import Config, TrainingConfig
from functools import partial


def create_model(config: Config, input_dim: int, output_dim: int) -> ApproximationModel:
    """Create model from config."""
    model_map = {
        "linear": LinearModel,
        # Add more models here as you implement them
    }

    model_cls = model_map.get(config.model.name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {config.model.name}")

    model_kwargs = vars(config.model).copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("loss", None)
    model_kwargs.update({"input_dim": input_dim, "output_dim": output_dim})

    assert issubclass(
        model_cls, ApproximationModel
    ), "Model must inherit from ApproximationModel"

    return model_cls(**model_kwargs)


@jax.jit
def mse_loss(pred, target):
    """MSE loss (JIT compiled)."""
    return jnp.mean((pred - target) ** 2)


@jax.jit
def cross_entropy_loss(pred, target):
    """Cross entropy loss (JIT compiled)."""
    return optax.softmax_cross_entropy_with_integer_labels(pred, target).mean()


def get_loss_fn(loss_str: Literal["mse", "cross_entropy"] = "mse") -> Callable:
    """Return loss function."""
    match loss_str:
        case "mse":
            return mse_loss
        case "cross_entropy":
            return cross_entropy_loss
        case _:
            raise ValueError(f"Unknown loss function: {loss_str}")


def get_loss_name(loss_fn: Callable) -> str:
    """Return loss function name."""
    if loss_fn == mse_loss:
        return "mse"
    elif loss_fn == cross_entropy_loss:
        return "cross_entropy"
    else:
        return "unknown"


def loss_wrapper_flattened(
    model: ApproximationModel,
    params_flat: Any,
    unravel_fn: Callable[[Any], Any],
    loss_fn: Callable,
    training_data: jnp.ndarray,
    training_targets: jnp.ndarray,
):
    params_unraveled = unravel_fn(params_flat)
    outputs = model.apply(params_unraveled, training_data)
    return loss_fn(outputs, training_targets)


def get_optimizer(
    optimizer_str: Literal["sgd", "adam"], lr: float
) -> optax.GradientTransformation:
    """Return optimizer."""
    match optimizer_str:
        case "sgd":
            return optax.sgd(lr)
        case "adam":
            return optax.adam(lr)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_str}")


def create_train_state(model, params, optimizer):
    """Create a training state using Flax's TrainState."""
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(state: train_state.TrainState, batch_data, batch_targets, loss_fn):
    """Single training step (fully JIT compiled)."""

    def loss_fn_wrapper(params):
        outputs = state.apply_fn(params, batch_data)
        return loss_fn(outputs, batch_targets)

    loss_value, grads = jax.value_and_grad(loss_fn_wrapper)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss_value


@partial(jax.jit, static_argnums=(3,))
def eval_step(state: train_state.TrainState, batch_data, batch_targets, loss_fn):
    """Single evaluation step (fully JIT compiled)."""
    outputs = state.apply_fn(state.params, batch_data)
    loss_value = loss_fn(outputs, batch_targets)
    return loss_value


def validate_model(
    state: train_state.TrainState,
    dataloader,
    loss_fn,
):
    """Validate model on validation set."""
    running_loss = 0.0
    total_samples = 0

    for batch_data, batch_targets in dataloader:
        batch_data = jnp.array(batch_data)
        batch_targets = jnp.array(batch_targets)

        loss_value = eval_step(state, batch_data, batch_targets, loss_fn)
        running_loss += float(loss_value) * batch_data.shape[0]
        total_samples += batch_data.shape[0]

    val_loss = running_loss / total_samples
    return val_loss


def train_model(
    model: ApproximationModel,
    dataloader: Tuple[Any, Any | None],
    training_config: TrainingConfig,
) -> Tuple[ApproximationModel, Any]:
    """Train the model."""
    train_loader, val_loader = dataloader

    loss_fn = get_loss_fn(training_config.loss)
    optimizer = get_optimizer(training_config.optimizer, training_config.lr)

    # Create training state
    params = initialize_model(
        model, input_shape=model.input_dim, key=jax.random.PRNGKey(0)
    )
    state = create_train_state(model, params, optimizer)

    for epoch in tqdm(range(training_config.epochs)):
        running_loss = 0.0
        total_samples = 0

        for batch_data, batch_targets in train_loader:
            # Convert to JAX arrays
            batch_data = jnp.array(batch_data)
            batch_targets = jnp.array(batch_targets)

            state, loss_value = train_step(state, batch_data, batch_targets, loss_fn)

            running_loss += float(loss_value) * batch_data.shape[0]
            total_samples += batch_data.shape[0]

        epoch_loss = running_loss / total_samples

        if val_loader is not None:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_loss = validate_model(state, val_loader, loss_fn)
                tqdm.write(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")
                tqdm.write(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

    # Save checkpoint of the model
    if training_config.save_checkpoint:
        save_model(model, state.params)

    return model, state.params


def save_model(model: ApproximationModel, params: Any):
    """Save model parameters using orbax for better compatibility."""
    try:
        from flax import serialization
        import pickle

        model_name = model.__class__.__name__
        os.makedirs("data/checkpoints", exist_ok=True)

        # Use Flax serialization for better handling of PyTrees
        bytes_output = serialization.to_bytes(params)
        with open(f"data/checkpoints/{model_name}.msgpack", "wb") as f:
            f.write(bytes_output)
    except ImportError:
        # Fallback to pickle if serialization not available
        import pickle

        model_name = model.__class__.__name__
        os.makedirs("data/checkpoints", exist_ok=True)
        with open(f"data/checkpoints/{model_name}.pkl", "wb") as f:
            pickle.dump(params, f)


def load_model(model: ApproximationModel):
    """Load model parameters."""
    model_name = model.__class__.__name__

    # Try msgpack first (Flax serialization)
    checkpoint_path = f"data/checkpoints/{model_name}.msgpack"
    if os.path.exists(checkpoint_path):
        from flax import serialization

        with open(checkpoint_path, "rb") as f:
            bytes_input = f.read()
        params = serialization.from_bytes(None, bytes_input)
        return params

    # Fallback to pickle
    checkpoint_path = f"data/checkpoints/{model_name}.pkl"
    if os.path.exists(checkpoint_path):
        import pickle

        with open(checkpoint_path, "rb") as f:
            params = pickle.load(f)
        return params

    return None


@jax.jit
def predict_jit(apply_fn, params, x):
    """JIT compiled prediction."""
    return apply_fn(params, x)


def predict(model: ApproximationModel, params: Any, x):
    """Make predictions."""
    x = jnp.array(x)
    return predict_jit(model.apply, params, x)


@jax.jit
def evaluate_jit(apply_fn, params, data, targets, loss_fn):
    """JIT compiled evaluation."""
    outputs = apply_fn(params, data)
    return loss_fn(outputs, targets)


def evaluate(
    model: ApproximationModel,
    params: Any,
    data,
    targets,
    loss_str: Literal["mse", "cross_entropy"] = "mse",
):
    """Evaluate model."""
    loss_fn = get_loss_fn(loss_str)
    data = jnp.array(data)
    targets = jnp.array(targets)

    loss_value = evaluate_jit(model.apply, params, data, targets, loss_fn)
    return float(loss_value)


class ApproximationModel(nn.Module):
    """Base class for approximation models using Flax."""

    input_dim: int
    output_dim: int

    def get_activation(self, act_str: str):
        """Get activation function."""
        match act_str:
            case "relu":
                return nn.relu
            case "tanh":
                return nn.tanh
            case _:
                raise ValueError(f"Unknown activation function: {act_str}")


class LinearModel(ApproximationModel):
    """Linear model with optional hidden layers."""

    hidden_dim: list[int] = field(default_factory=list)

    def setup(self) -> None:
        if not self.hidden_dim:
            self.layers = []
        else:
            self.layers = [
                nn.Dense(h, name=f"linear_{i}") for i, h in enumerate(self.hidden_dim)
            ]
        self.output_layer = nn.Dense(self.output_dim, name="output")

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


def initialize_model(model: ApproximationModel, input_shape: int, key=None):
    """Initialize model parameters."""
    if key is None:
        key = jax.random.PRNGKey(0)

    # Create dummy input for initialization
    dummy_input = jnp.ones((1, input_shape))
    params = model.init(key, dummy_input)

    return params


# Utility for batched operations with vmap
def batch_predict(model: ApproximationModel, params: Any, x_batch):
    """Vectorized prediction for efficient batch processing."""

    @jax.vmap
    def single_predict(x):
        return model.apply(params, x[None, :])[0]

    x_batch = jnp.array(x_batch)
    return single_predict(x_batch)
