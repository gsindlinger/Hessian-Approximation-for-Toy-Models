from __future__ import annotations

import json
import logging
import os
import pickle
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state
from tqdm import tqdm

from src.config import ModelConfig
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def create_train_state(model, params, optimizer):
    """Create a training state using Flax's TrainState."""
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(state: train_state.TrainState, batch_data, batch_targets, loss_fn):
    """Single training step"""

    def loss_fn_wrapper(params):
        outputs = state.apply_fn(params, batch_data)
        return loss_fn(outputs, batch_targets)

    loss_value, grads = jax.value_and_grad(loss_fn_wrapper)(state.params)
    # Check gradient norms
    grad_norm = optax.global_norm(grads)

    state = state.apply_gradients(grads=grads)

    return state, loss_value, grad_norm


def train_model(
    model: ApproximationModel,
    dataloader: JAXDataLoader,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    epochs: int,
) -> Tuple[ApproximationModel, Dict, List]:
    """Train the model."""

    # Create training state
    params = initialize_model(
        model, input_shape=model.input_dim, key=jax.random.PRNGKey(model.seed)
    )
    state = create_train_state(model, params, optimizer)

    loss_history = []

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        total_samples = 0

        for batch_data, batch_targets in dataloader:
            state, loss_value, grad_norm = train_step(
                state, batch_data, batch_targets, loss_fn
            )
            if grad_norm < 1e-6:
                logger.warning(
                    f"Gradient norm is very small ({grad_norm}). Possible vanishing gradients."
                )
            running_loss += float(loss_value) * batch_data.shape[0]
            total_samples += batch_data.shape[0]

        epoch_loss = running_loss / total_samples
        loss_history.append(epoch_loss)
        if epoch % 50 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Grad Norm: {grad_norm:.6f}"
            )

    assert isinstance(state.params, Dict)
    return model, state.params, loss_history


@jax.jit
def _predict(apply_fn, params, x):
    return apply_fn(params, x)


def predict(model: ApproximationModel, params: Dict, x):
    x = jnp.array(x)
    return _predict(model.apply, params, x)


@partial(jax.jit, static_argnames=("loss_fn", "apply_fn"))
def _evaluate(
    apply_fn: Callable,
    params: Dict,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_fn: Callable,
):
    outputs = apply_fn(params, inputs)
    return loss_fn(outputs, targets)


def evaluate_loss(
    model: ApproximationModel,
    params: Dict,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_fn: Callable,
):
    """Evaluate model."""
    loss_value = _evaluate(model.apply, params, inputs, targets, loss_fn)
    return float(loss_value)


def evaluate_loss_and_classification_accuracy(
    model: ApproximationModel,
    params: Dict,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_fn: Callable,
):
    loss, acc = _evaluate_loss_and_classification_accuracy(
        model.apply, params, inputs, targets, loss_fn
    )
    return float(loss), float(acc)


@partial(jax.jit, static_argnames=["apply_fn", "loss_fn"])
def _evaluate_loss_and_classification_accuracy(
    apply_fn: Callable,
    params: Dict,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_fn: Callable,
):
    logits = apply_fn(params, inputs)
    loss = loss_fn(logits, targets)

    assert isinstance(logits, jnp.ndarray)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == targets)
    return loss, acc


def initialize_model(model: ApproximationModel, input_shape: int, key=None):
    """Initialize model parameters."""
    if key is None:
        key = jax.random.PRNGKey(model.seed)

    # Create dummy input for initialization
    dummy_input = jnp.ones((1, input_shape), dtype=jnp.float32)
    params = model.init(key, dummy_input)

    # log device of params
    logger.info(
        "Device for Parameters: %s",
        {x.device for x in jax.tree_util.tree_leaves(params)},
    )
    return params


# Utility for batched operations with vmap
def batch_predict(model: ApproximationModel, params: Dict, x_batch):
    """Vectorized prediction for efficient batch processing."""

    @jax.vmap
    def single_predict(x):
        return model.apply(params, x[None, :])[0]

    x_batch = jnp.array(x_batch)
    return single_predict(x_batch)


METADATA_STR = "metadata"


def save_model_checkpoint(
    model_config: ModelConfig,
    params: Dict,
    metadata: Optional[Dict] = None,
) -> None:
    """Save model parameters using orbax for better compatibility."""

    # check wheter directory exists, if not create it
    assert model_config.directory is not None, (
        "Model directory must be specified in ModelConfig."
    )
    directory = model_config.directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    # first save the model and metadata as json as single json file
    model_json = model_config.serialize()
    if metadata is not None:
        model_json.update({METADATA_STR: metadata})
    with open(f"{directory}/model.json", "w") as f:
        json.dump(model_json, f, indent=4)

    try:
        # Use Flax serialization for better handling of PyTrees
        bytes_output = serialization.to_bytes(params)
        with open(f"{directory}/checkpoint.msgpack", "wb") as f:
            f.write(bytes_output)
    except ImportError:
        # Fallback to pickle if serialization not available
        try:
            with open(f"{directory}/checkpoint.pkl", "wb") as f:
                pickle.dump(params, f)
        except Exception as e:
            logger.error(f"Failed to save model parameters: {e}")


def check_saved_model(directory: str) -> bool:
    """Check if a saved model exists with saved parameters."""
    model_path = os.path.join(directory, "model.json")
    checkpoint_path_msgpack = os.path.join(directory, "checkpoint.msgpack")
    checkpoint_path_pkl = os.path.join(directory, "checkpoint.pkl")

    # read model definition and compare
    if not os.path.exists(model_path):
        return False

    with open(model_path, "r") as f:
        saved_model_data = json.load(f)

    # remove metadata before comparison
    saved_model_data.pop(METADATA_STR, None)

    model_config = ModelConfig.from_dict(saved_model_data)

    return (
        os.path.exists(checkpoint_path_msgpack) or os.path.exists(checkpoint_path_pkl)
    ) and model_config is not None


def load_model_checkpoint(
    directory: str,
) -> Tuple[Dict, ApproximationModel, ModelConfig, Dict]:
    """Load model and parameters from checkpoint directory.
    Returns:
        params: Model parameters as a PyTree.
        model: The model instance.
        metadata: Additional metadata if available, else empty dict.
    """

    if not check_saved_model(directory):
        raise FileNotFoundError(f"No saved model found in directory: {directory}")

    # Load model definition and compare with provided model
    with open(f"{directory}/model.json", "r") as f:
        model_data_serialized = json.load(f)
        metadata = model_data_serialized.pop(METADATA_STR, None)
        if not metadata:
            metadata = {}

        model_config = ModelConfig.from_dict(model_data_serialized)

        model = ModelRegistry.get_model(
            model_config=model_config,
        )

    # Load parameters
    checkpoint_path_msgpack = os.path.join(directory, "checkpoint.msgpack")
    checkpoint_path_pkl = os.path.join(directory, "checkpoint.pkl")
    if os.path.exists(checkpoint_path_msgpack):
        with open(checkpoint_path_msgpack, "rb") as f:
            bytes_input = f.read()
        params = serialization.from_bytes(
            model.init(jax.random.PRNGKey(0), jnp.ones((1, model.input_dim))),
            bytes_input,
        )
    elif os.path.exists(checkpoint_path_pkl):
        with open(checkpoint_path_pkl, "rb") as f:
            params = pickle.load(f)
    else:
        raise FileNotFoundError("No checkpoint file found.")

    return params, model, model_config, metadata
