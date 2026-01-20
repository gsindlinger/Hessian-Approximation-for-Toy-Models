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
    model_config: ModelConfig,
    dataloader: JAXDataLoader,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    epochs: int,
    seed: int = 42,
    save_epochs: Optional[List[int]] = None,
) -> Tuple[ApproximationModel, Dict, List]:
    """Train the model."""

    model = ModelRegistry.get_model(model_config=model_config, seed=seed)

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
            if epoch < 200 and epoch % 10 == 0 and grad_norm < 1e-6:
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
        # Save checkpoint if required
        if save_epochs is not None and epoch in save_epochs:
            assert isinstance(state.params, Dict)
            save_model_checkpoint(
                model_config=model.get_model_config(),
                params=state.params,
                epoch=epoch,
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
    epoch: Optional[int] = None,
) -> None:
    """
    Save model and parameters to checkpoint directory.
    If epoch is specified, saves parameters in epoch-specific subdirectory.
    If epoch is None or the last epoch, saves parameters (additionally) and the model configuration in the base directory.
    So when epoch is the last epoch, parameters are saved both in the epoch directory and base directory.
    """

    def save_params(directory: str, params: Dict) -> None:
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

    # check whether directory exists, if not create it
    assert model_config.directory is not None, (
        "Model directory must be specified in ModelConfig."
    )
    base_directory = model_config.directory

    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    if epoch is not None:
        epoch_directory = os.path.join(base_directory, f"epoch_{epoch}")
        if not os.path.exists(epoch_directory):
            os.makedirs(epoch_directory)
        save_params(epoch_directory, params)

    if epoch is None or epoch == model_config.training.epochs - 1:
        save_params(base_directory, params)

        # first save the model and metadata as json as single json file
        model_json = model_config.serialize()
        if metadata is not None:
            model_json.update({METADATA_STR: metadata})
        with open(f"{base_directory}/model.json", "w") as f:
            json.dump(model_json, f, indent=4)


def check_saved_model(directory: str, epochs: Optional[List[int]] = None) -> bool:
    """Check if a saved model exists with saved parameters. If epochs is provided,
    also checks if checkpoints for the specified epochs exist.
    """
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

    if not (
        (os.path.exists(checkpoint_path_msgpack) or os.path.exists(checkpoint_path_pkl))
        and model_config is not None
    ):
        return False

    if epochs is not None:
        for epoch in epochs:
            epoch_directory = os.path.join(directory, f"epoch_{epoch}")
            checkpoint_path_msgpack = os.path.join(
                epoch_directory, "checkpoint.msgpack"
            )
            checkpoint_path_pkl = os.path.join(epoch_directory, "checkpoint.pkl")
            if not (
                os.path.exists(checkpoint_path_msgpack)
                or os.path.exists(checkpoint_path_pkl)
            ):
                return False

    return True


def load_model_checkpoint(
    directory: str,
    epoch: Optional[int] = None,
) -> Tuple[Dict, ApproximationModel, ModelConfig, Dict]:
    """
    Load model and parameters from checkpoint directory. Must be the models base directory, i.e. not the epoch subdirectory.
    If epoch is specified, loads parameters from epoch-specific subdirectory.
    If epoch is None, loads parameters from the base directory.

    Note, that this method requires the model to be fully trained and saved previously.
    Intermediate epochs can only be loaded when the full model was saved at the end of training.
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
    if epoch is not None:
        directory = os.path.join(directory, f"epoch_{epoch}")
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
