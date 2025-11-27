from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, Literal, Tuple

import jax
import jax.numpy as jnp
from flax.training import train_state
from tqdm import tqdm

from ..config.config import Config, TrainingConfig
from ..data.data import AbstractDataset, create_dataset
from .base import ApproximationModel
from .dataclasses.model_context import ModelContext
from .linear_model import LinearModel
from .mlp import MLPModel
from .utils.checkpoint_storage import (
    check_saved_model,
    load_model_checkpoint,
    save_model_checkpoint,
)
from .utils.loss import get_loss_fn
from .utils.optimizers import get_optimizer

logger = logging.getLogger(__name__)


def create_model(config: Config, input_dim: int, output_dim: int) -> ApproximationModel:
    """Create model from config."""
    model_map = {
        "linear": LinearModel,
        "mlp": MLPModel,
        # Add more models here as you implement them
    }

    model_cls = model_map.get(config.model.name)
    if model_cls is None:
        logger.error(f"Unknown model: {config.model.name}")
        raise ValueError(f"Unknown model: {config.model.name}")

    model_kwargs = vars(config.model).copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("loss", None)
    model_kwargs.update({"input_dim": input_dim, "output_dim": output_dim})

    assert issubclass(model_cls, ApproximationModel), (
        "Model must inherit from ApproximationModel"
    )

    return model_cls(**model_kwargs)


def create_dataset_and_model(
    config: Config,
) -> Tuple[ApproximationModel, AbstractDataset]:
    dataset = create_dataset(config.dataset)
    model = create_model(
        config, input_dim=dataset.input_dim(), output_dim=dataset.output_dim()
    )
    return model, dataset


def train_or_load(config: Config, reload_model: bool = False) -> ModelContext:
    """Train or load a model based on the given config."""
    model, dataset = create_dataset_and_model(config)
    loss = get_loss_fn(config.model.loss)
    if not reload_model and check_saved_model(
        dataset_config=config.dataset,
        model_config=config.model,
        training_config=config.training,
    ):
        params = load_model_checkpoint(
            dataset_config=config.dataset,
            model_config=config.model,
            training_config=config.training,
        )
    else:
        model, params = train_model(
            model=model,
            dataloader=dataset.get_dataloaders(),
            training_config=config.training,
        )
        save_model_checkpoint(
            params=params,
            dataset_config=config.dataset,
            model_config=config.model,
            training_config=config.training,
        )
    return ModelContext(model=model, dataset=dataset, params=params, loss=loss)


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
    state = state.apply_gradients(grads=grads)

    return state, loss_value


@partial(jax.jit, static_argnums=(3,))
def eval_step(state: train_state.TrainState, batch_data, batch_targets, loss_fn):
    """Single evaluation step"""
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
        loss_value = eval_step(state, batch_data, batch_targets, loss_fn)
        running_loss += loss_value * batch_data.shape[0]
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
            state, loss_value = train_step(state, batch_data, batch_targets, loss_fn)
            running_loss += float(loss_value) * batch_data.shape[0]
            total_samples += batch_data.shape[0]

        epoch_loss = running_loss / total_samples

        if val_loader is not None:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_loss = validate_model(state, val_loader, loss_fn)
                tqdm.write(f"Epoch {epoch + 1} Train Loss: {epoch_loss:.4f}")
                tqdm.write(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

    return model, state.params


@jax.jit
def predict_jit(apply_fn, params, x):
    return apply_fn(params, x)


def predict(model: ApproximationModel, params: Dict, x):
    x = jnp.array(x)
    return predict_jit(model.apply, params, x)


@jax.jit
def evaluate_jit(apply_fn, params, data, targets, loss_fn):
    outputs = apply_fn(params, data)
    return loss_fn(outputs, targets)


def evaluate(
    model: ApproximationModel,
    params: Dict,
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


def initialize_model(model: ApproximationModel, input_shape: int, key=None):
    """Initialize model parameters."""
    if key is None:
        key = jax.random.PRNGKey(0)

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
