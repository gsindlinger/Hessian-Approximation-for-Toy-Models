from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any, Callable, Dict, MutableMapping, Tuple

import numpy as np

from src.config import ModelConfig
from src.hessians.utils.data import ModelContext
from src.utils.data.data import Dataset
from src.utils.loss import get_loss
from src.utils.models.approximation_model import ApproximationModel
from src.utils.optimizers import optimizer
from src.utils.train import train_model

TrainedModelEntry = Tuple[ApproximationModel, Dict, Callable]


def _freeze_hidden_dim(hidden_dim: Any) -> Any:
    if hidden_dim is None:
        return None
    if isinstance(hidden_dim, (list, tuple)):
        return tuple(_freeze_hidden_dim(item) for item in hidden_dim)
    return hidden_dim


def _array_digest(array: Any) -> str:
    np_array = np.asarray(array)
    return sha256(np_array.tobytes()).hexdigest()


def training_cache_key(
    config: ModelConfig,
    dataset: Dataset,
    *,
    seed: int,
    shuffle: bool,
) -> Tuple[Any, ...]:
    """Build a cache key that is stable across equivalent test fixtures."""
    training_cfg = asdict(config.training)
    return (
        config.architecture.value,
        _freeze_hidden_dim(config.hidden_dim),
        config.activation.value if config.activation is not None else None,
        config.loss.value,
        tuple(sorted(training_cfg.items())),
        np.asarray(dataset.inputs).shape,
        str(np.asarray(dataset.inputs).dtype),
        _array_digest(dataset.inputs),
        np.asarray(dataset.targets).shape,
        str(np.asarray(dataset.targets).dtype),
        _array_digest(dataset.targets),
        seed,
        shuffle,
    )


def train_model_for_dataset(
    config: ModelConfig,
    dataset: Dataset,
    *,
    seed: int,
    shuffle: bool = False,
) -> Tuple[ApproximationModel, Dict, Callable]:
    """Train a model against a dataset without mutating the original config fixture."""
    model_config = replace(
        config,
        input_dim=dataset.input_dim(),
        output_dim=dataset.output_dim(),
    )
    loss_fn = get_loss(model_config.loss)

    model, params, _ = train_model(
        model_config=model_config,
        dataloader=dataset.get_dataloader(
            batch_size=model_config.training.batch_size,
            seed=seed,
            shuffle=shuffle,
        ),
        loss_fn=loss_fn,
        optimizer=optimizer(
            model_config.training.optimizer,
            lr=model_config.training.learning_rate,
        ),
        epochs=model_config.training.epochs,
    )

    return model, params, loss_fn


def cached_train_model_for_dataset(
    config: ModelConfig,
    dataset: Dataset,
    registry: MutableMapping[Tuple[Any, ...], TrainedModelEntry],
    *,
    seed: int,
    shuffle: bool = False,
) -> TrainedModelEntry:
    """Train once per equivalent config/dataset pair and reuse the result."""
    cache_key = training_cache_key(
        config,
        dataset,
        seed=seed,
        shuffle=shuffle,
    )
    if cache_key not in registry:
        registry[cache_key] = train_model_for_dataset(
            config,
            dataset,
            seed=seed,
            shuffle=shuffle,
        )
    return registry[cache_key]


def create_model_context(
    dataset: Dataset,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
) -> ModelContext:
    model, params, loss_fn = model_params_loss
    return ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss_fn,
    )
