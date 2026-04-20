"""Shared pytest fixtures for training-heavy test scenarios.

The approximation tests are intentionally routed through a small set of shared
training scenarios so compatible modules reuse the same trained model instead of
retraining equivalent setups. The cache is backed by ``trained_model_registry``
and keyed by model config, dataset contents, training hyperparameters, seed,
and shuffle flag.

At the moment, a full test session trains 5 distinct models total:
- shared_linear_classification_scenario
- shared_linear_regression_scenario
- shared_multiclass_mlp_scenario
- test_jax_model simple_regression
- test_jax_model multi_feature_regression
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import pytest


@pytest.fixture(autouse=True)
def _reassert_x64():
    """Several computers call ``jax.config.update("jax_enable_x64", False)``
    inside ``@jax.jit`` bodies. Those updates fire at trace time and flip the
    flag globally, so later tests observe float32 and fail tolerances the
    suite was written against. Re-assert x64 before every test."""
    jax.config.update("jax_enable_x64", True)

from src.config import (
    ActivationFunction,
    LossType,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
)
from src.hessians.utils.data import ModelContext
from src.utils.data.data import (
    Dataset,
    RandomClassificationDataset,
    RandomRegressionDataset,
)
from src.utils.models.approximation_model import ApproximationModel
from tests._helpers import (
    cached_train_model_for_dataset,
    create_model_context,
)


@dataclass(frozen=True)
class TrainingScenario:
    name: str
    model_config: ModelConfig
    dataset: Dataset
    train_seed: int
    shuffle: bool = False


@pytest.fixture(scope="session")
def trained_model_registry() -> Dict[
    Tuple[Any, ...], Tuple[ApproximationModel, Dict, Callable]
]:
    """Cache trained models so compatible fixtures can reuse them across test modules."""
    return {}


def _make_training_scenario(
    *,
    name: str,
    tmp_path_factory,
    directory_name: str,
    model_config: ModelConfig,
    dataset: Dataset,
    train_seed: int,
    shuffle: bool = False,
) -> TrainingScenario:
    base = tmp_path_factory.mktemp(directory_name)
    model_config.directory = str(base / "model")
    return TrainingScenario(
        name=name,
        model_config=model_config,
        dataset=dataset,
        train_seed=train_seed,
        shuffle=shuffle,
    )


@pytest.fixture(scope="session")
def shared_multiclass_mlp_scenario(tmp_path_factory) -> TrainingScenario:
    return _make_training_scenario(
        name="shared_multiclass_mlp",
        tmp_path_factory=tmp_path_factory,
        directory_name="shared_multiclass_mlp",
        model_config=ModelConfig(
            architecture=ModelArchitecture.MLP,
            input_dim=12,
            hidden_dim=[8, 8],
            activation=ActivationFunction.TANH,
            output_dim=4,
            loss=LossType.CROSS_ENTROPY,
            training=TrainingConfig(
                learning_rate=1e-3,
                weight_decay=0,
                optimizer=OptimizerType.ADAMW,
                epochs=60,
                batch_size=64,
            ),
        ),
        dataset=RandomClassificationDataset(
            n_samples=256,
            n_features=12,
            n_informative=8,
            n_classes=4,
            seed=123,
        ),
        train_seed=42,
    )


@pytest.fixture(scope="session")
def shared_linear_regression_scenario(tmp_path_factory) -> TrainingScenario:
    return _make_training_scenario(
        name="shared_linear_regression",
        tmp_path_factory=tmp_path_factory,
        directory_name="shared_linear_regression",
        model_config=ModelConfig(
            architecture=ModelArchitecture.LINEAR,
            input_dim=10,
            hidden_dim=None,
            output_dim=2,
            loss=LossType.MSE,
            training=TrainingConfig(
                learning_rate=1e-3,
                optimizer=OptimizerType.SGD,
                epochs=100,
                batch_size=64,
            ),
        ),
        dataset=RandomRegressionDataset(
            n_samples=512,
            n_features=10,
            n_targets=2,
            noise=1.0,
            seed=0,
        ),
        train_seed=0,
    )


@pytest.fixture(scope="session")
def shared_linear_classification_scenario(tmp_path_factory) -> TrainingScenario:
    return _make_training_scenario(
        name="shared_linear_classification",
        tmp_path_factory=tmp_path_factory,
        directory_name="shared_linear_classification",
        model_config=ModelConfig(
            architecture=ModelArchitecture.LINEAR,
            input_dim=12,
            hidden_dim=None,
            activation=None,
            output_dim=4,
            loss=LossType.CROSS_ENTROPY,
            training=TrainingConfig(
                learning_rate=1e-3,
                weight_decay=0,
                optimizer=OptimizerType.ADAMW,
                epochs=60,
                batch_size=64,
            ),
        ),
        dataset=RandomClassificationDataset(
            n_samples=256,
            n_features=12,
            n_informative=8,
            n_classes=4,
            seed=123,
        ),
        train_seed=42,
    )


@pytest.fixture(scope="session")
def fim_classification_scenario(
    shared_multiclass_mlp_scenario: TrainingScenario,
) -> TrainingScenario:
    return shared_multiclass_mlp_scenario


@pytest.fixture(scope="session")
def fim_regression_scenario(
    shared_linear_regression_scenario: TrainingScenario,
) -> TrainingScenario:
    return shared_linear_regression_scenario


@pytest.fixture(scope="session")
def hessian_gnh_random_regression_scenario(
    shared_linear_regression_scenario: TrainingScenario,
) -> TrainingScenario:
    return shared_linear_regression_scenario


@pytest.fixture(scope="session")
def hessian_gnh_classification_scenario(
    shared_linear_classification_scenario: TrainingScenario,
) -> TrainingScenario:
    return shared_linear_classification_scenario


@pytest.fixture(scope="session")
def ekfac_linear_scenario(
    shared_linear_classification_scenario: TrainingScenario,
) -> TrainingScenario:
    return shared_linear_classification_scenario


@pytest.fixture(scope="session")
def ekfac_multi_layer_scenario(
    shared_multiclass_mlp_scenario: TrainingScenario,
) -> TrainingScenario:
    return shared_multiclass_mlp_scenario


@pytest.fixture(scope="session")
def block_test_config(
    request,
    shared_linear_classification_scenario: TrainingScenario,
    shared_multiclass_mlp_scenario: TrainingScenario,
) -> ModelConfig:
    """Shared model config for block-structure approximation tests."""
    scenario_name = getattr(request, "param", "linear")
    if scenario_name == "linear":
        return shared_linear_classification_scenario.model_config
    return shared_multiclass_mlp_scenario.model_config


@pytest.fixture(scope="session")
def block_test_dataset(
    block_test_config: ModelConfig,
    shared_linear_classification_scenario: TrainingScenario,
    shared_multiclass_mlp_scenario: TrainingScenario,
) -> Dataset:
    """Shared dataset for block-Hessian and block-FIM tests."""
    if block_test_config.architecture == ModelArchitecture.LINEAR:
        return shared_linear_classification_scenario.dataset
    return shared_multiclass_mlp_scenario.dataset


@pytest.fixture(scope="session")
def block_test_model_params_loss(
    trained_model_registry: Dict[
        Tuple[Any, ...], Tuple[ApproximationModel, Dict, Callable]
    ],
    block_test_config: ModelConfig,
    block_test_dataset: Dataset,
    shared_linear_classification_scenario: TrainingScenario,
    shared_multiclass_mlp_scenario: TrainingScenario,
) -> Tuple[ApproximationModel, Dict, Callable]:
    """Train once per shared block-test configuration and reuse across modules."""
    if block_test_config.architecture == ModelArchitecture.LINEAR:
        scenario = shared_linear_classification_scenario
    else:
        scenario = shared_multiclass_mlp_scenario

    return cached_train_model_for_dataset(
        block_test_config,
        block_test_dataset,
        trained_model_registry,
        seed=scenario.train_seed,
        shuffle=scenario.shuffle,
    )


@pytest.fixture(scope="session")
def block_test_model_context(
    block_test_dataset: Dataset,
    block_test_model_params_loss: Tuple[ApproximationModel, Dict, Callable],
) -> ModelContext:
    """Shared model context for block-structure approximation tests."""
    return create_model_context(block_test_dataset, block_test_model_params_loss)
