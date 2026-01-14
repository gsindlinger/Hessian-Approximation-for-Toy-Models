from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from jax.random import PRNGKey

from src.config import (
    LossType,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.fim import FIMComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets
from src.utils.data.data import (
    Dataset,
    RandomClassificationDataset,
    RandomRegressionDataset,
)
from src.utils.loss import get_loss, get_loss_name
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(params=["classification", "regression"], scope="session")
def config(request, tmp_path_factory):
    """Create model configuration for testing."""
    base = tmp_path_factory.mktemp(request.param)

    if request.param == "classification":
        architecture = ModelArchitecture.MLP
        hidden_dim = [10]
        loss = LossType.CROSS_ENTROPY
        tol_hvp = 0.2
        tol_ihvp = 0.1
        tol_cosine = 0.95
    else:  # regression
        architecture = ModelArchitecture.LINEAR
        hidden_dim = [5]
        loss = LossType.MSE
        tol_hvp = 0.6
        tol_ihvp = 0.1
        tol_cosine = 0.95

    model_config = ModelConfig(
        architecture=architecture,
        input_dim=10,  # Will be updated from dataset
        hidden_dim=hidden_dim,
        output_dim=2,  # Will be updated from dataset
        loss=loss,
        training=TrainingConfig(
            learning_rate=1e-3,
            optimizer=OptimizerType.SGD,
            epochs=50,
            batch_size=128,
        ),
        directory=str(base / "model"),
    )

    return {
        "model_config": model_config,
        "tol_hvp": tol_hvp,
        "tol_ihvp": tol_ihvp,
        "tol_cosine": tol_cosine,
    }


@pytest.fixture(scope="session")
def dataset(config: Dict) -> Dataset:
    """Create a random dataset for testing (classification or regression)."""
    seed = 0

    if config["model_config"].loss == LossType.CROSS_ENTROPY:
        return RandomClassificationDataset(
            n_samples=1500,
            n_features=20,
            n_informative=10,
            n_classes=5,
            seed=seed,
        )
    else:  # MSE / regression
        return RandomRegressionDataset(
            n_samples=1500,
            n_features=10,
            n_targets=2,
            noise=5.0,
            seed=seed,
        )


@pytest.fixture(scope="session")
def model_params_loss(
    config: Dict, dataset: Dataset
) -> Tuple[ApproximationModel, Dict, Callable]:
    """Train a model and return it with its parameters and loss function."""
    model_config = config["model_config"]

    # Update dimensions from dataset
    model_config.input_dim = dataset.input_dim()
    model_config.output_dim = dataset.output_dim()

    # Get model from registry
    model = ModelRegistry.get_model(model_config=model_config)

    # Train the model
    model, params, _ = train_model(
        model,
        dataset.get_dataloader(batch_size=model_config.training.batch_size, seed=0),
        loss_fn=get_loss(model_config.loss),
        optimizer=optimizer(
            model_config.training.optimizer, lr=model_config.training.learning_rate
        ),
        epochs=model_config.training.epochs,
    )

    return model, params, get_loss(model_config.loss)


@pytest.fixture(scope="session")
def model_context(
    dataset: Dataset, model_params_loss: Tuple[ApproximationModel, Dict, Callable]
) -> ModelContext:
    """Create a ModelContext for reference computations."""
    model, params, loss = model_params_loss

    return ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss,
    )


@pytest.fixture(scope="session")
def fim_data_model_context_with_pseudo_targets(
    dataset: Dataset,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
) -> Tuple[Tuple[DataActivationsGradients, DataActivationsGradients], ModelContext]:
    """Collect FIM data with pseudo targets and create reference context."""
    model, params, loss = model_params_loss

    # Generate pseudo targets for FIM computation
    pseudo_targets = generate_pseudo_targets(
        model=model,
        inputs=dataset.inputs,
        params=params,
        loss_fn=loss,
        rng_key=PRNGKey(1234),
    )

    # Collect activations and gradients
    collector = CollectorActivationsGradients(model=model, params=params, loss_fn=loss)

    fim_data = collector.collect(
        inputs=dataset.inputs,
        targets=pseudo_targets,
    )

    assert isinstance(fim_data, DataActivationsGradients)

    return (fim_data, DataActivationsGradients()), ModelContext.create(
        dataset=Dataset(inputs=dataset.inputs, targets=pseudo_targets),
        model=model,
        params=params,
        loss_fn=loss,
    )


# -----------------------------------------------------------------------------
# Reference implementation using autodiff
# -----------------------------------------------------------------------------


def reference_fim_from_autodiff(model_context: ModelContext, damping: float = 0.0):
    """
    Naive reference Fisher Information Matrix using explicit per-sample gradients.

    F = (1/N) * sum_i grad_i grad_i^T + damping * I
    """

    def loss_single(p_flat, x, y):
        params = model_context.unravel_fn(p_flat)
        preds = model_context.model_apply_fn(params, x)
        loss = model_context.loss_fn(preds, y, reduction="sum")
        return loss

    grad_fn = jax.grad(loss_single)

    p_flat = model_context.params_flat
    X = model_context.inputs
    Y = model_context.targets

    # (N, n_params)
    grads = jax.vmap(lambda x, y: grad_fn(p_flat, x, y))(X, Y)

    # Apply MSE scaling correction if needed
    if get_loss_name(model_context.loss_fn) == "mse":
        grads = -0.5 * grads

    F = grads.T @ grads / grads.shape[0]
    F = F + damping * jnp.eye(F.shape[0], dtype=F.dtype)
    return F


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_fim_matrix_matches_reference(
    fim_data_model_context_with_pseudo_targets: Tuple[
        Tuple[DataActivationsGradients, DataActivationsGradients], ModelContext
    ],
):
    """Full FIM matrix must match autodiff reference."""

    fim_data = fim_data_model_context_with_pseudo_targets[0]
    model_context = fim_data_model_context_with_pseudo_targets[1]

    fim = FIMComputer(compute_context=fim_data).build()

    F_collector = fim.estimate_hessian(damping=0.0)
    F_ref = reference_fim_from_autodiff(model_context, damping=0.0)

    rel_err = FullMatrixMetric.RELATIVE_FROBENIUS.compute(F_collector, F_ref)
    assert rel_err < 1e-4, f"FIM relative error too large: {rel_err}"


def test_fim_hvp_matches_reference(
    config: Dict,
    fim_data_model_context_with_pseudo_targets: Tuple[
        Tuple[DataActivationsGradients, DataActivationsGradients], ModelContext
    ],
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """FIM-vector product must match autodiff reference."""

    fim_data = fim_data_model_context_with_pseudo_targets[0]
    model_context = fim_data_model_context_with_pseudo_targets[1]
    fim = FIMComputer(compute_context=fim_data).build()

    _, params, _ = model_params_loss

    params_flat, _ = flatten_util.ravel_pytree(params)
    v1 = jnp.ones_like(params_flat)
    v2 = jax.random.normal(jax.random.PRNGKey(0), shape=params_flat.shape)

    F = reference_fim_from_autodiff(model_context, damping=0.0)

    def ref_hvp(v, damping: float):
        return F @ v + damping * v

    fim_hvp_1 = fim.estimate_hvp(v1, damping=0.1)
    ref_hvp_1 = ref_hvp(v1, damping=0.1)

    fim_hvp_2 = fim.estimate_hvp(v2, damping=0.1)
    ref_hvp_2 = ref_hvp(v2, damping=0.1)

    assert (
        VectorMetric.RELATIVE_ERROR.compute(fim_hvp_1, ref_hvp_1) < config["tol_hvp"]
    ), f"HVP error (ones): {VectorMetric.RELATIVE_ERROR.compute(fim_hvp_1, ref_hvp_1)}"

    assert (
        VectorMetric.RELATIVE_ERROR.compute(fim_hvp_2, ref_hvp_2) < config["tol_hvp"]
    ), (
        f"HVP error (random): {VectorMetric.RELATIVE_ERROR.compute(fim_hvp_2, ref_hvp_2)}"
    )


def test_fim_ihvp_matches_reference(
    config: Dict,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    fim_data_model_context_with_pseudo_targets: Tuple[
        Tuple[DataActivationsGradients, DataActivationsGradients], ModelContext
    ],
):
    """Inverse FIM-vector product must match direct solve reference."""
    _, params, _ = model_params_loss

    fim = FIMComputer(
        compute_context=fim_data_model_context_with_pseudo_targets[0]
    ).build()

    params_flat, _ = flatten_util.ravel_pytree(params)
    v = jax.random.normal(jax.random.PRNGKey(1), shape=params_flat.shape)

    F_ref = reference_fim_from_autodiff(
        fim_data_model_context_with_pseudo_targets[1], damping=0.1
    )

    fim_ihvp = fim.estimate_ihvp(v, damping=0.1)
    ref_ihvp = jnp.linalg.solve(F_ref, v)

    rel_error = VectorMetric.RELATIVE_ERROR.compute(fim_ihvp, ref_ihvp)
    assert rel_error < config["tol_ihvp"], (
        f"IHVP relative error too high: {rel_error:. 4f} >= {config['tol_ihvp']}"
    )
