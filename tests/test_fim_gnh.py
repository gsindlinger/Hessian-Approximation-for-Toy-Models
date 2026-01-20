from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest
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
from src.hessians.computer.gnh import GNHComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets
from src.utils.data.data import (
    Dataset,
    RandomClassificationDataset,
    RandomRegressionDataset,
)
from src.utils.loss import get_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
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
    model_config: ModelConfig = config["model_config"]

    # Update dimensions from dataset
    model_config.input_dim = dataset.input_dim()
    model_config.output_dim = dataset.output_dim()

    # Train the model
    model, params, _ = train_model(
        model_config=model_config,
        dataloader=dataset.get_dataloader(
            batch_size=model_config.training.batch_size, seed=0
        ),
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
    """Create a ModelContext for GNH computation."""
    model, params, loss = model_params_loss

    return ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss,
    )


@pytest.fixture(scope="session")
def fim_data(
    dataset: Dataset,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
) -> DataActivationsGradients:
    """Collect FIM data with pseudo targets."""
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
        save_directory=None,
    )

    assert isinstance(fim_data, DataActivationsGradients)
    return fim_data


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_fim_gnh_hessian_similarity(
    config: Dict,
    model_context: ModelContext,
    fim_data: DataActivationsGradients,
):
    """Test that FIM and GNH Hessian matrices are similar."""
    fim = FIMComputer(compute_context=(fim_data, DataActivationsGradients())).build()
    gnh = GNHComputer(compute_context=model_context).build()

    H_fim = fim.estimate_hessian(damping=0.0)
    H_gnh = gnh.estimate_hessian(damping=0.0)

    sim = FullMatrixMetric.COSINE_SIMILARITY.compute(H_fim, H_gnh)
    assert sim > config["tol_cosine"], (
        f"Cosine similarity too low: {sim:.4f} < {config['tol_cosine']}"
    )


def test_fim_gnh_hvp_consistency(
    config: Dict,
    model_context: ModelContext,
    fim_data: DataActivationsGradients,
):
    """Test that FIM and GNH HVP operations are consistent."""
    fim = FIMComputer(compute_context=(fim_data, DataActivationsGradients())).build()
    gnh = GNHComputer(compute_context=model_context).build()

    params_flat = model_context.params_flat

    # Test with ones vector
    v_ones = jnp.ones_like(params_flat)
    fim_hvp = fim.estimate_hvp(v_ones, damping=0.1)
    gnh_hvp = gnh.estimate_hvp(v_ones, damping=0.1)

    rel_error = VectorMetric.RELATIVE_ERROR.compute(fim_hvp, gnh_hvp)
    assert rel_error < config["tol_hvp"], (
        f"HVP relative error too high (ones): {rel_error:.4f} >= {config['tol_hvp']}"
    )

    # Test with random vector
    v_rand = jax.random.normal(PRNGKey(0), shape=params_flat.shape)
    fim_hvp_r = fim.estimate_hvp(v_rand, damping=0.1)
    gnh_hvp_r = gnh.estimate_hvp(v_rand, damping=0.1)

    rel_error_r = VectorMetric.RELATIVE_ERROR.compute(fim_hvp_r, gnh_hvp_r)
    assert rel_error_r < config["tol_hvp"], (
        f"HVP relative error too high (random): {rel_error_r:.4f} >= {config['tol_hvp']}"
    )


def test_fim_gnh_ihvp_consistency(
    config: Dict,
    model_context: ModelContext,
    fim_data: DataActivationsGradients,
):
    """Test that FIM and GNH IHVP operations are consistent."""
    fim = FIMComputer(compute_context=(fim_data, DataActivationsGradients())).build()
    gnh = GNHComputer(compute_context=model_context).build()

    params_flat = model_context.params_flat

    # Test with ones vector
    v_ones = jnp.ones_like(params_flat)
    fim_ihvp = fim.estimate_ihvp(v_ones, damping=0.1)
    gnh_ihvp = gnh.estimate_ihvp(v_ones, damping=0.1)

    rel_error = VectorMetric.RELATIVE_ERROR.compute(fim_ihvp, gnh_ihvp)
    assert rel_error < config["tol_ihvp"], (
        f"IHVP relative error too high (ones): {rel_error:.4f} >= {config['tol_ihvp']}"
    )

    # Test with random vector
    v_rand = jax.random.normal(PRNGKey(1), shape=params_flat.shape)
    fim_ihvp_r = fim.estimate_ihvp(v_rand, damping=0.1)
    gnh_ihvp_r = gnh.estimate_ihvp(v_rand, damping=0.1)

    rel_error_r = VectorMetric.RELATIVE_ERROR.compute(fim_ihvp_r, gnh_ihvp_r)
    assert rel_error_r < config["tol_ihvp"], (
        f"IHVP relative error too high (random): {rel_error_r:.4f} >= {config['tol_ihvp']}"
    )
