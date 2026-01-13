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
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.utils.data import ModelContext
from src.utils.data.data import (
    Dataset,
    RandomClassificationDataset,
    RandomRegressionDataset,
)
from src.utils.loss import get_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(
    params=["random_regression", "classification"],
    scope="session",
)
def config(request, tmp_path_factory):
    """Create model configuration for testing."""
    base = tmp_path_factory.mktemp(request.param)

    if request.param == "random_regression":
        architecture = ModelArchitecture.LINEAR
        hidden_dim = []
        loss = LossType.MSE
        tol_matrix = 1e-4
        tol_vector = 1e-5
    else:  # classification
        architecture = ModelArchitecture.LINEAR
        hidden_dim = []
        loss = LossType.CROSS_ENTROPY
        tol_matrix = 2e-2
        tol_vector = 2e-1

    model_config = ModelConfig(
        architecture=architecture,
        input_dim=10,  # Will be updated from dataset
        hidden_dim=hidden_dim if hidden_dim else None,
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
        "tol_matrix": tol_matrix,
        "tol_vector": tol_vector,
    }


@pytest.fixture(scope="session")
def dataset(config: Dict) -> Dataset:
    """Create a random dataset for testing (classification or regression)."""
    seed = 42

    if config["model_config"].loss == LossType.MSE:
        return RandomRegressionDataset(
            n_samples=1200,
            n_features=100,
            n_targets=10,
            noise=20.0,
            seed=seed,
        )
    else:  # CROSS_ENTROPY / classification
        return RandomClassificationDataset(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_classes=5,
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
        dataset.get_dataloader(batch_size=model_config.training.batch_size, seed=42),
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
    """Create a ModelContext for Hessian/GNH computation."""
    model, params, loss = model_params_loss

    return ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_exact_hessian_vs_gnh_matrix_equivalence(
    config: Dict,
    model_context: ModelContext,
):
    """For linear models, exact Hessian should closely match GNH."""
    hessian = HessianComputer(compute_context=model_context)
    gnh = GNHComputer(compute_context=model_context).build()

    H = hessian.compute_hessian()
    G = gnh.estimate_hessian()

    diff_fro = FullMatrixMetric.RELATIVE_FROBENIUS.compute(H, G)
    assert diff_fro < config["tol_matrix"], (
        f"Hessian vs GNH matrix difference too large: {diff_fro:. 6f} >= {config['tol_matrix']}"
    )


def test_batched_ihvp_matches_full_inverse(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """Test that batched IHVP matches explicit inverse multiplication."""
    hessian = HessianComputer(compute_context=model_context)
    _, params, _ = model_params_loss
    params_flat, _ = flatten_util.ravel_pytree(params)

    V = jax.random.normal(PRNGKey(0), shape=(10, params_flat.shape[0]))

    IHVP = hessian.compute_ihvp(V, damping=1e-2)
    H = hessian.compute_hessian(damping=1e-2)
    Hinv = jnp.linalg.inv(H)

    IHVP_ref = (Hinv @ V.T).T

    err = VectorMetric.RELATIVE_ERROR.compute(IHVP, IHVP_ref, reduction="mean")
    assert err < 1e-5, f"Batched IHVP error:  {err:.6e}"


def test_batched_hvp_matches_full_hessian(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """Test that batched HVP matches explicit Hessian multiplication."""
    hessian = HessianComputer(compute_context=model_context)
    _, params, _ = model_params_loss
    params_flat, _ = flatten_util.ravel_pytree(params)

    V = jax.random.normal(PRNGKey(1), shape=(10, params_flat.shape[0]))
    HVP = hessian.compute_hvp(V, damping=1e-2)

    H = hessian.compute_hessian(damping=1e-2)
    HVP_ref = (H @ V.T).T

    err = VectorMetric.RELATIVE_ERROR.compute(HVP, HVP_ref, reduction="mean")
    assert err < 1e-5, f"Batched HVP error: {err:.6e}"


def test_hessian_ihvp_roundtrip_unit_vectors(
    model_context: ModelContext,
):
    """Test that IHVP on identity matrix gives inverse Hessian."""
    hessian = HessianComputer(compute_context=model_context)

    n_params = model_context.params_flat.size
    I = jnp.eye(n_params)

    IHVP = hessian.compute_ihvp(I, damping=1e-2)
    Hinv = jnp.linalg.inv(hessian.compute_hessian(damping=1e-2))

    diff = jnp.max(jnp.abs(Hinv - IHVP.T))
    assert diff < 1e-6, f"IHVP roundtrip error: {diff:.6e}"


def test_hessian_vs_gnh_ihvp_consistency(
    config: Dict,
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """Test that Hessian and GNH IHVP implementations are consistent."""
    hessian = HessianComputer(compute_context=model_context)
    gnh = GNHComputer(compute_context=model_context).build()

    _, params, _ = model_params_loss
    params_flat, _ = flatten_util.ravel_pytree(params)

    V = jax.random.normal(PRNGKey(2), shape=(10, params_flat.shape[0]))

    ihvp_h = hessian.compute_ihvp(V, damping=1e-3)
    ihvp_g = gnh.estimate_ihvp(V, damping=1e-3)

    err = VectorMetric.RELATIVE_ERROR.compute(ihvp_h, ihvp_g, reduction="mean")
    assert err < config["tol_vector"], (
        f"Hessian vs GNH IHVP inconsistency: {err:.6e} >= {config['tol_vector']}"
    )
