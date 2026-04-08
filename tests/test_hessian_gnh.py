from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from jax.random import PRNGKey

from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.utils.data import ModelContext
from src.utils.data.data import Dataset
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
from tests.conftest import TrainingScenario
from tests._helpers import cached_train_model_for_dataset, create_model_context

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param(
            "hessian_gnh_random_regression_scenario",
            id="random_regression",
        ),
        pytest.param("hessian_gnh_classification_scenario", id="classification"),
    ],
    scope="session",
)
def training_scenario(request) -> TrainingScenario:
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def config(training_scenario: TrainingScenario) -> Dict:
    return {"model_config": training_scenario.model_config}


@pytest.fixture(scope="session")
def dataset(training_scenario: TrainingScenario) -> Dataset:
    """Create a random dataset for testing (classification or regression)."""
    return training_scenario.dataset


@pytest.fixture(scope="session")
def model_params_loss(
    trained_model_registry: Dict[Tuple, Tuple[ApproximationModel, Dict, Callable]],
    training_scenario: TrainingScenario,
) -> Tuple[ApproximationModel, Dict, Callable]:
    """Train a model and return it with its parameters and loss function."""
    return cached_train_model_for_dataset(
        training_scenario.model_config,
        training_scenario.dataset,
        trained_model_registry,
        seed=training_scenario.train_seed,
        shuffle=training_scenario.shuffle,
    )


@pytest.fixture(scope="session")
def model_context(
    dataset: Dataset, model_params_loss: Tuple[ApproximationModel, Dict, Callable]
) -> ModelContext:
    """Create a ModelContext for Hessian/GNH computation."""
    return create_model_context(dataset, model_params_loss)


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
    assert diff_fro < 1e-4, (
        f"Hessian vs GNH matrix difference too large: {diff_fro:. 6f} >= {1e-4}"
    )


def test_batched_ihvp_matches_full_inverse(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """Test that batched IHVP matches explicit inverse multiplication for the Hessian computation."""
    hessian = HessianComputer(compute_context=model_context)
    _, params, _ = model_params_loss
    params_flat, _ = flatten_util.ravel_pytree(params)

    V = jax.random.normal(PRNGKey(0), shape=(10, params_flat.shape[0]))

    IHVP = hessian.compute_ihvp(V, damping=1e-2)
    H = hessian.compute_hessian(damping=1e-2)
    Hinv = jnp.linalg.inv(H)

    IHVP_ref = (Hinv @ V.T).T

    err = VectorMetric.RELATIVE_ERROR.compute(IHVP, IHVP_ref)
    assert err < 1e-3, f"Batched IHVP error:  {err:.6e}"


def test_batched_hvp_matches_full_hessian(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """Test that batched HVP matches explicit Hessian multiplication for the Hessian computation."""
    hessian = HessianComputer(compute_context=model_context)
    _, params, _ = model_params_loss
    params_flat, _ = flatten_util.ravel_pytree(params)

    V = jax.random.normal(PRNGKey(1), shape=(10, params_flat.shape[0]))
    HVP = hessian.compute_hvp(V)

    H = hessian.compute_hessian()
    HVP_ref = (H @ V.T).T

    err = VectorMetric.RELATIVE_ERROR.compute(HVP, HVP_ref)
    assert err < 1e-3, f"Batched HVP error: {err:.6e}"


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
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """Test that Hessian and GNH IHVP implementations are close to equal for linear models."""
    hessian = HessianComputer(compute_context=model_context)
    gnh = GNHComputer(compute_context=model_context).build()

    _, params, _ = model_params_loss
    params_flat, _ = flatten_util.ravel_pytree(params)

    V = jax.random.normal(PRNGKey(2), shape=(10, params_flat.shape[0]))

    ihvp_h = hessian.compute_ihvp(V, damping=1e-2)
    ihvp_g = gnh.estimate_ihvp(V, damping=1e-2)

    err = VectorMetric.RELATIVE_ERROR.compute(ihvp_h, ihvp_g)
    assert err < 1e-3, f"Hessian vs GNH IHVP error: {err:.6e}"
