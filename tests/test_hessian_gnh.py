import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from jax.random import PRNGKey

from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.utils.data import ModelContext
from src.utils.data.data import (
    RandomClassificationDataset,
    RandomRegressionDataset,
)
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import cross_entropy_loss, mse_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.linear_model import LinearModel
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(
    scope="session",
    params=["random_regression", "classification"],
)
def setup(request):
    seed = 42

    if request.param == "random_regression":
        dataset = RandomRegressionDataset(
            n_samples=1200,
            n_features=100,
            n_targets=10,
            noise=20.0,
            seed=seed,
        )
        model = LinearModel(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=[],
            seed=seed,
        )
        loss_fn = mse_loss
        tol_matrix = 1e-4
        tol_vector = 1e-5

    elif request.param == "classification":
        dataset = RandomClassificationDataset(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_classes=5,
            seed=seed,
        )
        model = LinearModel(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=[],
            seed=seed,
        )
        loss_fn = cross_entropy_loss
        tol_matrix = 2e-2
        tol_vector = 2e-1

    model, params, _ = train_model(
        model,
        dataset.get_dataloader(batch_size=JAXDataLoader.get_batch_size(), seed=seed),
        loss_fn=loss_fn,
        optimizer=optimizer("sgd", lr=1e-3),
        epochs=50,
    )

    model_context = ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    return {
        "dataset": dataset,
        "model": model,
        "params": params,
        "model_context": model_context,
        "tol_matrix": tol_matrix,
        "tol_vector": tol_vector,
    }


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_exact_hessian_vs_gnh_matrix_equivalence(setup):
    """For linear regression, exact Hessian == GNH. For"""
    hessian = HessianComputer(setup["model_context"])
    gnh = GNHComputer(setup["model_context"])

    H = hessian.compute_hessian()
    G = gnh.estimate_hessian()

    diff_fro = FullMatrixMetric.RELATIVE_FROBENIUS.compute(H, G)
    assert diff_fro < setup["tol_matrix"]


def test_batched_ihvp_matches_full_inverse(setup):
    hessian = HessianComputer(setup["model_context"])
    params_flat, _ = flatten_util.ravel_pytree(setup["params"])

    V = jax.random.normal(PRNGKey(0), shape=(10, params_flat.shape[0]))

    IHVP = hessian.compute_ihvp(V, damping=1e-2)
    H = hessian.compute_hessian(damping=1e-2)
    Hinv = jnp.linalg.inv(H)

    IHVP_ref = (Hinv @ V.T).T

    err = VectorMetric.RELATIVE_ERROR.compute(IHVP, IHVP_ref, reduction="mean")
    assert err < 1e-5


def test_batched_hvp_matches_full_hessian(setup):
    hessian = HessianComputer(setup["model_context"])
    params_flat, _ = flatten_util.ravel_pytree(setup["params"])

    V = jax.random.normal(PRNGKey(1), shape=(10, params_flat.shape[0]))
    HVP = hessian.compute_hvp(V, damping=1e-2)

    H = hessian.compute_hessian(damping=1e-2)
    HVP_ref = (H @ V.T).T

    err = VectorMetric.RELATIVE_ERROR.compute(HVP, HVP_ref, reduction="mean")
    assert err < 1e-5


def test_hessian_ihvp_roundtrip_unit_vectors(setup):
    hessian = HessianComputer(setup["model_context"])

    n_params = setup["model_context"].params_flat.size
    I = jnp.eye(n_params)

    IHVP = hessian.compute_ihvp(I, damping=1e-2)
    Hinv = jnp.linalg.inv(hessian.compute_hessian(damping=1e-2))

    diff = jnp.max(jnp.abs(Hinv - IHVP.T))
    assert diff < 1e-6


def test_hessian_vs_gnh_ihvp_consistency(setup):
    hessian = HessianComputer(setup["model_context"])
    gnh = GNHComputer(setup["model_context"])

    params_flat, _ = flatten_util.ravel_pytree(setup["params"])

    V = jax.random.normal(PRNGKey(2), shape=(10, params_flat.shape[0]))

    ihvp_h = hessian.compute_ihvp(V, damping=1e-3)
    ihvp_g = gnh.estimate_ihvp(V, damping=1e-3)

    err = VectorMetric.RELATIVE_ERROR.compute(ihvp_h, ihvp_g, reduction="mean")
    assert err < setup["tol_vector"]
