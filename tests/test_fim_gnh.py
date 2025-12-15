import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from jax.random import PRNGKey

from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.utils.data import ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets
from src.utils.data.data import (
    RandomClassificationDataset,
    RandomRegressionDataset,
)
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import cross_entropy_loss, mse_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.linear_model import LinearModel
from src.utils.models.mlp import MLP
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="session", params=["classification", "regression"])
def setup(request):
    seed = 0

    if request.param == "classification":
        dataset = RandomClassificationDataset(
            n_samples=1500,
            n_features=20,
            n_informative=10,
            n_classes=5,
            seed=seed,
        )
        model = MLP(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=[10],
            activation="relu",
            seed=seed,
        )
        loss_fn = cross_entropy_loss
        tol_hvp = 0.2
        tol_ihvp = 0.1
        tol_cosine = 0.95
    else:
        dataset = RandomRegressionDataset(
            n_samples=1500,
            n_features=10,
            n_targets=2,
            noise=5.0,
            seed=seed,
        )
        model = LinearModel(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=[5],
            seed=seed,
        )
        loss_fn = mse_loss
        tol_hvp = 0.6
        tol_ihvp = 0.1
        tol_cosine = 0.95

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

    pseudo_targets = generate_pseudo_targets(
        model=model,
        inputs=dataset.inputs,
        params=params,
        loss_fn=loss_fn,
        rng_key=PRNGKey(seed + 1234),
    )
    model_context_fim = ModelContext.create(
        dataset=dataset.replace_targets(pseudo_targets),
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    return {
        "dataset": dataset,
        "model": model,
        "params": params,
        "model_context": model_context,
        "fim_model_context": model_context_fim,
        "loss_fn": loss_fn,
        "tol_hvp": tol_hvp,
        "tol_ihvp": tol_ihvp,
        "tol_cosine": tol_cosine,
    }


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_fim_gnh_hessian_similarity(setup):
    fim = FIMComputer(setup["fim_model_context"])
    gnh = GNHComputer(setup["model_context"])

    H_fim = fim.estimate_hessian(damping=0.0)
    H_gnh = gnh.estimate_hessian(damping=0.0)

    sim = FullMatrixMetric.COSINE_SIMILARITY.compute(H_fim, H_gnh)
    assert sim > setup["tol_cosine"], f"Cosine similarity too low: {sim}"


def test_fim_gnh_hvp_consistency(setup):
    fim = FIMComputer(setup["fim_model_context"])
    gnh = GNHComputer(setup["model_context"])

    params_flat, _ = flatten_util.ravel_pytree(setup["params"])

    v = jnp.ones_like(params_flat)
    v_rand = jax.random.normal(PRNGKey(0), shape=params_flat.shape)

    fim_hvp = fim.estimate_hvp(v, damping=0.1)
    gnh_hvp = gnh.estimate_hvp(v, damping=0.1)
    assert VectorMetric.RELATIVE_ERROR.compute(fim_hvp, gnh_hvp) < setup["tol_hvp"]

    fim_hvp_r = fim.estimate_hvp(v_rand, damping=0.1)
    gnh_hvp_r = gnh.estimate_hvp(v_rand, damping=0.1)
    assert VectorMetric.RELATIVE_ERROR.compute(fim_hvp_r, gnh_hvp_r) < setup["tol_hvp"]


def test_fim_gnh_ihvp_consistency(setup):
    fim = FIMComputer(setup["fim_model_context"])
    gnh = GNHComputer(setup["model_context"])

    params_flat, _ = flatten_util.ravel_pytree(setup["params"])

    v = jnp.ones_like(params_flat)
    v_rand = jax.random.normal(PRNGKey(1), shape=params_flat.shape)

    fim_ihvp = fim.estimate_ihvp(v, damping=0.1)
    gnh_ihvp = gnh.estimate_ihvp(v, damping=0.1)
    assert VectorMetric.RELATIVE_ERROR.compute(fim_ihvp, gnh_ihvp) < setup["tol_ihvp"]

    fim_ihvp_r = fim.estimate_ihvp(v_rand, damping=0.1)
    gnh_ihvp_r = gnh.estimate_ihvp(v_rand, damping=0.1)
    assert (
        VectorMetric.RELATIVE_ERROR.compute(fim_ihvp_r, gnh_ihvp_r) < setup["tol_ihvp"]
    )
