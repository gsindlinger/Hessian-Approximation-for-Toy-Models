import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from jax.random import PRNGKey

from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets
from src.utils.data.data import (
    Dataset,
    RandomClassificationDataset,
    RandomRegressionDataset,
)
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import cross_entropy_loss, get_loss_name, mse_loss
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

    pseudo_targets = generate_pseudo_targets(
        model=model,
        inputs=dataset.inputs,
        params=params,
        loss_fn=loss_fn,
        rng_key=PRNGKey(seed + 1234),
    )

    model_context = ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    model_context_fim_reference = ModelContext.create(
        dataset=Dataset(inputs=dataset.inputs, targets=pseudo_targets),
        model=model,
        params=params,
        loss_fn=loss_fn,
    )

    activation_gradients_collector = CollectorActivationsGradients(
        model=model, params=params, loss_fn=loss_fn
    )
    fim_data = activation_gradients_collector.collect(
        inputs=dataset.inputs,
        targets=pseudo_targets,
        save_directory=None,
    )

    assert isinstance(fim_data, DataActivationsGradients)

    return {
        "dataset": dataset,
        "model": model,
        "params": params,
        "model_context": model_context,
        "fim_model_context": fim_data,
        "model_context_fim_reference": model_context_fim_reference,
        "loss_fn": loss_fn,
        "tol_hvp": tol_hvp,
        "tol_ihvp": tol_ihvp,
        "tol_cosine": tol_cosine,
    }


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

    # Remove this line - rescaling happens in the loss now
    if get_loss_name(model_context.loss_fn) == "mse":
        grads = -0.5 * grads

    F = grads.T @ grads / grads.shape[0]
    F = F + damping * jnp.eye(F.shape[0], dtype=F.dtype)
    return F


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_fim_matrix_matches_reference(setup):
    """Full FIM matrix must match autodiff reference."""
    fim = FIMComputer(setup["fim_model_context"])

    F_collector = fim.estimate_hessian(damping=0.0)
    F_ref = reference_fim_from_autodiff(
        setup["model_context_fim_reference"], damping=0.0
    )
    hessian = HessianComputer(setup["model_context"]).compute_hessian(damping=0.0)

    # plot heatmaps for debugging
    import matplotlib.pyplot as plt

    plt.subplot(1, 3, 1)
    plt.title("FIM Collector")
    plt.imshow(F_collector, cmap="viridis")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("FIM Reference")
    plt.imshow(F_ref, cmap="viridis")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("Hessian Reference")
    plt.imshow(hessian, cmap="viridis")
    plt.colorbar()
    plt.show()

    rel_err = FullMatrixMetric.RELATIVE_FROBENIUS.compute(F_collector, F_ref)
    assert rel_err < 1e-5, f"FIM relative error too large: {rel_err}"


def test_fim_hvp_matches_reference(setup):
    """FIM-vector product must match autodiff reference."""
    fim = FIMComputer(setup["fim_model_context"])

    params_flat, _ = flatten_util.ravel_pytree(setup["params"])
    v1 = jnp.ones_like(params_flat)
    v2 = jax.random.normal(jax.random.PRNGKey(0), shape=params_flat.shape)

    F = reference_fim_from_autodiff(setup["model_context_fim_reference"], damping=0.0)

    def ref_hvp(v, damping: float):
        return F @ v + damping * v

    fim_hvp_1 = fim.estimate_hvp(v1, damping=0.1)
    ref_hvp_1 = ref_hvp(v1, damping=0.1)

    fim_hvp_2 = fim.estimate_hvp(v2, damping=0.1)
    ref_hvp_2 = ref_hvp(v2, damping=0.1)

    assert VectorMetric.RELATIVE_ERROR.compute(fim_hvp_1, ref_hvp_1) < setup["tol_hvp"]
    assert VectorMetric.RELATIVE_ERROR.compute(fim_hvp_2, ref_hvp_2) < setup["tol_hvp"]


def test_fim_ihvp_matches_reference(setup):
    """Inverse FIM-vector product must match direct solve reference."""
    fim = FIMComputer(setup["fim_model_context"])

    params_flat, _ = flatten_util.ravel_pytree(setup["params"])
    v = jax.random.normal(jax.random.PRNGKey(1), shape=params_flat.shape)

    F_ref = reference_fim_from_autodiff(
        setup["model_context_fim_reference"], damping=0.1
    )

    fim_ihvp = fim.estimate_ihvp(v, damping=0.1)
    ref_ihvp = jnp.linalg.solve(F_ref, v)

    assert VectorMetric.RELATIVE_ERROR.compute(fim_ihvp, ref_ihvp) < setup["tol_ihvp"]
