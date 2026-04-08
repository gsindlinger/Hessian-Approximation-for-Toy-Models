from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util

from src.config import (
    PseudoTargetGenerationStrategy,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.fim import FIMComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.utils.data.data import Dataset
from src.utils.loss import get_loss_name
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
from tests._helpers import (
    cached_train_model_for_dataset,
    create_model_context,
)
from tests.conftest import TrainingScenario

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("fim_classification_scenario", id="classification"),
        pytest.param("fim_regression_scenario", id="regression"),
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
    """Create a ModelContext for reference computations."""
    return create_model_context(dataset, model_params_loss)


@pytest.fixture(scope="session")
def fim_data_model_context_without_pseudo_targets(
    dataset: Dataset,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
) -> Tuple[DataActivationsGradients, ModelContext]:
    """Collect FIM data with pseudo targets and create reference context."""
    model, params, loss = model_params_loss

    # Collect activations and gradients
    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss,
        pseudo_target_strategy=PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
    )

    fim_data = collector.collect(dataset=dataset)

    assert isinstance(fim_data, DataActivationsGradients)

    return fim_data, ModelContext.create(
        dataset=Dataset(inputs=dataset.inputs, targets=dataset.targets),
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
    # log p(y|x) = -1/2 * ||y - f(x)||^2, L_n = (1/D)*||y-f||^2 => grad_log_p = -(D/2) * grad_L
    if get_loss_name(model_context.loss_fn) == "mse":
        assert Y is not None
        output_dim = Y.shape[-1]
        grads = -(output_dim / 2.0) * grads

    F = grads.T @ grads / grads.shape[0]
    F = F + damping * jnp.eye(F.shape[0], dtype=F.dtype)
    return F


def reference_from_autodiff_all_classes(
    model_context: ModelContext,
    dataset: Dataset,
    damping: float = 0.0,
):
    """
    Compute true FIM by summing over all classes weighted by predicted probabilities.
    FIM = E_x[Σ_c p(c|x) * g_c g_c^T]
    """

    assert get_loss_name(model_context.loss_fn) == "cross_entropy", (
        "Reference with all classes is only implemented for classification."
    )

    def loss_single(p_flat, x, y):
        params = model_context.unravel_fn(p_flat)
        preds = model_context.model_apply_fn(params, x)
        loss = model_context.loss_fn(preds, y, reduction="sum")
        return loss

    grad_fn = jax.grad(loss_single)

    p_flat = model_context.params_flat
    X = model_context.inputs
    n_classes = dataset.output_dim()
    n_params = p_flat.shape[0]

    def per_sample_fim(x):
        """Compute FIM contribution for single sample x"""
        preds = model_context.model_apply_fn(model_context.unravel_fn(p_flat), x)
        probs = jax.nn.softmax(preds)

        # Initialize FIM for this sample
        fim_x = jnp.zeros((n_params, n_params), dtype=p_flat.dtype)

        for c in range(n_classes):
            # Pass class index directly (not one-hot) for integer label loss
            grad_c = grad_fn(p_flat, x, c)

            # Accumulate: p(c|x) * g_c g_c^T
            fim_x += probs[c] * jnp.outer(grad_c, grad_c)

        return fim_x

    # Average FIM over all samples
    fim_per_sample = jax.vmap(per_sample_fim)(X)  # (N, n_params, n_params)
    F = jnp.mean(fim_per_sample, axis=0)
    F = F + damping * jnp.eye(F.shape[0], dtype=F.dtype)

    return F


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_fim_matrix_matches_reference(
    fim_data_model_context_without_pseudo_targets: Tuple[
        DataActivationsGradients, ModelContext
    ],
):
    """Full FIM matrix must match autodiff reference."""

    fim_data = fim_data_model_context_without_pseudo_targets[0]
    model_context = fim_data_model_context_without_pseudo_targets[1]

    fim = FIMComputer(compute_context=fim_data).build()

    F_collector = fim.estimate_hessian(damping=0.0)
    F_ref = reference_fim_from_autodiff(model_context, damping=0.0)
    assert FullMatrixMetric.RELATIVE_FROBENIUS.compute(F_collector, F_ref) < 1e-5, (
        "FIM matrix does not match reference."
    )


def test_fim_hvp_matches_reference(
    fim_data_model_context_without_pseudo_targets: Tuple[
        DataActivationsGradients, ModelContext
    ],
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """FIM-vector product must match autodiff reference."""

    fim_data = fim_data_model_context_without_pseudo_targets[0]
    model_context = fim_data_model_context_without_pseudo_targets[1]
    fim = FIMComputer(compute_context=fim_data).build()

    _, params, _ = model_params_loss

    params_flat, _ = flatten_util.ravel_pytree(params)
    v1 = jnp.ones_like(params_flat)
    v2 = jax.random.normal(jax.random.PRNGKey(0), shape=params_flat.shape)

    F = reference_fim_from_autodiff(model_context, damping=0.0)

    def ref_hvp(v, damping: float):
        return F @ v + damping * v

    fim_hvp_1 = fim.estimate_hvp(v1)
    ref_hvp_1 = ref_hvp(v1, damping=0.0)

    fim_hvp_2 = fim.estimate_hvp(v2)
    ref_hvp_2 = ref_hvp(v2, damping=0.0)

    assert VectorMetric.RELATIVE_ERROR.compute(fim_hvp_1, ref_hvp_1) < 1e-5, (
        "FIM HVP (v1) does not match reference."
    )
    assert VectorMetric.RELATIVE_ERROR.compute(fim_hvp_2, ref_hvp_2) < 1e-5, (
        "FIM HVP (v2) does not match reference."
    )


def test_fim_ihvp_matches_reference(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    fim_data_model_context_without_pseudo_targets: Tuple[
        DataActivationsGradients, ModelContext
    ],
):
    """Inverse FIM-vector product must match direct solve reference."""
    _, params, _ = model_params_loss

    fim = FIMComputer(
        compute_context=fim_data_model_context_without_pseudo_targets[0]
    ).build()

    params_flat, _ = flatten_util.ravel_pytree(params)
    v = jax.random.normal(jax.random.PRNGKey(1), shape=params_flat.shape)

    F_ref = reference_fim_from_autodiff(
        fim_data_model_context_without_pseudo_targets[1], damping=0.0
    )

    eigenvals, _ = jnp.linalg.eigh(F_ref)
    damping = 0.1 * jnp.mean(eigenvals[eigenvals > 0])
    F_ref = F_ref + damping * jnp.eye(F_ref.shape[0], dtype=F_ref.dtype)

    fim_ihvp = fim.estimate_ihvp(v, damping=damping)
    ref_ihvp = jnp.linalg.solve(F_ref, v)

    assert VectorMetric.RELATIVE_ERROR.compute(fim_ihvp, ref_ihvp) < 1e-4, (
        "FIM IHVP does not match reference."
    )


def test_fim_with_pseudo_targets_for_all_classes(
    training_scenario: TrainingScenario,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
):
    """FIM computed with pseudo-targets for all classes must match direct autodiff reference."""
    if training_scenario.model_config.loss.value != "cross_entropy":
        pytest.skip("This check only applies to the classification FIM scenario.")

    model, params, loss = model_params_loss

    # Collect activations and gradients
    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss,
        pseudo_target_strategy=PseudoTargetGenerationStrategy.ALL_CLASSES,
    )

    fim_data = collector.collect(
        dataset=dataset,
    )

    assert isinstance(fim_data, DataActivationsGradients)

    fim = FIMComputer(compute_context=fim_data).build()

    F_collector_single = fim.estimate_hessian(damping=0.0)
    F_collector_all_classes = reference_from_autodiff_all_classes(
        ModelContext.create(
            dataset=Dataset(inputs=dataset.inputs, targets=dataset.targets),
            model=model,
            params=params,
            loss_fn=loss,
        ),
        dataset,
        damping=0.0,
    )

    assert jnp.allclose(F_collector_single, F_collector_all_classes, atol=1e-5), (
        "FIM matrix with all classes as pseudo-targets does not match reference."
    )
