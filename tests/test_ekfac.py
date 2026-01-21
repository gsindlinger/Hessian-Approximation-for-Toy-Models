from typing import Callable, Dict, Optional, Tuple

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
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import (
    generate_pseudo_targets,
    sample_gradients,
)
from src.utils.data.data import Dataset, RandomClassificationDataset
from src.utils.loss import get_loss
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(params=["linear", "multi_layer"], scope="session")
def config(request, tmp_path_factory):
    """Create model configuration for testing."""
    base = tmp_path_factory.mktemp(request.param)

    if request.param == "linear":
        architecture = ModelArchitecture.LINEAR
        hidden_dim = []
    else:
        architecture = ModelArchitecture.MLP
        hidden_dim = [5]

    return ModelConfig(
        architecture=architecture,
        input_dim=10,  # Will be updated from dataset
        hidden_dim=hidden_dim if hidden_dim else None,
        output_dim=2,  # Will be updated from dataset
        loss=LossType.CROSS_ENTROPY,
        training=TrainingConfig(
            learning_rate=1e-3,
            optimizer=OptimizerType.SGD,
            epochs=30,
            batch_size=128,
        ),
        directory=str(base / "model"),
    )


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """Create a random classification dataset for testing."""
    return RandomClassificationDataset(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=2,
        seed=123,
    )


@pytest.fixture(scope="session")
def model_params_loss(
    config: ModelConfig, dataset: Dataset
) -> Tuple[ApproximationModel, Dict, Callable]:
    """Train a model and return it with its parameters and loss function."""
    # Update dimensions from dataset
    config.input_dim = dataset.input_dim()
    config.output_dim = dataset.output_dim()

    # Train the model
    model, params, _ = train_model(
        model_config=config,
        dataloader=dataset.get_dataloader(
            batch_size=config.training.batch_size, seed=123
        ),
        loss_fn=get_loss(config.loss),
        optimizer=optimizer(
            config.training.optimizer, lr=config.training.learning_rate
        ),
        epochs=config.training.epochs,
    )

    return model, params, get_loss(config.loss)


@pytest.fixture(scope="session")
def model_context(
    dataset: Dataset, model_params_loss: Tuple[ApproximationModel, Dict, Callable]
) -> ModelContext:
    """Create a ModelContext for Hessian computation."""
    model, params, loss = model_params_loss

    return ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss,
    )


def _collector_data(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    config: ModelConfig,
    dataset: Dataset,
    run_suffix: str,
    batch_size: Optional[int] = None,
    use_pseudo_targets: bool = True,
) -> Tuple[DataActivationsGradients, DataActivationsGradients]:
    """Collect EK-FAC data with two runs."""
    model, params, loss = model_params_loss

    base_dir = config.directory
    assert base_dir is not None, "Model directory must be set"

    run1_dir = f"{base_dir}/ekfac{run_suffix}/run1"
    run2_dir = f"{base_dir}/ekfac{run_suffix}/run2"

    collector_data = []

    # Generate pseudo targets for each run
    for run_idx, run_dir in enumerate([run1_dir, run2_dir]):
        if use_pseudo_targets:
            targets = generate_pseudo_targets(
                model=model,
                inputs=dataset.inputs,
                params=params,
                loss_fn=loss,
                rng_key=PRNGKey(run_idx),
            )
        else:
            targets = dataset.targets

        collector = CollectorActivationsGradients(
            model=model,
            params=params,
            loss_fn=loss,
        )

        collector_data_temp = collector.collect(
            inputs=dataset.inputs,
            targets=targets,
            save_directory=run_dir,
            batch_size=batch_size,
            try_load=True,
        )

        collector_data.append(collector_data_temp)

    return tuple(collector_data)


@pytest.fixture(scope="session")
def collector_data_single(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> Tuple[DataActivationsGradients, DataActivationsGradients]:
    """Collect EK-FAC data (single batch collection)."""
    return _collector_data(
        config=config,
        model_params_loss=model_params_loss,
        dataset=dataset,
        run_suffix="_single",
    )


@pytest.fixture(scope="session")
def collector_data_batched(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> Tuple[DataActivationsGradients, DataActivationsGradients]:
    """Collect EK-FAC data (batched collection)."""
    return _collector_data(
        config=config,
        model_params_loss=model_params_loss,
        dataset=dataset,
        run_suffix="_batched",
        batch_size=128,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def compute_full_implicit_matrices(
    computer: HessianEstimator, dim: int, damping: float
):
    """Helper method to compute full Hessian and Inverse Hessian matrices by applying hvp / ihvp estimation to basis vectors."""
    hvp_cols, ihvp_cols = [], []
    for i in range(dim):
        e = jnp.zeros(dim).at[i].set(1.0)
        hvp_cols.append(computer.estimate_hvp(e, damping))
        ihvp_cols.append(computer.estimate_ihvp(e, damping))
    return jnp.column_stack(hvp_cols), jnp.column_stack(ihvp_cols)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_ekfac_existence(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
):
    """Test if EKFAC Hessian approximation can be computed without errors."""

    collector_data = _collector_data(
        config=config,
        model_params_loss=model_params_loss,
        dataset=dataset,
        run_suffix="_existence_test",
    )

    ekfac_computer = EKFACComputer(compute_context=collector_data).build(
        base_directory=config.directory
    )

    H = ekfac_computer.estimate_hessian(damping=1e-3)
    assert jnp.isfinite(H).all()


def test_gradient_consistency(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    config: ModelConfig,
    dataset: Dataset,
):
    """
    Verify E-KFAC collector vs true gradients (sanity test).
    The following should hold for each linear layer l:

        ∇_{W_l} log p(y | x; θ) =  a_{l-1}^T s_l

    where:
        - a_{l-1} = activations of the previous layer
        - s_l = preactivation gradients = ∇_{W_l a_{l-1}} log p(y | x; θ)
    """
    model, params, loss = model_params_loss

    collector_data_single = _collector_data(
        config=config,
        model_params_loss=model_params_loss,
        dataset=dataset,
        run_suffix="_gradient_consistency",
        use_pseudo_targets=False,
    )

    def loss_fn_apply(p):
        return loss(model.apply(p, dataset.inputs), dataset.targets, reduction="sum")

    gt_grads = jax.grad(loss_fn_apply)(params)

    activations = collector_data_single[0].activations
    gradients = collector_data_single[0].gradients
    layer_names = collector_data_single[0].layer_names

    for i, (layer, gt) in enumerate(gt_grads["params"].items()):
        assert layer == layer_names[i], "Layer names do not match"
        W_grad = gt["kernel"]
        a, g = activations[layer], gradients[layer]

        ag = jnp.einsum("ni,no->io", a, g)
        assert jnp.allclose(W_grad, ag, atol=1e-4)

        kron = sum(jnp.kron(a[i], g[i]) for i in range(a.shape[0])) / a.shape[0]
        assert jnp.allclose(W_grad.reshape(-1) / a.shape[0], kron, atol=1e-4)


def test_kfac_hessian(
    model_context: ModelContext,
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether K-FAC Hessian can be computed without errors and matches dimensions of other methods."""
    damping = 0.1

    H = HessianComputer(compute_context=model_context).compute_hessian(damping=damping)
    G = (
        GNHComputer(compute_context=model_context)
        .build()
        .estimate_hessian(damping=damping)
    )
    K = (
        KFACComputer(compute_context=collector_data_single)
        .build()
        .estimate_hessian(damping=damping)
    )

    assert H.shape == G.shape == K.shape


def test_kfac_via_kron_equals_eigenvector_method(
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """This test verifies that the K-FAC Hessian computed via the Kronecker product of the covariances directly equals
    the K-FAC Hessian computed via the eigenvector method implemented in KFACComputer."""

    activations, gradients, layer_names = (
        collector_data_single[0].activations,
        collector_data_single[0].gradients,
        collector_data_single[0].layer_names,
    )

    # Compute covariances manually
    covariances_activations = {}
    covariances_gradients = {}

    for layer in layer_names:
        a = activations[layer]
        g = gradients[layer]

        covariances_activations[layer] = (a.T @ a) / a.shape[0]
        covariances_gradients[layer] = (g.T @ g) / g.shape[0]

    # Compute hessian by the block diagonal hessian of the kronecker product of the covariances
    H_kron_blocks = {}
    for layer in layer_names:
        A_cov = covariances_activations[layer]
        G_cov = covariances_gradients[layer]
        H_kron_blocks[layer] = jnp.kron(A_cov, G_cov)

    kron_comparison_method_H = jax.scipy.linalg.block_diag(*H_kron_blocks.values())

    # Use KFACComputer to compute the Hessian via eigenvalue decomposition
    kfac_computer = KFACComputer(compute_context=collector_data_single).build()
    eigenvector_method_H = kfac_computer.estimate_hessian(damping=0.0)

    assert jnp.allclose(
        kron_comparison_method_H,
        eigenvector_method_H,
        atol=1e-4,
    )


def test_ekfac_hessian(
    model_context: ModelContext,
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether EK-FAC Hessian can be computed and matches dimensions."""
    damping = 0.1

    H = HessianComputer(compute_context=model_context).compute_hessian(damping=damping)
    G = (
        GNHComputer(compute_context=model_context)
        .build()
        .estimate_hessian(damping=damping)
    )
    E = (
        EKFACComputer(compute_context=collector_data_single)
        .build()
        .estimate_hessian(damping=damping)
    )

    assert H.shape == G.shape == E.shape


def test_collector_batched_vs_single_consistency(
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
    collector_data_batched: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether EK-FAC data collected with different batch sizes is consistent."""

    ekfac_computer = EKFACComputer(
        compute_context=(collector_data_single[0], collector_data_single[1])
    ).build()

    ekfac_computer_batched = EKFACComputer(
        compute_context=(collector_data_batched[0], collector_data_batched[1])
    ).build()

    assert isinstance(ekfac_computer, EKFACComputer)
    assert isinstance(ekfac_computer_batched, EKFACComputer)

    for layer in ekfac_computer.compute_context[0].layer_names:
        A_single = ekfac_computer.precomputed_data.activation_eigenvectors[layer]
        G_single = ekfac_computer.precomputed_data.gradient_eigenvectors[layer]
        L_single = ekfac_computer.precomputed_data.eigenvalue_corrections[layer]

        A_batched = ekfac_computer_batched.precomputed_data.activation_eigenvectors[
            layer
        ]
        G_batched = ekfac_computer_batched.precomputed_data.gradient_eigenvectors[layer]
        L_batched = ekfac_computer_batched.precomputed_data.eigenvalue_corrections[
            layer
        ]

        assert jnp.allclose(A_single, A_batched, rtol=1e-6, atol=1e-5)
        assert jnp.allclose(G_single, G_batched, rtol=1e-6, atol=1e-5)
        assert jnp.allclose(L_single, L_batched, rtol=1e-6, atol=1e-5)

    # Compute end to end hessian and compare
    damping = 0.1

    H_single = ekfac_computer.estimate_hessian(damping)
    H_batched = ekfac_computer_batched.estimate_hessian(damping)

    assert jnp.allclose(H_single, H_batched, rtol=1e-6, atol=1e-5)


def test_ekfac_hvp_ihvp_consistency(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """
    Test whether the HVP and IHVP implementations are consistent
    in the sense of comparing it with multiplication of the full hessian / inverse hessian with the test vector.
    """
    comp = EKFACComputer(compute_context=collector_data_single).build()
    assert isinstance(comp, EKFACComputer)
    damping = 0.1

    model, params, loss = model_params_loss
    assert model_context.targets is not None, "ModelContext targets must not be None"
    v = sample_gradients(
        model=model,
        params=params,
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=loss,
        rng_key=PRNGKey(0),
        n_vectors=1,
    )[0]

    H = comp.estimate_hessian(damping)
    Hinv = comp.estimate_inverse_hessian(damping)
    ihvp = comp.estimate_ihvp(v, damping)
    hvp = comp.estimate_hvp(v, damping)

    ihvp_round_trip = H @ ihvp
    assert jnp.allclose(ihvp_round_trip, v, atol=1e-3, rtol=1e-1)
    hvp_round_trip = Hinv @ hvp
    assert jnp.allclose(hvp_round_trip, v, atol=1e-3, rtol=1e-1)


def test_kfac_hvp_ihvp_consistency(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether the HVP and IHVP implementations are consistent
    in the sense of comparing it with multiplication of the full hessian / inverse hessian with the test vector."""
    comp = KFACComputer(compute_context=collector_data_single).build()
    assert isinstance(comp, KFACComputer)
    damping = 0.1

    assert model_context.targets is not None, "ModelContext targets must not be None"

    model, params, loss = model_params_loss
    v = sample_gradients(
        model=model,
        params=params,
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=loss,
        rng_key=PRNGKey(1),
        n_vectors=1,
    )[0]

    H = comp.estimate_hessian(damping)
    Hinv = comp.estimate_inverse_hessian(damping)
    ihvp = comp.estimate_ihvp(v, damping)
    hvp = comp.estimate_hvp(v, damping)

    ihvp_round_trip = H @ ihvp
    assert jnp.allclose(ihvp_round_trip, v, atol=1e-4, rtol=1e-1)
    hvp_round_trip = Hinv @ hvp
    assert jnp.allclose(hvp_round_trip, v, atol=1e-4, rtol=1e-1)


def test_ekfac_explicit_vs_implicit_equivalence(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether the EK-FAC Hessian explicit computation matches the implicit computation via HVPs / IHVPs on basis vectors."""
    comp = EKFACComputer(compute_context=collector_data_single).build()
    assert isinstance(comp, EKFACComputer)
    damping = 0.1

    _, params, _ = model_params_loss
    dim = flatten_util.ravel_pytree(params)[0].shape[0]

    H = comp.estimate_hessian(damping)
    Hinv = comp.estimate_inverse_hessian(damping)
    H_imp, Hinv_imp = compute_full_implicit_matrices(comp, dim, damping)

    assert jnp.allclose(H, H_imp, atol=1e-2)
    assert jnp.allclose(Hinv, Hinv_imp, atol=1e-2)


def test_kfac_explicit_vs_implicit_equivalence(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether the K-FAC Hessian explicit computation matches the implicit computation via HVPs / IHVPs on basis vectors."""
    comp = KFACComputer(compute_context=collector_data_single).build()
    assert isinstance(comp, KFACComputer)
    damping = 0.1

    _, params, _ = model_params_loss
    dim = flatten_util.ravel_pytree(params)[0].shape[0]

    H = comp.estimate_hessian(damping)
    Hinv = comp.estimate_inverse_hessian(damping)

    H_imp, Hinv_imp = compute_full_implicit_matrices(comp, dim, damping)

    assert jnp.allclose(H, H_imp, atol=1e-2)
    assert jnp.allclose(Hinv, Hinv_imp, atol=1e-2)


def test_ekfac_ihvp_batched_shape_and_finiteness(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether EK-FAC IHVP with batched vectors works and produces finite outputs of correct shape."""
    comp = EKFACComputer(compute_context=collector_data_single).build()
    assert isinstance(comp, EKFACComputer)
    damping = 0.1

    model, params, loss = model_params_loss
    assert model_context.targets is not None, "ModelContext targets must not be None"
    V = sample_gradients(
        model=model,
        params=params,
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=loss,
        rng_key=PRNGKey(2),
        n_vectors=5,
    )

    IHVP = comp.estimate_ihvp(V, damping)

    assert IHVP.shape == V.shape
    assert jnp.isfinite(IHVP).all()


def test_ekfac_ihvp_batched_vs_single_consistency(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether EK-FAC IHVP with batched vectors is consistent with single vector IHVP computation."""
    comp = EKFACComputer(compute_context=collector_data_single).build()

    model, params, loss = model_params_loss
    assert model_context.targets is not None, "ModelContext targets must not be None"
    V = sample_gradients(
        model=model,
        params=params,
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=loss,
        rng_key=PRNGKey(3),
        n_vectors=4,
    )

    IHVP_batch = comp.estimate_ihvp(V, pseudo_inverse_factor=1e-4)

    for i in range(V.shape[0]):
        IHVP_single = comp.estimate_ihvp(V[i], pseudo_inverse_factor=1e-4)

        assert VectorMetric.RELATIVE_ERROR.compute(IHVP_batch[i], IHVP_single) < 1e-3


def test_ekfac_ihvp_hessian_roundtrip_batched(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test whether EK-FAC IHVP with batched vectors is consistent in the sense of comparing it with multiplication of the full hessian / inverse hessian with the test vector."""
    comp = EKFACComputer(compute_context=collector_data_single).build()
    assert isinstance(comp, EKFACComputer)

    model, params, loss = model_params_loss
    damping = 0.1
    assert model_context.targets is not None, "ModelContext targets must not be None"
    V = sample_gradients(
        model=model,
        params=params,
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=loss,
        rng_key=PRNGKey(4),
        n_vectors=5,
    )

    H = comp.estimate_hessian(damping)
    IHVP = comp.estimate_ihvp(V, damping)

    roundtrip = (H @ IHVP.T).T
    assert jnp.allclose(roundtrip, V, atol=1e-3)
