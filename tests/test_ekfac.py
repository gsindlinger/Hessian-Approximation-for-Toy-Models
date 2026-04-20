from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from jax.random import PRNGKey

from src.config import (
    ModelConfig,
    PseudoTargetGenerationStrategy,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import (
    sample_gradients,
)
from src.utils.data.data import Dataset
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
from tests.conftest import TrainingScenario
from tests._helpers import (
    cached_train_model_for_dataset,
    create_model_context,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("ekfac_linear_scenario", id="linear"),
        pytest.param("ekfac_multi_layer_scenario", id="multi_layer"),
    ],
    scope="session",
)
def training_scenario(request) -> TrainingScenario:
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def config(training_scenario: TrainingScenario) -> ModelConfig:
    return training_scenario.model_config


@pytest.fixture(scope="session")
def dataset(training_scenario: TrainingScenario) -> Dataset:
    """Create a random classification dataset for testing."""
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
    """Create a ModelContext for Hessian computation."""
    return create_model_context(dataset, model_params_loss)


def _collector_data(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    config: ModelConfig,
    dataset: Dataset,
    run_suffix: str,
    batch_size: Optional[int] = None,
    pseudo_target_strategy: PseudoTargetGenerationStrategy = PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
    pseudo_target_repetitions: int = 100,
) -> DataActivationsGradients:
    """Collect EK-FAC data with two runs."""
    model, params, loss = model_params_loss

    base_dir = config.directory
    assert base_dir is not None, "Model directory must be set"

    run_dir = f"{base_dir}/ekfac_{run_suffix}"

    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss,
        pseudo_target_strategy=pseudo_target_strategy,
        pseudo_target_repetitions=pseudo_target_repetitions,
    )

    collector_data = collector.collect(
        dataset=dataset,
        save_directory=run_dir,
        batch_size=batch_size,
        try_load=True,
        rng_key=PRNGKey(42),
    )

    return collector_data


@pytest.fixture(scope="session")
def collector_data_single(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> DataActivationsGradients:
    """Collect EK-FAC data (single batch collection)."""
    return _collector_data(
        config=config,
        model_params_loss=model_params_loss,
        dataset=dataset,
        run_suffix="_single",
        pseudo_target_strategy=PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
        pseudo_target_repetitions=1,
    )


@pytest.fixture(scope="session")
def collector_data_batched(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> DataActivationsGradients:
    """Collect EK-FAC data (batched collection)."""
    return _collector_data(
        config=config,
        model_params_loss=model_params_loss,
        dataset=dataset,
        run_suffix="_batched",
        batch_size=128,
        pseudo_target_strategy=PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
        pseudo_target_repetitions=1,
    )


@pytest.fixture(scope="session")
def collector_data_all_classes(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> DataActivationsGradients:
    """Collect EK-FAC data (single batch collection with pseudo targets)."""
    return _collector_data(
        config=config,
        model_params_loss=model_params_loss,
        dataset=dataset,
        run_suffix="_single_all_classes",
        pseudo_target_strategy=PseudoTargetGenerationStrategy.ALL_CLASSES,
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
    collector_data_single: DataActivationsGradients,
):
    """Test if EKFAC Hessian approximation can be computed without errors."""
    ekfac_computer = EKFACComputer(compute_context=collector_data_single).build()

    H = ekfac_computer.estimate_hessian(damping=1e-3)
    assert jnp.isfinite(H).all()


def test_gradient_consistency(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
    collector_data_single: DataActivationsGradients,
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

    def loss_fn_apply(p):
        return loss(model.apply(p, dataset.inputs), dataset.targets, reduction="sum")

    gt_grads = jax.grad(loss_fn_apply)(params)

    activations = collector_data_single.activations
    gradients = collector_data_single.gradients
    layer_names = collector_data_single.layer_names

    for i, (layer, gt) in enumerate(gt_grads["params"].items()):
        assert layer == layer_names[i], "Layer names do not match"
        W_grad = gt["kernel"]
        # gradients have shape (N, O, k); take k=0 slice -> (N, O)
        a, g = activations[layer], gradients[layer][..., 0]

        ag = jnp.einsum("ni,no->io", a, g)
        assert jnp.allclose(W_grad, ag, atol=1e-4)

        kron = sum(jnp.kron(a[i], g[i]) for i in range(a.shape[0])) / a.shape[0]
        assert jnp.allclose(W_grad.reshape(-1) / a.shape[0], kron, atol=1e-4)


def test_kfac_hessian(
    model_context: ModelContext,
    collector_data_single: DataActivationsGradients,
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
    collector_data_single: DataActivationsGradients,
    model_context: ModelContext,
):
    """This test verifies that the K-FAC Hessian computed via the Kronecker product of the covariances directly equals
    the K-FAC Hessian computed via the eigenvector method implemented in KFACComputer."""

    assert all(
        gradients.shape[-1] == 1
        for gradients in collector_data_single.gradients.values()
    ), "This test assumes that gradients have shape (N, O, 1) for each layer. "

    activations, gradients, layer_names = (
        collector_data_single.activations,
        collector_data_single.gradients,
        collector_data_single.layer_names,
    )

    # Compute covariances manually
    covariances_activations = {}
    covariances_gradients = {}

    for layer in layer_names:
        a = activations[layer]
        g = gradients[layer][..., 0]  # (N, O, 1) -> (N, O)

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
        atol=1e-3,
    )


def test_ekfac_hessian(
    model_context: ModelContext,
    collector_data_single: DataActivationsGradients,
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
    collector_data_single: DataActivationsGradients,
    collector_data_batched: DataActivationsGradients,
):
    """Test whether EK-FAC data collected with different batch sizes is consistent."""

    ekfac_computer = EKFACComputer(compute_context=collector_data_single).build()

    ekfac_computer_batched = EKFACComputer(
        compute_context=collector_data_batched
    ).build()

    assert isinstance(ekfac_computer, EKFACComputer)
    assert isinstance(ekfac_computer_batched, EKFACComputer)

    for layer in ekfac_computer.compute_context.layer_names:
        single_block = ekfac_computer.layer_matrix.blocks[(layer, layer)]
        batched_block = ekfac_computer_batched.layer_matrix.blocks[(layer, layer)]

        assert jnp.allclose(single_block.Q_A, batched_block.Q_A, rtol=1e-6, atol=1e-5)
        assert jnp.allclose(single_block.Q_G, batched_block.Q_G, rtol=1e-6, atol=1e-5)
        assert jnp.allclose(
            single_block.Lambda, batched_block.Lambda, rtol=1e-6, atol=1e-5
        )

    # Compute end to end hessian and compare
    damping = 0.1

    H_single = ekfac_computer.estimate_hessian(damping)
    H_batched = ekfac_computer_batched.estimate_hessian(damping)

    assert jnp.allclose(H_single, H_batched, rtol=1e-6, atol=1e-5)


def test_ekfac_hvp_ihvp_consistency(
    model_context: ModelContext,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: DataActivationsGradients,
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
    collector_data_single: DataActivationsGradients,
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
    assert jnp.allclose(ihvp_round_trip, v, atol=1e-3, rtol=1e-1)
    hvp_round_trip = Hinv @ hvp
    assert jnp.allclose(hvp_round_trip, v, atol=1e-3, rtol=1e-1)


def test_ekfac_explicit_vs_implicit_equivalence(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    collector_data_single: DataActivationsGradients,
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
    collector_data_single: DataActivationsGradients,
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
    collector_data_single: DataActivationsGradients,
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
    collector_data_single: DataActivationsGradients,
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
    collector_data_single: DataActivationsGradients,
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
    assert jnp.allclose(roundtrip, V, atol=5e-3)


def test_ekfac_all_classes_better_single_run(
    model_context: ModelContext,
    collector_data_single: DataActivationsGradients,
    collector_data_all_classes: DataActivationsGradients,
):
    """Test whether EK-FAC collected with all classes performs better than single run EK-FAC in terms of Frobenius norm to true Hessian."""
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

    error_single = FullMatrixMetric.FROBENIUS.compute(H, E)

    E_all_classes = (
        EKFACComputer(compute_context=collector_data_all_classes)
        .build()
        .estimate_hessian(damping=damping)
    )

    error_all_classes_H = FullMatrixMetric.FROBENIUS.compute(H, E_all_classes)
    error_all_classes_GNH = FullMatrixMetric.FROBENIUS.compute(G, E_all_classes)

    assert error_all_classes_H <= error_single, (
        "EK-FAC collected with all classes should perform better than single run EK-FAC."
    )
    assert error_all_classes_GNH <= error_single, (
        "EK-FAC collected with all classes should perform better than single run EK-FAC."
    )


def test_ekfac_better_approximation_than_kfac(
    model_context: ModelContext,
    collector_data_all_classes: DataActivationsGradients,
):
    """
    Test whether EK-FAC provides a better approximation than K-FAC.

    According to George et al. (2018), EKFAC should satisfy:
    ||G - G_EKFAC||_F <= ||G - G_KFAC||_F

    where G is the true GNH (Gauss-Newton Hessian).
    """
    damping = 0.0

    # Compute true GNH as reference
    G = (
        GNHComputer(compute_context=model_context)
        .build()
        .estimate_hessian(damping=damping)
    )

    # Compute KFAC approximation
    K = (
        KFACComputer(compute_context=collector_data_all_classes)
        .build()
        .estimate_hessian(damping=damping)
    )

    # Compute EKFAC approximation
    E = (
        EKFACComputer(compute_context=collector_data_all_classes)
        .build()
        .estimate_hessian(damping=damping)
    )

    # Compute Frobenius norm errors
    error_kfac = FullMatrixMetric.FROBENIUS.compute(G, K)
    error_ekfac = FullMatrixMetric.FROBENIUS.compute(G, E)

    print("\nApproximation errors to GNH:")
    print(f"  KFAC error:  {error_kfac:.6f}")
    print(f"  EKFAC error: {error_ekfac:.6f}")
    print(f"  Improvement: {(error_kfac - error_ekfac) / error_kfac * 100:.2f}%")

    # EKFAC should be at least as good as KFAC
    assert error_ekfac <= error_kfac * 1.00, (
        f"EKFAC should provide better or equal approximation to GNH than KFAC. "
        f"KFAC error: {error_kfac:.6f}, EKFAC error: {error_ekfac:.6f}"
    )


def test_ekfac_better_approximation_than_kfac_relative(
    model_context: ModelContext,
    collector_data_all_classes: DataActivationsGradients,
):
    """
    Test EKFAC vs KFAC using relative Frobenius error.

    This variant uses relative error to account for different scales:
    relative_error = ||G - G_approx||_F / ||G||_F
    """
    damping = 0.0001

    # Compute true GNH as reference
    G = (
        GNHComputer(compute_context=model_context)
        .build()
        .estimate_hessian(damping=damping)
    )

    # Compute KFAC approximation
    K = (
        KFACComputer(compute_context=collector_data_all_classes)
        .build()
        .estimate_hessian(damping=damping)
    )

    # Compute EKFAC approximation
    E = (
        EKFACComputer(compute_context=collector_data_all_classes)
        .build()
        .estimate_hessian(damping=damping)
    )

    # Compute relative Frobenius norm errors
    error_kfac_rel = FullMatrixMetric.RELATIVE_FROBENIUS.compute(G, K)
    error_ekfac_rel = FullMatrixMetric.RELATIVE_FROBENIUS.compute(G, E)

    print("\nRelative approximation errors to GNH:")
    print(f"  KFAC relative error:  {error_kfac_rel:.6f}")
    print(f"  EKFAC relative error: {error_ekfac_rel:.6f}")
    print(
        f"  Relative improvement: {(error_kfac_rel - error_ekfac_rel) / error_kfac_rel * 100:.2f}%"
    )

    # EKFAC should be at least as good as KFAC
    assert error_ekfac_rel <= error_kfac_rel * 1.00, (
        f"EKFAC should provide better or equal approximation to GNH than KFAC. "
        f"KFAC relative error: {error_kfac_rel:.6f}, EKFAC relative error: {error_ekfac_rel:.6f}"
    )
