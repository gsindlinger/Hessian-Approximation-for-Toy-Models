from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from jax.random import PRNGKey

from src.config import (
    ActivationFunction,
    DatasetEnum,
    LossType,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    PseudoTargetGenerationStrategy,
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
    sample_gradients,
)
from src.utils.data.data import (
    Dataset,
    DownloadableDataset,
    RandomClassificationDataset,
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


@pytest.fixture(params=["linear", "multi_layer"], scope="session")
def config(request, tmp_path_factory):
    """Create model configuration for testing."""
    base = tmp_path_factory.mktemp(request.param)

    if request.param == "linear":
        architecture = ModelArchitecture.LINEAR
        hidden_dim = []
        activation = None
    else:
        architecture = ModelArchitecture.MLP
        activation = ActivationFunction.TANH
        hidden_dim = [16] * 2

    return ModelConfig(
        architecture=architecture,
        input_dim=10,  # Will be updated from dataset
        hidden_dim=hidden_dim if hidden_dim else None,
        activation=activation,
        output_dim=2,  # Will be updated from dataset
        loss=LossType.CROSS_ENTROPY,
        training=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=0,
            optimizer=OptimizerType.ADAMW,
            epochs=100,
            batch_size=32,
        ),
        directory=str(base / "model"),
    )


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """Create a random classification dataset for testing."""
    sklearn_dataset = DownloadableDataset.load(
        dataset=DatasetEnum.SKLEARN_DIGITS,
        directory="./experiments/datasets/sklearn_digits",
        store_on_disk=True,
    )
    sklearn_dataset, _ = sklearn_dataset.train_test_split(test_size=0.1, seed=42)
    return sklearn_dataset
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
            batch_size=config.training.batch_size, seed=42
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
def collector_data_single_with_pseudo_targets(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> DataActivationsGradients:
    """Collect EK-FAC data (single batch collection with pseudo targets)."""
    return _collector_data(
        config=config,
        model_params_loss=model_params_loss,
        dataset=dataset,
        run_suffix="_single_with_pseudo_targets",
        pseudo_target_strategy=PseudoTargetGenerationStrategy.MCMC,
        pseudo_target_repetitions=10,
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
        pseudo_target_strategy=PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
        pseudo_target_repetitions=1,
    )

    def loss_fn_apply(p):
        return loss(model.apply(p, dataset.inputs), dataset.targets, reduction="sum")

    gt_grads = jax.grad(loss_fn_apply)(params)

    activations = collector_data_single.activations
    gradients = collector_data_single.gradients
    layer_names = collector_data_single.layer_names

    for i, (layer, gt) in enumerate(gt_grads["params"].items()):
        assert layer == layer_names[i], "Layer names do not match"
        W_grad = gt["kernel"]
        a, g = activations[layer], gradients[layer][0]

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
        gradients.shape[0] == 1
        for gradients in collector_data_single.gradients.values()
    ), "This test assumes that gradients have shape (1, N, O) for each layer. "

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
        g = gradients[layer][0]

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

    true_hessian_computer = HessianComputer(compute_context=model_context)
    true_hessian_H = true_hessian_computer.compute_hessian(damping=0.0)

    # plot all three matrices for visual inspection and save the plot (need the scale of the values)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(kron_comparison_method_H, cmap="viridis", aspect="auto")
    axs[0].set_title("Kron Comparison Method Hessian")
    axs[1].imshow(eigenvector_method_H, cmap="viridis", aspect="auto")
    axs[1].set_title("Eigenvector Method Hessian")
    axs[2].imshow(true_hessian_H, cmap="viridis", aspect="auto")
    axs[2].set_title("True Hessian")
    plt.colorbar(
        axs[0].imshow(kron_comparison_method_H, cmap="viridis", aspect="auto"),
        ax=axs[0],
    )
    plt.colorbar(
        axs[1].imshow(eigenvector_method_H, cmap="viridis", aspect="auto"), ax=axs[1]
    )
    plt.colorbar(
        axs[2].imshow(true_hessian_H, cmap="viridis", aspect="auto"), ax=axs[2]
    )
    plt.tight_layout()
    plt.savefig("kfac_hessian_comparison.png")

    assert jnp.allclose(
        kron_comparison_method_H,
        eigenvector_method_H,
        atol=1e-4,
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
    assert jnp.allclose(ihvp_round_trip, v, atol=1e-4, rtol=1e-1)
    hvp_round_trip = Hinv @ hvp
    assert jnp.allclose(hvp_round_trip, v, atol=1e-4, rtol=1e-1)


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
    assert jnp.allclose(roundtrip, V, atol=1e-3)


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
    assert error_ekfac <= error_kfac * 1.05, (
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
    assert error_ekfac_rel <= error_kfac_rel * 1.05, (
        f"EKFAC should provide better or equal approximation to GNH than KFAC. "
        f"KFAC relative error: {error_kfac_rel:.6f}, EKFAC relative error: {error_ekfac_rel:.6f}"
    )


# def test_ekfac_correction_diagnostics(
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """
#     Diagnostic test to examine the scale of eigenvalue corrections vs base eigenvalues.

#     This helps debug cases where EKFAC performs worse than expected by checking if:
#     1. Corrections are much larger than base eigenvalues (dominating)
#     2. Corrections are much smaller than base eigenvalues (negligible)
#     3. Corrections have appropriate scale
#     """
#     ekfac_computer = EKFACComputer(compute_context=collector_data_all_classes).build()
#     assert isinstance(ekfac_computer, EKFACComputer)

#     print("\n" + "="*70)
#     print("EKFAC Eigenvalue Correction Diagnostics")
#     print("="*70)

#     for layer in ekfac_computer.compute_context.layer_names:
#         lambda_A = ekfac_computer.precomputed_data.activation_eigenvalues[layer]
#         lambda_G = ekfac_computer.precomputed_data.gradient_eigenvalues[layer]
#         corrections = ekfac_computer.precomputed_data.eigenvalue_corrections[layer]

#         # Compute base KFAC eigenvalues (Kronecker product)
#         base_eigenvalues = lambda_A[:, None] * lambda_G[None, :]  # (I, O)

#         # Statistics
#         correction_mean = jnp.mean(corrections)
#         correction_std = jnp.std(corrections)
#         correction_min = jnp.min(corrections)
#         correction_max = jnp.max(corrections)

#         base_mean = jnp.mean(base_eigenvalues)
#         base_std = jnp.std(base_eigenvalues)
#         base_min = jnp.min(base_eigenvalues)
#         base_max = jnp.max(base_eigenvalues)

#         ratio_mean = correction_mean / (base_mean + 1e-10)

#         print(f"\nLayer: {layer}")
#         print(f"  Base eigenvalues (KFAC):")
#         print(f"    Mean: {base_mean:.6e}, Std: {base_std:.6e}")
#         print(f"    Range: [{base_min:.6e}, {base_max:.6e}]")
#         print(f"  Eigenvalue corrections (EKFAC):")
#         print(f"    Mean: {correction_mean:.6e}, Std: {correction_std:.6e}")
#         print(f"    Range: [{correction_min:.6e}, {correction_max:.6e}]")
#         print(f"  Correction/Base ratio (mean): {ratio_mean:.6f}")

#         if ratio_mean > 10:
#             print(f"  ⚠️  WARNING: Corrections are {ratio_mean:.1f}x larger than base eigenvalues!")
#             print(f"      This may indicate an issue with normalization or computation.")
#         elif ratio_mean < 0.01:
#             print(f"  ℹ️  INFO: Corrections are very small ({ratio_mean:.4f}x base eigenvalues)")
#             print(f"      EKFAC may not provide much benefit over KFAC.")
#         else:
#             print(f"  ✓  Corrections have reasonable scale relative to base eigenvalues.")

#     print("="*70)

#     assert False  # This test is diagnostic; no pass/fail criteria

# def test_ekfac_correction_manual_verification(
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """Manually verify EKFAC correction computation."""

#     # Build EKFAC
#     ekfac_computer = EKFACComputer(compute_context=collector_data_all_classes).build()

#     # Get first layer data
#     layer = collector_data_all_classes.layer_names[0]
#     activations = collector_data_all_classes.activations[layer]  # (N, I)
#     gradients = collector_data_all_classes.gradients[layer]      # (K, N, O)
#     probabilities = collector_data_all_classes.probabilities     # (N, K)

#     Q_A = ekfac_computer.precomputed_data.activation_eigenvectors[layer]  # (I, I)
#     Q_G = ekfac_computer.precomputed_data.gradient_eigenvectors[layer]    # (O, O)
#     lambda_A = ekfac_computer.precomputed_data.activation_eigenvalues[layer]  # (I,)
#     lambda_G = ekfac_computer.precomputed_data.gradient_eigenvalues[layer]    # (O,)

#     N, I = activations.shape
#     K, _, O = gradients.shape

#     print(f"\nLayer: {layer}")
#     print(f"N={N}, I={I}, K={K}, O={O}")

#     # Transform to eigenbasis
#     a_tilde = activations @ Q_A  # (N, I) - note: Q_A columns are eigenvectors
#     # Or should it be Q_A.T @ activations.T?

#     # Let's check both conventions
#     a_tilde_v1 = jnp.einsum("ni,ij->nj", activations, Q_A)   # (N, I)
#     a_tilde_v2 = jnp.einsum("ij,nj->ni", Q_A.T, activations) # (N, I)

#     print(f"a_tilde_v1 == a_tilde_v2: {jnp.allclose(a_tilde_v1, a_tilde_v2)}")

#     # Check that transformed activations have diagonal covariance = Lambda_A
#     a_tilde_cov_v1 = (a_tilde_v1.T @ a_tilde_v1) / N
#     a_tilde_cov_v2 = (a_tilde_v2.T @ a_tilde_v2) / N

#     print(f"v1 diag matches lambda_A: {jnp.allclose(jnp.diag(a_tilde_cov_v1), lambda_A, rtol=1e-3)}")
#     print(f"v2 diag matches lambda_A: {jnp.allclose(jnp.diag(a_tilde_cov_v2), lambda_A, rtol=1e-3)}")
#     print(f"v1 is diagonal: {jnp.allclose(a_tilde_cov_v1, jnp.diag(jnp.diag(a_tilde_cov_v1)), atol=1e-5)}")
#     print(f"v2 is diagonal: {jnp.allclose(a_tilde_cov_v2, jnp.diag(jnp.diag(a_tilde_cov_v2)), atol=1e-5)}")

#     # get relative error of a_tilde_cov_v1, jnp.diag(jnp.diag(a_tilde_cov_v1))
#     print(f"v1 relative error to diagonal: {jnp.linalg.norm(a_tilde_cov_v1 - jnp.diag(jnp.diag(a_tilde_cov_v1))) / jnp.linalg.norm(a_tilde_cov_v1)}")
#     print(f"v2 relative error to diagonal: {jnp.linalg.norm(a_tilde_cov_v2 - jnp.diag(jnp.diag(a_tilde_cov_v2))) / jnp.linalg.norm(a_tilde_cov_v2)}")

#     # Similarly for gradients
#     g_tilde = jnp.einsum("op,knp->kno", Q_G.T, gradients)  # Your convention: (K, N, O)

#     # Verify gradient covariance in eigenbasis
#     # G_cov should be diagonal with eigenvalues lambda_G
#     weighted_g = gradients * jnp.sqrt(probabilities.T[:, :, None])  # (K, N, O)
#     weighted_g_flat = weighted_g.reshape(-1, O)
#     G_cov_original = (weighted_g_flat.T @ weighted_g_flat) / N

#     weighted_g_tilde = g_tilde * jnp.sqrt(probabilities.T[:, :, None])
#     weighted_g_tilde_flat = weighted_g_tilde.reshape(-1, O)
#     G_cov_transformed = (weighted_g_tilde_flat.T @ weighted_g_tilde_flat) / N

#     print(f"G_cov_transformed diag matches lambda_G: {jnp.allclose(jnp.diag(G_cov_transformed), lambda_G, rtol=1e-3)}")
#     print(f"G_cov_transformed is diagonal: {jnp.allclose(G_cov_transformed, jnp.diag(jnp.diag(G_cov_transformed)), atol=1e-5)}")

#     # relative error to diagonal
#     print(f"G_cov_transformed relative error to diagonal: {jnp.linalg.norm(G_cov_transformed - jnp.diag(jnp.diag(G_cov_transformed))) / jnp.linalg.norm(G_cov_transformed)}")


#     # Now compute EKFAC correction manually
#     # Using the correct a_tilde (whichever gives diagonal covariance)
#     a_tilde = a_tilde_v2  # Assuming this is correct based on your code

#     # EKFAC: E[sum_c p(c|x) (a_tilde_i * g_tilde_cj)^2]
#     correction_manual = jnp.zeros((I, O))
#     for n in range(N):
#         for c in range(K):
#             p_c = probabilities[n, c]
#             outer_sq = jnp.outer(a_tilde[n], g_tilde[c, n]) ** 2  # (I, O)
#             correction_manual += p_c * outer_sq
#     correction_manual /= N

#     # Compare with computed correction
#     correction_computed = ekfac_computer.precomputed_data.eigenvalue_corrections[layer]

#     print(f"\nManual vs Computed correction:")
#     print(f"  Close: {jnp.allclose(correction_manual, correction_computed, rtol=1e-3)}")
#     print(f"  Max diff: {jnp.max(jnp.abs(correction_manual - correction_computed))}")

#     # Compare with KFAC
#     kfac_eigenvalues = jnp.outer(lambda_A, lambda_G)

#     print(f"\nEKFAC correction vs KFAC eigenvalues:")
#     print(f"  Correction mean: {jnp.mean(correction_manual)}")
#     print(f"  KFAC mean: {jnp.mean(kfac_eigenvalues)}")
#     print(f"  Ratio: {jnp.mean(correction_manual) / jnp.mean(kfac_eigenvalues)}")

#     # Check structure difference
#     ratio = correction_manual / (kfac_eigenvalues + 1e-12)
#     print(f"  Ratio std: {jnp.std(ratio)}")  # Should be > 0 if EKFAC differs from KFAC

#     # Key diagnostic: check if a_tilde and g_tilde are actually correlated
#     # If they were independent, EKFAC would equal KFAC
#     for i in range(min(3, I)):
#         for j in range(min(3, O)):
#             # Compute E[a_i^2], E[sum_c p_c g_cj^2], E[sum_c p_c (a_i g_cj)^2]
#             E_a2 = jnp.mean(a_tilde[:, i] ** 2)
#             E_g2 = jnp.sum(probabilities * (g_tilde[:, :, j].T ** 2), axis=1).mean()
#             E_ag2 = jnp.sum(probabilities * ((a_tilde[:, i:i+1] * g_tilde[:, :, j].T) ** 2), axis=1).mean()

#             # If independent: E[ag^2] = E[a^2] * E[g^2]
#             independent_assumption = E_a2 * E_g2

#             print(f"  ({i},{j}): E[a²]={E_a2:.6f}, E[g²]={E_g2:.6f}, "
#                   f"E[(ag)²]={E_ag2:.6f}, E[a²]*E[g²]={independent_assumption:.6f}, "
#                   f"ratio={E_ag2/independent_assumption:.4f}")

#     assert False  # Force output

# def test_hessian_submatrix_comparison_nonzero(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """Compare actual Hessian submatrices in non-zero regions."""
#     damping = 0.0001

#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=damping)
#     K = KFACComputer(compute_context=collector_data_all_classes).build().estimate_hessian(damping=damping)
#     E = EKFACComputer(compute_context=collector_data_all_classes).build().estimate_hessian(damping=damping)

#     # Find indices where GNH has significant values
#     threshold = 0.001 * jnp.max(jnp.abs(G))
#     significant_indices = jnp.where(jnp.abs(G) > threshold)

#     print(f"GNH shape: {G.shape}")
#     print(f"GNH max: {jnp.max(G)}, min: {jnp.min(G)}")
#     print(f"Number of significant elements: {len(significant_indices[0])}")

#     # Find a non-zero block
#     # Try middle of matrix
#     mid = G.shape[0] // 2
#     print(f"\nGNH[{mid}:{mid+10},{mid}:{mid+10}]:")
#     print(G[mid:mid+10, mid:mid+10])
#     print(f"\nKFAC[{mid}:{mid+10},{mid}:{mid+10}]:")
#     print(K[mid:mid+10, mid:mid+10])
#     print(f"\nEKFAC[{mid}:{mid+10},{mid}:{mid+10}]:")
#     print(E[mid:mid+10, mid:mid+10])

#     # Compute errors only on non-zero region
#     mask = jnp.abs(G) > threshold

#     G_masked = G[mask]
#     K_masked = K[mask]
#     E_masked = E[mask]

#     error_kfac = jnp.sqrt(jnp.mean((G_masked - K_masked)**2))
#     error_ekfac = jnp.sqrt(jnp.mean((G_masked - E_masked)**2))

#     print(f"\nRMSE on significant elements:")
#     print(f"  KFAC:  {error_kfac}")
#     print(f"  EKFAC: {error_ekfac}")

#     # Element-wise comparison
#     diff_kfac = jnp.abs(G - K)
#     diff_ekfac = jnp.abs(G - E)
#     diff_kfac_ekfac = jnp.abs(K - E)

#     # Where does EKFAC improve over KFAC?
#     improvement = diff_kfac - diff_ekfac  # positive = EKFAC better
#     print(f"\nEKFAC vs KFAC improvement (positive = EKFAC better):")
#     print(f"  Mean improvement: {jnp.mean(improvement)}")
#     print(f"  Max improvement: {jnp.max(improvement)}")
#     print(f"  Min improvement (max degradation): {jnp.min(improvement)}")
#     print(f"  Elements where EKFAC better: {jnp.sum(improvement > 1e-8)}")
#     print(f"  Elements where KFAC better: {jnp.sum(improvement < -1e-8)}")

#     # Check the structure of KFAC vs EKFAC difference
#     print(f"\nKFAC-EKFAC difference stats:")
#     print(f"  Mean: {jnp.mean(diff_kfac_ekfac)}")
#     print(f"  Std: {jnp.std(diff_kfac_ekfac)}")
#     print(f"  Max: {jnp.max(diff_kfac_ekfac)}")

#     assert False

# def test_block_diagonal_error_analysis(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """Analyze how much error comes from block-diagonal assumption vs within-block."""
#     damping = 0.0001

#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=damping)
#     K = KFACComputer(compute_context=collector_data_all_classes).build().estimate_hessian(damping=damping)
#     E = EKFACComputer(compute_context=collector_data_all_classes).build().estimate_hessian(damping=damping)

#     # Get layer dimensions
#     layer_names = collector_data_all_classes.layer_names
#     ekfac = EKFACComputer(compute_context=collector_data_all_classes).build()

#     # Calculate block boundaries
#     boundaries = [0]
#     for layer in layer_names:
#         I = ekfac.precomputed_data.activation_eigenvectors[layer].shape[0]
#         O = ekfac.precomputed_data.gradient_eigenvectors[layer].shape[0]
#         boundaries.append(boundaries[-1] + I * O)

#     print(f"Block boundaries: {boundaries}")

#     # Extract block-diagonal part of GNH
#     G_block_diag = jnp.zeros_like(G)
#     for i in range(len(boundaries) - 1):
#         start, end = boundaries[i], boundaries[i+1]
#         G_block_diag = G_block_diag.at[start:end, start:end].set(G[start:end, start:end])

#     # Compute errors
#     total_error_kfac = jnp.linalg.norm(G - K, 'fro')
#     total_error_ekfac = jnp.linalg.norm(G - E, 'fro')

#     # Error from block-diagonal assumption (off-block elements)
#     off_block_error = jnp.linalg.norm(G - G_block_diag, 'fro')

#     # Error within blocks
#     within_block_error_kfac = jnp.linalg.norm(G_block_diag - K, 'fro')
#     within_block_error_ekfac = jnp.linalg.norm(G_block_diag - E, 'fro')

#     print(f"\nError decomposition:")
#     print(f"  Total GNH norm: {jnp.linalg.norm(G, 'fro'):.6f}")
#     print(f"  Off-block-diagonal error (structural): {off_block_error:.6f}")
#     print(f"  Within-block error (KFAC): {within_block_error_kfac:.6f}")
#     print(f"  Within-block error (EKFAC): {within_block_error_ekfac:.6f}")
#     print(f"  Total error (KFAC): {total_error_kfac:.6f}")
#     print(f"  Total error (EKFAC): {total_error_ekfac:.6f}")

#     print(f"\nError breakdown (percentage of total KFAC error):")
#     print(f"  Off-block contribution: {100 * off_block_error / total_error_kfac:.1f}%")
#     print(f"  Within-block contribution: {100 * within_block_error_kfac / total_error_kfac:.1f}%")

#     print(f"\nWithin-block improvement (EKFAC vs KFAC):")
#     improvement = (within_block_error_kfac - within_block_error_ekfac) / within_block_error_kfac * 100
#     print(f"  {improvement:.2f}%")

#     # Per-layer analysis
#     print(f"\nPer-layer within-block errors:")
#     for i, layer in enumerate(layer_names):
#         start, end = boundaries[i], boundaries[i+1]
#         G_block = G[start:end, start:end]
#         K_block = K[start:end, start:end]
#         E_block = E[start:end, start:end]

#         err_kfac = jnp.linalg.norm(G_block - K_block, 'fro')
#         err_ekfac = jnp.linalg.norm(G_block - E_block, 'fro')

#         print(f"  {layer}: KFAC={err_kfac:.6f}, EKFAC={err_ekfac:.6f}, improvement={(err_kfac-err_ekfac)/err_kfac*100:.2f}%")

#     assert False


# def test_eigenbasis_diagonal_comparison(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """Compare KFAC and EKFAC to GNH in the Kronecker eigenbasis."""
#     damping = 0.0

#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=damping)

#     ekfac = EKFACComputer(compute_context=collector_data_all_classes).build()

#     layer_names = collector_data_all_classes.layer_names

#     # Calculate block boundaries
#     boundaries = [0]
#     for layer in layer_names:
#         I = ekfac.precomputed_data.activation_eigenvectors[layer].shape[0]
#         O = ekfac.precomputed_data.gradient_eigenvectors[layer].shape[0]
#         boundaries.append(boundaries[-1] + I * O)

#     print("Eigenbasis diagonal comparison per layer:\n")

#     for i, layer in enumerate(layer_names):
#         start, end = boundaries[i], boundaries[i+1]

#         Q_A = ekfac.precomputed_data.activation_eigenvectors[layer]
#         Q_G = ekfac.precomputed_data.gradient_eigenvectors[layer]

#         # Extract block from GNH
#         G_block = G[start:end, start:end]

#         # Transform GNH block to eigenbasis: (Q_A ⊗ Q_G)^T G (Q_A ⊗ Q_G)
#         Q_kron = jnp.kron(Q_A, Q_G)
#         G_eigenbasis = Q_kron.T @ G_block @ Q_kron

#         # Get diagonal of GNH in eigenbasis
#         G_diag_eigenbasis = jnp.diag(G_eigenbasis)

#         # KFAC diagonal in eigenbasis
#         lambda_A = ekfac.precomputed_data.activation_eigenvalues[layer]
#         lambda_G = ekfac.precomputed_data.gradient_eigenvalues[layer]
#         kfac_diag = jnp.outer(lambda_A, lambda_G).flatten()

#         # EKFAC diagonal in eigenbasis (the corrections)
#         ekfac_diag = ekfac.precomputed_data.eigenvalue_corrections[layer].flatten()

#         # Compare diagonals
#         err_kfac_diag = jnp.sqrt(jnp.mean((G_diag_eigenbasis - kfac_diag)**2))
#         err_ekfac_diag = jnp.sqrt(jnp.mean((G_diag_eigenbasis - ekfac_diag)**2))

#         print(f"Layer: {layer}")
#         print(f"  Diagonal RMSE - KFAC: {err_kfac_diag:.6e}, EKFAC: {err_ekfac_diag:.6e}")
#         if err_kfac_diag > 1e-10:
#             print(f"  Diagonal improvement: {(err_kfac_diag - err_ekfac_diag) / err_kfac_diag * 100:.2f}%")

#         # Check off-diagonal magnitude in eigenbasis
#         off_diag_norm = jnp.linalg.norm(G_eigenbasis - jnp.diag(G_diag_eigenbasis), 'fro')
#         diag_norm = jnp.linalg.norm(G_diag_eigenbasis)
#         print(f"  Off-diagonal / Diagonal norm ratio: {off_diag_norm / (diag_norm + 1e-10):.4f}")

#         # Sample comparison
#         # Find indices with significant values
#         significant = jnp.argsort(jnp.abs(G_diag_eigenbasis))[-5:]
#         print(f"  Top 5 diagonal elements:")
#         for idx in significant:
#             print(f"    [{idx}]: GNH={G_diag_eigenbasis[idx]:.6e}, KFAC={kfac_diag[idx]:.6e}, EKFAC={ekfac_diag[idx]:.6e}")
#         print()

#     assert False


# def test_ekfac_diagonal_vs_gnh_diagonal_direct(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """
#     Directly compute the diagonal of GNH in the eigenbasis and compare to EKFAC.

#     The GNH diagonal in eigenbasis should equal:
#     diag((Q_A ⊗ Q_G)^T F (Q_A ⊗ Q_G))

#     where F is the true Fisher/GNH.

#     EKFAC computes:
#     E[sum_c p(c|x) (ã_i * g̃_{c,o})²]

#     These should be equal!
#     """

#     # Get the GNH
#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=0.0)

#     # Build EKFAC
#     ekfac = EKFACComputer(compute_context=collector_data_all_classes).build()

#     layer_names = collector_data_all_classes.layer_names

#     # Get boundaries
#     boundaries = [0]
#     for layer in layer_names:
#         I = ekfac.precomputed_data.activation_eigenvectors[layer].shape[0]
#         O = ekfac.precomputed_data.gradient_eigenvectors[layer].shape[0]
#         boundaries.append(boundaries[-1] + I * O)

#     print("\n" + "="*70)
#     print("Direct GNH diagonal vs EKFAC correction comparison")
#     print("="*70)

#     for i, layer in enumerate(layer_names):
#         start, end = boundaries[i], boundaries[i+1]

#         Q_A = ekfac.precomputed_data.activation_eigenvectors[layer]
#         Q_G = ekfac.precomputed_data.gradient_eigenvectors[layer]

#         # Get GNH block
#         G_block = G[start:end, start:end]

#         # Transform to eigenbasis
#         Q_kron = jnp.kron(Q_A, Q_G)
#         G_eigenbasis = Q_kron.T @ G_block @ Q_kron

#         # Get diagonal
#         gnh_diag = jnp.diag(G_eigenbasis)

#         # Get EKFAC correction (should equal GNH diagonal in eigenbasis)
#         ekfac_diag = ekfac.precomputed_data.eigenvalue_corrections[layer].flatten()

#         # Get KFAC eigenvalues for comparison
#         lambda_A = ekfac.precomputed_data.activation_eigenvalues[layer]
#         lambda_G = ekfac.precomputed_data.gradient_eigenvalues[layer]
#         kfac_diag = jnp.outer(lambda_A, lambda_G).flatten()

#         print(f"\nLayer: {layer} (size {end-start})")

#         # Stats
#         print(f"  GNH diag:   mean={jnp.mean(gnh_diag):.6e}, std={jnp.std(gnh_diag):.6e}, "
#               f"min={jnp.min(gnh_diag):.6e}, max={jnp.max(gnh_diag):.6e}")
#         print(f"  EKFAC diag: mean={jnp.mean(ekfac_diag):.6e}, std={jnp.std(ekfac_diag):.6e}, "
#               f"min={jnp.min(ekfac_diag):.6e}, max={jnp.max(ekfac_diag):.6e}")
#         print(f"  KFAC diag:  mean={jnp.mean(kfac_diag):.6e}, std={jnp.std(kfac_diag):.6e}, "
#               f"min={jnp.min(kfac_diag):.6e}, max={jnp.max(kfac_diag):.6e}")

#         # RMSE
#         rmse_ekfac = jnp.sqrt(jnp.mean((gnh_diag - ekfac_diag)**2))
#         rmse_kfac = jnp.sqrt(jnp.mean((gnh_diag - kfac_diag)**2))

#         print(f"  RMSE to GNH diag - EKFAC: {rmse_ekfac:.6e}, KFAC: {rmse_kfac:.6e}")

#         if rmse_kfac > 1e-10:
#             improvement = (rmse_kfac - rmse_ekfac) / rmse_kfac * 100
#             print(f"  Improvement: {improvement:.2f}%")

#         # Correlation
#         corr_ekfac = jnp.corrcoef(gnh_diag, ekfac_diag)[0, 1]
#         corr_kfac = jnp.corrcoef(gnh_diag, kfac_diag)[0, 1]
#         print(f"  Correlation with GNH diag - EKFAC: {corr_ekfac:.6f}, KFAC: {corr_kfac:.6f}")

#         # Check specific elements
#         print(f"  Sample elements (5 largest by GNH magnitude):")
#         top_idx = jnp.argsort(jnp.abs(gnh_diag))[-5:]
#         for idx in top_idx:
#             idx = int(idx)
#             print(f"    [{idx}]: GNH={gnh_diag[idx]:.6e}, EKFAC={ekfac_diag[idx]:.6e}, "
#                   f"KFAC={kfac_diag[idx]:.6e}, E/G ratio={ekfac_diag[idx]/(gnh_diag[idx]+1e-15):.4f}")

#     print("="*70)
#     assert False


# def test_ekfac_eigenvalue_overlap_paper_metric(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """
#     Test eigenvalue overlap as defined in the paper (Equation 16).

#     EvalOverlap(Ĝ, BG) = 1 - ||sort(λ(Ĝ)) - sort(λ(BG))||_2 / ||sort(λ(BG))||_2

#     This compares the SORTED eigenvalues of the approximation (KFAC/EKFAC)
#     with the SORTED eigenvalues of the block-diagonal GNH.

#     Key insight: This metric compares eigenvalue SPECTRA, not element-wise
#     diagonal values. Even if EKFAC has the correct diagonal in the eigenbasis,
#     the full matrix eigenvalues may differ.
#     """

#     # Get the GNH
#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=0.0)

#     # Build approximations
#     ekfac = EKFACComputer(compute_context=collector_data_all_classes).build()
#     kfac = KFACComputer(compute_context=collector_data_all_classes).build()

#     K = kfac.estimate_hessian(damping=0.0)
#     E = ekfac.estimate_hessian(damping=0.0)

#     layer_names = collector_data_all_classes.layer_names

#     # Get boundaries
#     boundaries = [0]
#     for layer in layer_names:
#         I = ekfac.precomputed_data.activation_eigenvectors[layer].shape[0]
#         O = ekfac.precomputed_data.gradient_eigenvectors[layer].shape[0]
#         boundaries.append(boundaries[-1] + I * O)

#     print("\n" + "="*70)
#     print("Eigenvalue Overlap (Paper Metric - Equation 16)")
#     print("EvalOverlap = 1 - ||sort(λ_approx) - sort(λ_BG)||_2 / ||sort(λ_BG)||_2")
#     print("="*70)

#     total_overlap_kfac = 0.0
#     total_overlap_ekfac = 0.0
#     total_params = 0

#     for i, layer in enumerate(layer_names):
#         start, end = boundaries[i], boundaries[i+1]
#         layer_size = end - start

#         # Extract blocks
#         G_block = G[start:end, start:end]  # Block-diagonal GNH (BG)
#         K_block = K[start:end, start:end]  # KFAC approximation
#         E_block = E[start:end, start:end]  # EKFAC approximation

#         # Compute eigenvalues and sort them
#         lambda_G = jnp.sort(jnp.linalg.eigvalsh(G_block))
#         lambda_K = jnp.sort(jnp.linalg.eigvalsh(K_block))
#         lambda_E = jnp.sort(jnp.linalg.eigvalsh(E_block))

#         # Compute overlap metric (Equation 16)
#         norm_G = jnp.linalg.norm(lambda_G)

#         if norm_G > 1e-10:
#             overlap_kfac = 1 - jnp.linalg.norm(lambda_K - lambda_G) / norm_G
#             overlap_ekfac = 1 - jnp.linalg.norm(lambda_E - lambda_G) / norm_G
#         else:
#             overlap_kfac = 1.0
#             overlap_ekfac = 1.0

#         # Weighted accumulation (Equation 17)
#         total_overlap_kfac += overlap_kfac * layer_size
#         total_overlap_ekfac += overlap_ekfac * layer_size
#         total_params += layer_size

#         print(f"\nLayer: {layer} (dim={layer_size})")
#         print(f"  Eigenvalue Overlap:")
#         print(f"    KFAC:  {overlap_kfac:.6f}")
#         print(f"    EKFAC: {overlap_ekfac:.6f}")
#         print(f"    Improvement: {(overlap_ekfac - overlap_kfac):.6f} ({(overlap_ekfac - overlap_kfac) * 100:.2f} pp)")

#         # Additional diagnostics
#         print(f"  Eigenvalue stats (sorted):")
#         print(f"    BG:    min={lambda_G[0]:.6e}, max={lambda_G[-1]:.6e}, mean={jnp.mean(lambda_G):.6e}")
#         print(f"    KFAC:  min={lambda_K[0]:.6e}, max={lambda_K[-1]:.6e}, mean={jnp.mean(lambda_K):.6e}")
#         print(f"    EKFAC: min={lambda_E[0]:.6e}, max={lambda_E[-1]:.6e}, mean={jnp.mean(lambda_E):.6e}")

#     # Aggregated overlap (Equation 17)
#     agg_overlap_kfac = total_overlap_kfac / total_params
#     agg_overlap_ekfac = total_overlap_ekfac / total_params

#     print("\n" + "-"*70)
#     print("Aggregated Eigenvalue Overlap (weighted by layer size):")
#     print(f"  KFAC:  {agg_overlap_kfac:.6f}")
#     print(f"  EKFAC: {agg_overlap_ekfac:.6f}")
#     print(f"  Improvement: {(agg_overlap_ekfac - agg_overlap_kfac):.6f} ({(agg_overlap_ekfac - agg_overlap_kfac) * 100:.2f} pp)")
#     print("="*70)

#     # EKFAC should have higher overlap than KFAC
#     assert agg_overlap_ekfac >= agg_overlap_kfac - 0.01, (
#         f"EKFAC should have at least as good eigenvalue overlap as KFAC. "
#         f"KFAC: {agg_overlap_kfac:.6f}, EKFAC: {agg_overlap_ekfac:.6f}"
#     )

# def test_ekfac_eigenvalue_ordering_check(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """
#     Check if there's an ordering mismatch between EKFAC corrections and the
#     Kronecker product structure used in Hessian reconstruction.
#     """

#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=0.0)
#     ekfac = EKFACComputer(compute_context=collector_data_all_classes).build()

#     layer = collector_data_all_classes.layer_names[0]

#     Q_A = ekfac.precomputed_data.activation_eigenvectors[layer]
#     Q_G = ekfac.precomputed_data.gradient_eigenvectors[layer]
#     ekfac_correction = ekfac.precomputed_data.eigenvalue_corrections[layer]

#     I, _ = Q_A.shape
#     O, _ = Q_G.shape

#     # Get GNH block
#     G_block = G[:I*O, :I*O]

#     # Transform to eigenbasis
#     Q_kron = jnp.kron(Q_A, Q_G)
#     G_eigenbasis = Q_kron.T @ G_block @ Q_kron
#     gnh_diag = jnp.diag(G_eigenbasis)

#     # EKFAC diagonal flattened
#     ekfac_diag = ekfac_correction.flatten()  # Row-major: [i*O + o] = correction[i, o]

#     print(f"Layer: {layer}, I={I}, O={O}")
#     print(f"\nDirect comparison (should match if ordering is correct):")
#     print(f"  GNH diag[:10]:   {gnh_diag[:10]}")
#     print(f"  EKFAC diag[:10]: {ekfac_diag[:10]}")
#     print(f"  Close: {jnp.allclose(gnh_diag, ekfac_diag, rtol=0.01)}")

#     # Try alternative ordering: column-major (Fortran order)
#     ekfac_diag_colmajor = ekfac_correction.flatten(order='F')  # [i + j*I] = correction[i, j]
#     print(f"\nColumn-major ordering:")
#     print(f"  GNH diag[:10]:   {gnh_diag[:10]}")
#     print(f"  EKFAC col-major[:10]: {ekfac_diag_colmajor[:10]}")
#     print(f"  Close: {jnp.allclose(gnh_diag, ekfac_diag_colmajor, rtol=0.01)}")

#     # Try transposed correction
#     ekfac_diag_transposed = ekfac_correction.T.flatten()
#     print(f"\nTransposed then flattened:")
#     print(f"  GNH diag[:10]:   {gnh_diag[:10]}")
#     print(f"  EKFAC transposed[:10]: {ekfac_diag_transposed[:10]}")
#     print(f"  Close: {jnp.allclose(gnh_diag, ekfac_diag_transposed, rtol=0.01)}")

#     # Check which ordering gives best match
#     rmse_rowmajor = jnp.sqrt(jnp.mean((gnh_diag - ekfac_diag)**2))
#     rmse_colmajor = jnp.sqrt(jnp.mean((gnh_diag - ekfac_diag_colmajor)**2))
#     rmse_transposed = jnp.sqrt(jnp.mean((gnh_diag - ekfac_diag_transposed)**2))

#     print(f"\nRMSE comparison:")
#     print(f"  Row-major (current): {rmse_rowmajor:.6e}")
#     print(f"  Column-major: {rmse_colmajor:.6e}")
#     print(f"  Transposed: {rmse_transposed:.6e}")

#     # Also check if kron(Q_G, Q_A) instead of kron(Q_A, Q_G) gives better results
#     Q_kron_alt = jnp.kron(Q_G, Q_A)
#     G_eigenbasis_alt = Q_kron_alt.T @ G_block @ Q_kron_alt
#     gnh_diag_alt = jnp.diag(G_eigenbasis_alt)

#     rmse_alt_rowmajor = jnp.sqrt(jnp.mean((gnh_diag_alt - ekfac_diag)**2))
#     rmse_alt_transposed = jnp.sqrt(jnp.mean((gnh_diag_alt - ekfac_diag_transposed)**2))

#     print(f"\nWith kron(Q_G, Q_A) instead of kron(Q_A, Q_G):")
#     print(f"  vs row-major: {rmse_alt_rowmajor:.6e}")
#     print(f"  vs transposed: {rmse_alt_transposed:.6e}")

#     assert False


# def test_eigenvalue_distribution_analysis(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """
#     Analyze the eigenvalue distributions more carefully.
#     """

#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=0.0)
#     ekfac = EKFACComputer(compute_context=collector_data_all_classes).build()
#     kfac = KFACComputer(compute_context=collector_data_all_classes).build()

#     K = kfac.estimate_hessian(damping=0.0)
#     E = ekfac.estimate_hessian(damping=0.0)

#     layer_names = collector_data_all_classes.layer_names

#     boundaries = [0]
#     for layer in layer_names:
#         I = ekfac.precomputed_data.activation_eigenvectors[layer].shape[0]
#         O = ekfac.precomputed_data.gradient_eigenvectors[layer].shape[0]
#         boundaries.append(boundaries[-1] + I * O)

#     print("\n" + "="*80)
#     print("Eigenvalue Distribution Analysis")
#     print("="*80)

#     for i, layer in enumerate(layer_names):
#         start, end = boundaries[i], boundaries[i+1]

#         G_block = G[start:end, start:end]
#         K_block = K[start:end, start:end]
#         E_block = E[start:end, start:end]

#         # Get eigenvalues (these are what the overlap metric uses)
#         eig_G = jnp.sort(jnp.linalg.eigvalsh(G_block))
#         eig_K = jnp.sort(jnp.linalg.eigvalsh(K_block))
#         eig_E = jnp.sort(jnp.linalg.eigvalsh(E_block))

#         print(f"\nLayer: {layer} (dim={end-start})")

#         # Count near-zero eigenvalues
#         threshold = 1e-10
#         n_zero_G = jnp.sum(jnp.abs(eig_G) < threshold)
#         n_zero_K = jnp.sum(jnp.abs(eig_K) < threshold)
#         n_zero_E = jnp.sum(jnp.abs(eig_E) < threshold)

#         print(f"  Near-zero eigenvalues (|λ| < {threshold}):")
#         print(f"    BG: {n_zero_G}, KFAC: {n_zero_K}, EKFAC: {n_zero_E}")

#         # Look at the non-zero eigenvalue distributions
#         nonzero_G = eig_G[jnp.abs(eig_G) >= threshold]
#         nonzero_K = eig_K[jnp.abs(eig_K) >= threshold]
#         nonzero_E = eig_E[jnp.abs(eig_E) >= threshold]

#         print(f"  Non-zero eigenvalue stats:")
#         print(f"    BG:    n={len(nonzero_G)}, min={jnp.min(nonzero_G):.4e}, max={jnp.max(nonzero_G):.4e}, mean={jnp.mean(nonzero_G):.4e}")
#         print(f"    KFAC:  n={len(nonzero_K)}, min={jnp.min(nonzero_K):.4e}, max={jnp.max(nonzero_K):.4e}, mean={jnp.mean(nonzero_K):.4e}")
#         print(f"    EKFAC: n={len(nonzero_E)}, min={jnp.min(nonzero_E):.4e}, max={jnp.max(nonzero_E):.4e}, mean={jnp.mean(nonzero_E):.4e}")

#         # Compare top eigenvalues (most important for the metric)
#         print(f"  Top 10 eigenvalues:")
#         print(f"    BG:    {eig_G[-10:]}")
#         print(f"    KFAC:  {eig_K[-10:]}")
#         print(f"    EKFAC: {eig_E[-10:]}")

#         # Compute overlap only on non-zero part
#         if len(nonzero_G) > 0:
#             # Pad to same length if needed
#             max_len = max(len(nonzero_G), len(nonzero_K), len(nonzero_E))

#             def pad_sorted(arr, length):
#                 if len(arr) < length:
#                     return jnp.concatenate([jnp.zeros(length - len(arr)), arr])
#                 return arr

#             nonzero_G_padded = pad_sorted(nonzero_G, max_len)
#             nonzero_K_padded = pad_sorted(nonzero_K, max_len)
#             nonzero_E_padded = pad_sorted(nonzero_E, max_len)

#             overlap_K_nonzero = 1 - jnp.linalg.norm(nonzero_K_padded - nonzero_G_padded) / jnp.linalg.norm(nonzero_G_padded)
#             overlap_E_nonzero = 1 - jnp.linalg.norm(nonzero_E_padded - nonzero_G_padded) / jnp.linalg.norm(nonzero_G_padded)

#             print(f"  Eigenvalue overlap (non-zero only):")
#             print(f"    KFAC: {overlap_K_nonzero:.6f}, EKFAC: {overlap_E_nonzero:.6f}")

#         # Check: is the issue that EKFAC changes the RANK?
#         rank_G = jnp.linalg.matrix_rank(G_block)
#         rank_K = jnp.linalg.matrix_rank(K_block)
#         rank_E = jnp.linalg.matrix_rank(E_block)
#         print(f"  Matrix rank: BG={rank_G}, KFAC={rank_K}, EKFAC={rank_E}")

#         # Check condition number
#         cond_G = jnp.max(jnp.abs(eig_G)) / (jnp.min(jnp.abs(eig_G[jnp.abs(eig_G) > 1e-15])) + 1e-15)
#         cond_K = jnp.max(jnp.abs(eig_K)) / (jnp.min(jnp.abs(eig_K[jnp.abs(eig_K) > 1e-15])) + 1e-15)
#         cond_E = jnp.max(jnp.abs(eig_E)) / (jnp.min(jnp.abs(eig_E[jnp.abs(eig_E) > 1e-15])) + 1e-15)
#         print(f"  Condition number: BG={cond_G:.2e}, KFAC={cond_K:.2e}, EKFAC={cond_E:.2e}")

#     print("="*80)
#     assert False

# def test_off_diagonal_contribution_to_eigenvalues(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """
#     Verify that off-diagonal elements in the Kronecker eigenbasis
#     are responsible for the larger true eigenvalues.
#     """

#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=0.0)
#     ekfac = EKFACComputer(compute_context=collector_data_all_classes).build()

#     layer_names = collector_data_all_classes.layer_names

#     boundaries = [0]
#     for layer in layer_names:
#         I = ekfac.precomputed_data.activation_eigenvectors[layer].shape[0]
#         O = ekfac.precomputed_data.gradient_eigenvectors[layer].shape[0]
#         boundaries.append(boundaries[-1] + I * O)

#     print("\n" + "="*80)
#     print("Off-diagonal contribution analysis")
#     print("="*80)

#     for i, layer in enumerate(layer_names):
#         start, end = boundaries[i], boundaries[i+1]

#         Q_A = ekfac.precomputed_data.activation_eigenvectors[layer]
#         Q_G = ekfac.precomputed_data.gradient_eigenvectors[layer]
#         Q_kron = jnp.kron(Q_A, Q_G)

#         # Get GNH block and transform to Kronecker eigenbasis
#         G_block = G[start:end, start:end]
#         G_eigenbasis = Q_kron.T @ G_block @ Q_kron

#         # Separate diagonal and off-diagonal
#         G_diag_only = jnp.diag(jnp.diag(G_eigenbasis))
#         G_offdiag = G_eigenbasis - G_diag_only

#         # Eigenvalues of different components
#         eig_full = jnp.sort(jnp.linalg.eigvalsh(G_eigenbasis))
#         eig_diag_only = jnp.sort(jnp.diag(G_eigenbasis))  # These ARE the eigenvalues when off-diag=0

#         # What EKFAC computes
#         ekfac_diag = ekfac.precomputed_data.eigenvalue_corrections[layer].flatten()
#         ekfac_diag_sorted = jnp.sort(ekfac_diag)

#         print(f"\nLayer: {layer}")
#         print(f"  Top 5 eigenvalues comparison:")
#         print(f"    Full GNH (with off-diag):  {eig_full[-5:]}")
#         print(f"    Diagonal only (no off-diag): {eig_diag_only[-5:]}")
#         print(f"    EKFAC (should match diag):  {ekfac_diag_sorted[-5:]}")

#         # The key question: does off-diagonal INCREASE the top eigenvalues?
#         top_increase = eig_full[-5:] - eig_diag_only[-5:]
#         print(f"    Increase from off-diag: {top_increase}")
#         print(f"    Mean increase: {jnp.mean(top_increase):.6f}")

#         # Off-diagonal Frobenius norm
#         offdiag_norm = jnp.linalg.norm(G_offdiag, 'fro')
#         diag_norm = jnp.linalg.norm(jnp.diag(G_eigenbasis))
#         print(f"  Off-diagonal / Diagonal Frobenius norm: {offdiag_norm / diag_norm:.4f}")

#         # Compute what eigenvalue overlap WOULD be if we used diagonal-only GNH
#         eig_kfac = jnp.sort(jnp.linalg.eigvalsh(
#             KFACComputer(compute_context=collector_data_all_classes).build().estimate_hessian(damping=0.0)[start:end, start:end]
#         ))

#         # Overlap with full GNH
#         overlap_ekfac_vs_full = 1 - jnp.linalg.norm(ekfac_diag_sorted - eig_full) / jnp.linalg.norm(eig_full)
#         overlap_kfac_vs_full = 1 - jnp.linalg.norm(eig_kfac - eig_full) / jnp.linalg.norm(eig_full)

#         # Overlap with diagonal-only GNH (this is what EKFAC is actually approximating!)
#         overlap_ekfac_vs_diag = 1 - jnp.linalg.norm(ekfac_diag_sorted - eig_diag_only) / jnp.linalg.norm(eig_diag_only)
#         overlap_kfac_vs_diag = 1 - jnp.linalg.norm(eig_kfac - eig_diag_only) / jnp.linalg.norm(eig_diag_only)

#         print(f"  Eigenvalue overlap vs FULL GNH:")
#         print(f"    KFAC: {overlap_kfac_vs_full:.6f}, EKFAC: {overlap_ekfac_vs_full:.6f}")
#         print(f"  Eigenvalue overlap vs DIAGONAL-ONLY GNH (fair comparison):")
#         print(f"    KFAC: {overlap_kfac_vs_diag:.6f}, EKFAC: {overlap_ekfac_vs_diag:.6f}")

#     print("="*80)
#     assert False

# def test_kronecker_eigenbasis_vs_true_eigenbasis(
#     model_context: ModelContext,
#     collector_data_all_classes: DataActivationsGradients,
# ):
#     """
#     Compare the Kronecker eigenbasis (Q_A ⊗ Q_G) with the true eigenbasis of GNH.

#     This tests the fundamental assumption: are the eigenvectors of the
#     block-diagonal GNH separable as Kronecker products?
#     """

#     G = GNHComputer(compute_context=model_context).build().estimate_hessian(damping=0.0)
#     ekfac = EKFACComputer(compute_context=collector_data_all_classes).build()

#     layer_names = collector_data_all_classes.layer_names

#     # Get boundaries
#     boundaries = [0]
#     for layer in layer_names:
#         I = ekfac.precomputed_data.activation_eigenvectors[layer].shape[0]
#         O = ekfac.precomputed_data.gradient_eigenvectors[layer].shape[0]
#         boundaries.append(boundaries[-1] + I * O)

#     print("\n" + "="*80)
#     print("Kronecker Eigenbasis vs True GNH Eigenbasis Comparison")
#     print("="*80)

#     for i, layer in enumerate(layer_names):
#         start, end = boundaries[i], boundaries[i+1]

#         Q_A = ekfac.precomputed_data.activation_eigenvectors[layer]
#         Q_G = ekfac.precomputed_data.gradient_eigenvectors[layer]

#         # Kronecker eigenbasis (what KFAC/EKFAC use)
#         Q_kron = jnp.kron(Q_A, Q_G)  # (I*O, I*O)

#         # True eigenbasis of GNH block
#         G_block = G[start:end, start:end]
#         lambda_true, Q_true = jnp.linalg.eigh(G_block)  # True eigenvectors

#         print(f"\nLayer: {layer} (dim={end-start})")

#         # Test 1: Subspace overlap
#         # If Kronecker basis spans same subspace, Q_kron^T Q_true should have large singular values
#         overlap_matrix = Q_kron.T @ Q_true
#         singular_values = jnp.linalg.svd(overlap_matrix, compute_uv=False)

#         print(f"  Subspace overlap (singular values of Q_kron^T @ Q_true):")
#         print(f"    Top 10: {singular_values[:10]}")
#         print(f"    Bottom 10: {singular_values[-10:]}")
#         print(f"    Mean: {jnp.mean(singular_values):.6f}")

#         # If eigenbases aligned, top singular values should be ~1
#         alignment_score = jnp.mean(singular_values[:10])
#         print(f"    Alignment score (mean of top 10): {alignment_score:.6f}")

#         # Test 2: Check if any true eigenvector is approximately separable
#         # For each true eigenvector, find best Kronecker approximation
#         separability_scores = []
#         for j in range(min(20, Q_true.shape[1])):  # Check first 20 eigenvectors
#             v_true = Q_true[:, j].reshape(-1, 1)  # (I*O, 1)

#             # Best rank-1 Kronecker approximation via SVD
#             V_reshaped = v_true.reshape(Q_A.shape[0], Q_G.shape[0])
#             U, s, Vt = jnp.linalg.svd(V_reshaped, full_matrices=False)

#             # Rank-1 approximation
#             V_rank1 = s[0] * jnp.outer(U[:, 0], Vt[0, :])
#             v_kron_approx = V_rank1.flatten().reshape(-1, 1)

#             # Measure how close v_true is to a rank-1 Kronecker product
#             separability = jnp.linalg.norm(v_true - v_kron_approx) / jnp.linalg.norm(v_true)
#             separability_scores.append(separability)

#         print(f"\n  Separability of true eigenvectors (0 = perfectly separable):")
#         print(f"    Mean error: {jnp.mean(jnp.array(separability_scores)):.6f}")
#         print(f"    Min error: {jnp.min(jnp.array(separability_scores)):.6f}")
#         print(f"    Max error: {jnp.max(jnp.array(separability_scores)):.6f}")

#         # Test 3: Diagonalization quality
#         # How well does Kronecker basis diagonalize GNH?
#         G_in_kron_basis = Q_kron.T @ G_block @ Q_kron
#         G_in_true_basis = Q_true.T @ G_block @ Q_true  # Should be exactly diagonal

#         off_diag_kron = G_in_kron_basis - jnp.diag(jnp.diag(G_in_kron_basis))
#         off_diag_true = G_in_true_basis - jnp.diag(jnp.diag(G_in_true_basis))

#         off_diag_norm_kron = jnp.linalg.norm(off_diag_kron, 'fro')
#         off_diag_norm_true = jnp.linalg.norm(off_diag_true, 'fro')
#         diag_norm = jnp.linalg.norm(jnp.diag(G_in_kron_basis))

#         print(f"\n  Diagonalization quality:")
#         print(f"    True basis off-diag norm: {off_diag_norm_true:.6e} (should be ~0)")
#         print(f"    Kron basis off-diag norm: {off_diag_norm_kron:.6e}")
#         print(f"    Off-diag / Diag ratio (Kron): {off_diag_norm_kron / diag_norm:.4f}")

#         # Test 4: Eigenvector alignment for top eigenvalues
#         # Compare direction of eigenvectors corresponding to largest eigenvalues
#         top_k = 5
#         idx_true_top = jnp.argsort(lambda_true)[-top_k:]

#         print(f"\n  Top {top_k} eigenvector alignment:")
#         for j in idx_true_top:
#             v_true = Q_true[:, j]

#             # Project onto Kronecker subspace
#             v_proj = Q_kron @ (Q_kron.T @ v_true)

#             # Measure alignment
#             cosine_sim = jnp.abs(jnp.dot(v_true, v_proj)) / (jnp.linalg.norm(v_true) * jnp.linalg.norm(v_proj))
#             projection_error = jnp.linalg.norm(v_true - v_proj) / jnp.linalg.norm(v_true)

#             print(f"    Eigval {lambda_true[j]:.6e}: cosine={cosine_sim:.6f}, proj_error={projection_error:.6f}")

#     print("="*80)
#     assert False
