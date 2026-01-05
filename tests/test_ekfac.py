from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util
from jax.random import PRNGKey
from jaxtyping import Float

from src.config import Config, HessianApproximationConfig, ModelConfig
from src.hessians.approximator.ekfac import EKFACApproximator
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import EKFACData, ModelContext
from src.hessians.utils.pseudo_targets import (
    generate_pseudo_targets,
    sample_gradients,
)
from src.utils.data.data import Dataset, RandomClassificationDataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import cross_entropy_loss
from src.utils.models.linear_model import LinearModel
from src.utils.models.mlp import MLP
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(params=["linear", "multi_layer"], scope="session")
def config(request, tmp_path_factory):
    base = tmp_path_factory.mktemp(request.param)

    hidden_dim = [] if request.param == "linear" else [5]

    return Config(
        dataset_path="random",
        seed=123,
        model=ModelConfig(
            model_name="test",
            directory=str(base / "model"),
            metadata={"hidden_dim": hidden_dim, "activation": "relu"},
        ),
        hessian_approximation=HessianApproximationConfig(
            method="EKFAC",
            directory=str(base / "hessian"),
        ),
    )


@pytest.fixture(scope="session")
def dataset(config: Config) -> Dataset:
    return RandomClassificationDataset(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=2,
        seed=config.seed,
    )


@pytest.fixture(scope="session")
def model_and_params(config: Config, dataset: Dataset):
    config.model.metadata = config.model.metadata or {}
    hidden_dim = config.model.metadata["hidden_dim"]

    if hidden_dim:
        model = MLP(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=hidden_dim,
            activation="relu",
            seed=config.seed,
        )
    else:
        model = LinearModel(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=[],
            seed=config.seed,
        )

    model, params, _ = train_model(
        model,
        dataset.get_dataloader(
            batch_size=JAXDataLoader.get_batch_size(), seed=config.seed
        ),
        loss_fn=cross_entropy_loss,
        optimizer=optimizer("sgd", lr=1e-3),
        epochs=30,
    )

    return model, params


@pytest.fixture(scope="session")
def model_context(dataset: Dataset, model_and_params: Tuple[MLP, Dict]):
    model, params = model_and_params
    return ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=cross_entropy_loss,
    )


@pytest.fixture(scope="session")
def ekfac_data(
    config: Config, model_and_params: Tuple[MLP, Dict], model_context: ModelContext
):
    base = config.hessian_approximation.directory
    assert base is not None, "Hessian approximation directory must be set in config"
    collector_dir = base + "/collector_ekfac_single"

    # if directory already exists, skip collection
    try:
        EKFACApproximator.load_data(base)
        return EKFACApproximator.load_data(base)[0]
    except FileNotFoundError:
        pass

    collector = CollectorActivationsGradients(
        model=model_and_params[0],
        params=model_and_params[1],
        loss_fn=cross_entropy_loss,
    )

    assert model_context.targets is not None, "ModelContext targets must not be None"
    collector.collect(
        inputs=model_context.inputs,
        targets=model_context.targets,
        save_directory=collector_dir,
    )

    ekfac_approximator = EKFACApproximator(collector_dir, collector_dir)
    ekfac_approximator.build(config, base)
    ekfac_data, _ = EKFACApproximator.load_data(base)

    assert isinstance(ekfac_data, EKFACData)
    return ekfac_data


@pytest.fixture(scope="session")
def ekfac_data_batched(
    config: Config, model_and_params: Tuple[MLP, Dict], model_context: ModelContext
):
    base = config.hessian_approximation.directory
    assert base is not None, "Hessian approximation directory must be set in config"
    collector_dir = base + "/collector_ekfac_batched"

    # if directory already exists, skip collection
    try:
        EKFACApproximator.load_data(base)
        return EKFACApproximator.load_data(base)[0]
    except FileNotFoundError:
        pass

    collector = CollectorActivationsGradients(
        model=model_and_params[0],
        params=model_and_params[1],
        loss_fn=cross_entropy_loss,
    )

    assert model_context.targets is not None, "ModelContext targets must not be None"
    collector.collect(
        inputs=model_context.inputs,
        targets=model_context.targets,
        save_directory=collector_dir,
        batch_size=128,
    )

    ekfac_approximator = EKFACApproximator(collector_dir, collector_dir)
    ekfac_approximator.build(config, base)
    ekfac_data, _ = EKFACApproximator.load_data(base)

    assert isinstance(ekfac_data, EKFACData)
    return ekfac_data


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def compute_full_implicit_matrices(
    computer: HessianEstimator, dim: int, damping: Optional[Float]
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
    config: Config, model_and_params: Tuple[MLP, Dict], dataset: Dataset
):
    """Test if Hessian approximation can be computed without errors.
    Replicates the proper application scenario with pseudo-targets.

    This test also ensures that the data is loaded initially"""
    model, params = model_and_params

    data1 = dataset.replace_targets(
        generate_pseudo_targets(
            model=model,
            inputs=dataset.inputs,
            params=params,
            loss_fn=cross_entropy_loss,
            rng_key=PRNGKey(0),
        )
    )
    data2 = dataset.replace_targets(
        generate_pseudo_targets(
            model=model,
            inputs=dataset.inputs,
            params=params,
            loss_fn=cross_entropy_loss,
            rng_key=PRNGKey(1),
        )
    )

    base = config.hessian_approximation.directory
    assert base is not None, "Hessian approximation directory must be set in config"
    run1, run2 = base + "/run1", base + "/run2"

    collector = CollectorActivationsGradients(model, params, loss_fn=cross_entropy_loss)
    collector.collect(
        inputs=data1.inputs,
        targets=data1.targets,
        save_directory=run1,
    )
    collector.collect(
        inputs=data2.inputs,
        targets=data2.targets,
        save_directory=run2,
    )

    EKFACApproximator(run1, run2).build(config, base)
    ekfac_data, _ = EKFACApproximator.load_data(base)

    assert isinstance(ekfac_data, EKFACData)
    H = EKFACComputer(ekfac_data).estimate_hessian(damping=1e-3)
    assert jnp.isfinite(H).all()


def test_gradient_consistency(
    config: Config, model_and_params: Tuple[MLP, Dict], dataset: Dataset
):
    """
    Verify E-KFAC collector vs true gradients (sanity test).
    The following should hold for each linear layer l:

        ∇_{W_l} log p(y | x; θ) =  a_{l-1}^T s_l

    where:
        - a_{l-1} = activations of the previous layer
        - s_l = preactivation gradients = ∇_{W_l a_{l-1}} log p(y | x; θ)
    """
    model, params = model_and_params

    loss_fn = cross_entropy_loss

    def loss_fn_apply(p):
        return loss_fn(model.apply(p, dataset.inputs), dataset.targets, reduction="sum")

    gt_grads = jax.grad(loss_fn_apply)(params)

    assert config.hessian_approximation.directory is not None, (
        "Hessian approximation directory must be set in config"
    )
    collector_dir = config.hessian_approximation.directory + "/collector"
    collector = CollectorActivationsGradients(model, params, loss_fn=loss_fn)
    collector.collect(
        inputs=dataset.inputs,
        targets=dataset.targets,
        save_directory=collector_dir,
    )

    collector_data = CollectorActivationsGradients.load(collector_dir)
    activations, gradients, layer_names = (
        collector_data.activations,
        collector_data.gradients,
        collector_data.layer_names,
    )

    for i, (layer, gt) in enumerate(gt_grads["params"].items()):
        assert layer == layer_names[i], "Layer names do not match"
        W_grad = gt["kernel"]
        a, g = activations[layer], gradients[layer]

        ag = jnp.einsum("ni,no->io", a, g)
        assert jnp.allclose(W_grad, ag, atol=1e-4)

        kron = sum(jnp.kron(a[i], g[i]) for i in range(a.shape[0])) / a.shape[0]
        assert jnp.allclose(W_grad.reshape(-1) / a.shape[0], kron, atol=1e-4)


def test_kfac_hessian(
    config: Config,
    model_context: ModelContext,
):
    """Test whether K-FAC Hessian can be computed without errors and matches dimensions of other methods."""
    damping = 0.1
    H = HessianComputer(model_context).compute_hessian(damping=damping)
    G = GNHComputer(model_context).estimate_hessian(damping=damping)

    base = config.hessian_approximation.directory
    assert base is not None, "Hessian approximation directory must be set in config"

    ekfac_data = EKFACApproximator.load_data(base)[0]

    assert isinstance(ekfac_data, EKFACData)
    K = KFACComputer(ekfac_data).estimate_hessian(damping=damping)

    assert H.shape == G.shape == K.shape


def test_kfac_via_kron_equals_eigenvector_method(
    config: Config, ekfac_data: EKFACData, model_context: ModelContext
):
    """This test verifies that the K-FAC Hessian computed via the Kronecker product of the covariances directly equals
    the K-FAC Hessian computed via the eigenvector method implemented in KFACComputer.

    It can also easily extend to compare with the true hessian in order to check whether the order A kron G vs. G kron A is correct.
    """
    base = config.hessian_approximation.directory
    assert base is not None, "Hessian approximation directory must be set in config"
    run1, run2 = base + "/run1", base + "/run2"
    ekfac_approximator = EKFACApproximator(run1, run2)

    collector_data = CollectorActivationsGradients.load(directory=run1)
    activations, gradients, layer_names = (
        collector_data.activations,
        collector_data.gradients,
        collector_data.layer_names,
    )

    covariances_activations, covariances_gradients = (
        ekfac_approximator.compute_covariances(
            activations=activations, gradients=gradients
        )
    )

    # compute hessian by the block diagonal hessian of the kronecker product of the covariances
    H_kron_blocks = {}
    for layer in layer_names:
        A_cov = covariances_activations[layer]
        G_cov = covariances_gradients[layer]
        H_kron_blocks[layer] = jnp.kron(A_cov, G_cov)

    kron_comparison_method_H = jax.scipy.linalg.block_diag(*H_kron_blocks.values())

    # compute hessian by the eigenvector method
    eigenvector_method = KFACComputer(ekfac_data)
    eigenvector_method_H = eigenvector_method.estimate_hessian(
        damping=0.0
    )  # to ensure everything is computed

    assert jnp.allclose(
        kron_comparison_method_H,
        eigenvector_method_H,
        atol=1e-5,
    )

    # Optionally, compare with true hessian if desired
    # true_hessian_H = HessianComputer(model_context).compute_hessian(damping=0.0)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.title("Kron Comparison Method H")
    # plt.imshow(kron_comparison_method_H, cmap="viridis")
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.title("Eigenvector Method H")
    # plt.imshow(eigenvector_method_H, cmap="viridis")
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.title("True Hessian H")
    # plt.imshow(true_hessian_H, cmap="viridis")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()


def test_ekfac_hessian(model_context: ModelContext, ekfac_data: EKFACData):
    damping = 0.1

    H = HessianComputer(model_context).compute_hessian(damping=damping)
    G = GNHComputer(model_context).estimate_hessian(damping=damping)

    E = EKFACComputer(ekfac_data).estimate_hessian(damping=damping)

    assert H.shape == G.shape == E.shape


def test_collector_batched_vs_single_consistency(
    ekfac_data: EKFACData,
    ekfac_data_batched: EKFACData,
):
    """Test whether EK-FAC data collected with different batch sizes is consistent."""
    for layer in ekfac_data.layer_names:
        A_single = ekfac_data.activation_eigenvectors[layer]
        G_single = ekfac_data.gradient_eigenvectors[layer]
        L_single = ekfac_data.eigenvalue_corrections[layer]

        A_batched = ekfac_data_batched.activation_eigenvectors[layer]
        G_batched = ekfac_data_batched.gradient_eigenvectors[layer]
        L_batched = ekfac_data_batched.eigenvalue_corrections[layer]

        assert jnp.allclose(A_single, A_batched, rtol=1e-6, atol=1e-5)
        assert jnp.allclose(G_single, G_batched, rtol=1e-6, atol=1e-5)
        assert jnp.allclose(L_single, L_batched, rtol=1e-6, atol=1e-5)

    # compute end to end hessian and compare
    damping = 0.1
    comp_single = EKFACComputer(ekfac_data)
    comp_batched = EKFACComputer(ekfac_data_batched)

    H_single = comp_single.estimate_hessian(damping)
    H_batched = comp_batched.estimate_hessian(damping)

    assert jnp.allclose(H_single, H_batched, rtol=1e-6, atol=1e-5)


def test_ekfac_hvp_ihvp_consistency(
    model_context: ModelContext,
    model_and_params: Tuple[MLP, Dict],
    ekfac_data: EKFACData,
):
    """
    Test whether the HVP and IHVP implementations are consistent
    in the sense of comparing it with multiplication of the full hessian / inverse hessian with the test vector.
    """
    comp = EKFACComputer(ekfac_data)
    damping = 0.1

    assert model_context.targets is not None, "ModelContext targets must not be None"
    v = sample_gradients(
        model=model_and_params[0],
        params=model_and_params[1],
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=cross_entropy_loss,
        rng_key=PRNGKey(0),
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


def test_kfac_hvp_ihvp_consistency(
    model_context: ModelContext,
    model_and_params: Tuple[MLP, Dict],
    ekfac_data: EKFACData,
):
    """Test whether the HVP and IHVP implementations are consistent
    in the sense of comparing it with multiplication of the full hessian / inverse hessian with the test vector."""
    comp = KFACComputer(ekfac_data)
    damping = 0.1

    assert model_context.targets is not None, "ModelContext targets must not be None"
    v = sample_gradients(
        model=model_and_params[0],
        params=model_and_params[1],
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=cross_entropy_loss,
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
    model_and_params: Tuple[MLP, Dict], ekfac_data: EKFACData
):
    """Test whether the EK-FAC Hessian explicit computation matches the implicit computation via HVPs / IHVPs on basis vectors."""
    comp = EKFACComputer(ekfac_data)
    damping = 0.1

    dim = flatten_util.ravel_pytree(model_and_params[1])[0].shape[0]

    H = comp.estimate_hessian(damping)
    Hinv = comp.estimate_inverse_hessian(damping)
    H_imp, Hinv_imp = compute_full_implicit_matrices(comp, dim, damping)

    assert jnp.allclose(H, H_imp, atol=1e-3)
    assert jnp.allclose(Hinv, Hinv_imp, atol=1e-3)


def test_kfac_explicit_vs_implicit_equivalence(
    model_and_params: Tuple[MLP, Dict], ekfac_data: EKFACData
):
    """Test whether the K-FAC Hessian explicit computation matches the implicit computation via HVPs / IHVPs on basis vectors."""
    comp = KFACComputer(ekfac_data)
    damping = 0.1

    dim = flatten_util.ravel_pytree(model_and_params[1])[0].shape[0]

    H = comp.estimate_hessian(damping)
    Hinv = comp.estimate_inverse_hessian(damping)

    H_imp, Hinv_imp = compute_full_implicit_matrices(comp, dim, damping)

    assert jnp.allclose(H, H_imp, atol=1e-3)
    assert jnp.allclose(Hinv, Hinv_imp, atol=1e-3)


def test_ekfac_ihvp_batched_shape_and_finiteness(
    model_context: ModelContext,
    model_and_params: Tuple[MLP, Dict],
    ekfac_data: EKFACData,
):
    """Test whether EK-FAC IHVP with batched vectors works and produces finite outputs of correct shape."""
    comp = EKFACComputer(ekfac_data)
    damping = 0.1

    assert model_context.targets is not None, "ModelContext targets must not be None"
    V = sample_gradients(
        model=model_and_params[0],
        params=model_and_params[1],
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=cross_entropy_loss,
        rng_key=PRNGKey(2),
        n_vectors=5,
    )

    IHVP = comp.estimate_ihvp(V, damping)

    assert IHVP.shape == V.shape
    assert jnp.isfinite(IHVP).all()


def test_ekfac_ihvp_batched_vs_single_consistency(
    model_context: ModelContext,
    model_and_params: Tuple[MLP, Dict],
    ekfac_data: EKFACData,
):
    """Test whether EK-FAC IHVP with batched vectors is consistent with single vector IHVP computation."""
    comp = EKFACComputer(ekfac_data)
    damping = 0.1

    assert model_context.targets is not None, "ModelContext targets must not be None"
    V = sample_gradients(
        model=model_and_params[0],
        params=model_and_params[1],
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=cross_entropy_loss,
        rng_key=PRNGKey(3),
        n_vectors=4,
    )

    IHVP_batch = comp.estimate_ihvp(V, damping)

    for i in range(V.shape[0]):
        IHVP_single = comp.estimate_ihvp(V[i], damping)
        assert jnp.allclose(IHVP_batch[i], IHVP_single, rtol=1e-6, atol=1e-4)


def test_ekfac_ihvp_hessian_roundtrip_batched(
    config: Config, model_context: ModelContext, model_and_params: Tuple[MLP, Dict]
):
    """Test whether EK-FAC IHVP with batched vectors is consistent in the sense of comparing it with multiplication of the full hessian / inverse hessian with the test vector."""
    base = config.hessian_approximation.directory
    assert base is not None

    ekfac_data = EKFACApproximator.load_data(base)[0]
    assert isinstance(ekfac_data, EKFACData)

    comp = EKFACComputer(ekfac_data)
    damping = 0.1

    assert model_context.targets is not None, "ModelContext targets must not be None"
    V = sample_gradients(
        model=model_and_params[0],
        params=model_and_params[1],
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=cross_entropy_loss,
        rng_key=PRNGKey(4),
        n_vectors=5,
    )

    H = comp.estimate_hessian(damping)
    IHVP = comp.estimate_ihvp(V, damping)

    roundtrip = (H @ IHVP.T).T
    assert jnp.allclose(roundtrip, V, rtol=1e-2, atol=1e-5)
