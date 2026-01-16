from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import kfac_jax
import optax
import pytest
from jax.random import PRNGKey
from optax import softmax_cross_entropy

from src.config import (
    DatasetEnum,
    LossType,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import (
    generate_pseudo_targets,
)
from src.utils.data.data import Dataset, DownloadableDataset, RandomClassificationDataset
from src.utils.loss import get_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def config(tmp_path_factory):
    """Create model configuration for testing."""
    base = tmp_path_factory.mktemp("kfac_jax_tests")
    

    return ModelConfig(
        architecture=ModelArchitecture.MLP,
        input_dim=64,  # Will be updated from dataset
        hidden_dim=[16, 16],
        output_dim=10,  # Will be updated from dataset
        loss=LossType.CROSS_ENTROPY,
        training=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer=OptimizerType.ADAMW,
            epochs=200,
            batch_size=128,
        ),
        directory=str(base / "model"),
    )


@pytest.fixture(params=["random", "digits"], scope="session")
def dataset(request) -> Dataset:
    """Create a random classification dataset for testing."""
    if request.param == "random":
        return RandomClassificationDataset(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_classes=2,
            seed=123,
        )
    elif request.param == "digits":
        return DownloadableDataset.load(
            dataset=DatasetEnum.DIGITS, directory="./experiments/datasets/digits"
        ) 
    return DownloadableDataset.load(
        dataset=DatasetEnum.DIGITS, directory="./experiments/datasets/digits"
    )


@pytest.fixture(scope="session")
def model_params_loss(
    config: ModelConfig, dataset: Dataset
) -> Tuple[ApproximationModel, Dict, Callable]:
    """Train a model and return it with its parameters and loss function."""
    # Update dimensions from dataset
    config.input_dim = dataset.input_dim()
    config.output_dim = dataset.output_dim()

    # Get model from registry
    model = ModelRegistry.get_model(model_config=config)

    # Train the model
    model, params, _ = train_model(
        model,
        dataset.get_dataloader(batch_size=config.training.batch_size, seed=123),
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
    use_pseudo_targets: bool = False,
) -> Tuple[DataActivationsGradients, DataActivationsGradients]:
    """Collect EK-FAC data with two runs."""
    model, params, loss = model_params_loss

    base_dir = config.directory
    assert base_dir is not None, "Model directory must be set"

    run1_dir = f"{base_dir}/{dataset.__class__.__name__}/collector_{run_suffix}/run1"
    run2_dir = f"{base_dir}/{dataset.__class__.__name__}/collector_{run_suffix}/run2"

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
@pytest.mark.parametrize("dataset", ["digits", "random"], indirect=True)
def test_kfac_jax_consistency(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
    collector_data_single: Tuple[DataActivationsGradients, DataActivationsGradients],
):
    """Test that EK-FAC Hessian estimates are consistent between single batch and batched data collection."""
    model, params, loss_fn_loaded = model_params_loss

    kfac_computer = KFACComputer(collector_data_single).build()
    kfac_hessian = kfac_computer.estimate_hessian()

    def loss_fn(params, batch):
        inputs, targets = batch
        predictions = model.apply(params, inputs)
        # Register your loss with KFAC
        kfac_jax.register_softmax_cross_entropy_loss(predictions, targets)
        log_p = jax.nn.log_softmax(predictions, axis=-1)
        loss = - jax.vmap(lambda x, y: x[y])(log_p, targets)
        return loss

    # Prepare your batch
    inputs = dataset.inputs
    targets = dataset.targets
    batch = (inputs, targets)
    batch_size = inputs.shape[0]

    # Create the KFAC estimator
    kf_estimator = kfac_jax.BlockDiagonalCurvature(
        func=loss_fn,
        layer_tag_to_block_ctor=dict(
            dense=kfac_jax.DenseTwoKroneckerFactored,
            conv2d=None,
            scale_and_shift=kfac_jax.ScaleAndShiftFull,
        ),
    )

    # Initialize and compute KFAC approximation
    rng = jax.random.PRNGKey(42)
    func_args = (params, batch)

    kf_state = kf_estimator.init(
        rng=rng,
        func_args=func_args,
        exact_powers_to_cache=None,
        approx_powers_to_cache=None,
        cache_eigenvalues=False,
    )

    kf_state = kf_estimator.update_curvature_matrix_estimate(
        state=kf_state,
        ema_old=0.0,
        ema_new=1.0,
        identity_weight=0.0,
        batch_size=batch_size,
        rng=rng,
        func_args=func_args,
        estimation_mode="fisher_empirical",
    )

    # Get the KFAC blocks - this returns a list/tuple, not a dict
    kfac_blocks = kf_estimator.to_diagonal_block_dense_matrix(kf_state)
    
    # FIX: Use the blocks directly (it's a list, not a dict)
    kfac_hessian_jax = jax.scipy.linalg.block_diag(*kfac_blocks)

    print("KFAC Hessian from Collector:", kfac_hessian.shape)
    print("KFAC Hessian from kfac_jax:", kfac_hessian_jax.shape)
    
    # # plot heatmaps for visual comparison
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("KFAC Hessian from Collector")
    # plt.imshow(kfac_hessian, aspect='auto', cmap='viridis')
    # plt.colorbar() 
    # plt.subplot(1, 2, 2)
    # plt.title("KFAC Hessian from kfac_jax")
    # plt.imshow(kfac_hessian_jax, aspect='auto', cmap='viridis')
    # plt.colorbar()
    # # save the figure
    # plt.savefig(f"kfac_hessian_comparison_{dataset.__class__.__name__}.png")
    # plt.close()    
    
    # # plot the eigenspectra for visual comparison
    # eigs_collector = jnp.linalg.eigvalsh(kfac_hessian)
    # eigs_kfac_jax = jnp.linalg.eigvalsh(kfac_hessian_jax)
    # plt.figure(figsize=(8, 6))
    # plt.plot(eigs_collector, label='Collector KFAC Hessian', marker='o')
    # plt.plot(eigs_kfac_jax, label='kfac_jax KFAC Hessian', marker='x')
    # plt.title("Eigenspectra Comparison")
    # plt.xlabel("Index")
    # plt.ylabel("Eigenvalue")
    # plt.legend()
    # # save the figure
    # plt.savefig(f"kfac_hessian_eigenspectra_comparison_{dataset.__class__.__name__}.png")
    # plt.close()
    
    # # plot eigenvectors as heatmaps for visual comparison
    # _, vecs_collector = jnp.linalg.eigh(kfac_hessian)
    # _, vecs_kfac_jax = jnp.linalg.eigh(kfac_hessian_jax)
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Eigenvectors from Collector KFAC Hessian")
    # plt.imshow(vecs_collector, aspect='auto', cmap='viridis')
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.title("Eigenvectors from kfac_jax KFAC Hessian")
    # plt.imshow(vecs_kfac_jax, aspect='auto', cmap='viridis')
    # plt.colorbar()
    # # save the figure
    # plt.savefig(f"kfac_hessian_eigenvectors_comparison_{dataset.__class__.__name__}.png")
    # plt.close()

    # Compute and print metrics
    metrics = [
        FullMatrixMetric.RELATIVE_FROBENIUS,
        FullMatrixMetric.RELATIVE_SPECTRAL,
    ]
    
    for metric in metrics:
        value = metric.compute(kfac_hessian, kfac_hessian_jax)
        print(f"{metric.name} between KFAC Hessians: {value:.6e}")
    assert FullMatrixMetric.RELATIVE_FROBENIUS.compute(kfac_hessian, kfac_hessian_jax) < 1e-3
    
    
def test_gradients_close(
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
):
    """Test that gradients computed with kfac_jax loss registration match standard loss gradients."""
    model, params, _ = model_params_loss
    inputs = dataset.inputs
    targets = dataset.targets
    
    def loss_no_kfac(params):
        logits = model.apply(params, inputs)
        
        assert isinstance(logits, jnp.ndarray), "Model predictions must be a jnp.ndarray."
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).mean()

    def loss_with_kfac(params):
        logits = model.apply(params, inputs)
        assert isinstance(logits, jnp.ndarray), "Model predictions must be a jnp.ndarray."
        kfac_jax.register_softmax_cross_entropy_loss(logits, targets)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        return (-jax.vmap(lambda x, y: x[y])(log_p, targets)).mean()

    g1 = jax.grad(loss_no_kfac)(params)
    g2 = jax.grad(loss_with_kfac)(params)

    jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b, rtol=1e-5, atol=1e-5), g1, g2)
)
