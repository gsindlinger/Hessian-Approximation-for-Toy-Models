from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import pytest
from jax.random import PRNGKey
from jaxtyping import Array

from src.config import (
    LossType,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    PseudoTargetGenerationStrategy,
    TrainingConfig,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.utils.data.data import Dataset, RandomClassificationDataset
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
    else:
        architecture = ModelArchitecture.MLP
        hidden_dim = [10]

    return ModelConfig(
        architecture=architecture,
        input_dim=-1,  # Will be updated from dataset
        hidden_dim=hidden_dim if hidden_dim else None,
        output_dim=-1,  # Will be updated from dataset
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


def _collect_fim_data(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
    suffix: str = "",
    pseudo_target_strategy: PseudoTargetGenerationStrategy = PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
    pseudo_target_repetitions: int = 1,
    rng_key: Optional[Array] = None,
) -> DataActivationsGradients:
    """Helper function to collect FIM data with different strategies."""
    base = config.directory
    assert base is not None, "Model directory must be set in config"

    collector_dir = f"{base}/collector{suffix}"
    model, params, loss = model_params_loss

    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss,
        pseudo_target_strategy=pseudo_target_strategy,
        pseudo_target_repetitions=pseudo_target_repetitions,
    )

    collector_data = collector.collect(
        dataset=dataset,
        save_directory=collector_dir,
        try_load=True,
        rng_key=rng_key,
    )

    assert collector_data is not None, "Failed to collect FIM data."
    return collector_data


@pytest.fixture(scope="session")
def fim_data_empirical(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> DataActivationsGradients:
    """Collect FIM data using EMPIRICAL_FISHER strategy."""
    return _collect_fim_data(
        config,
        model_params_loss,
        dataset,
        suffix="_empirical",
        pseudo_target_strategy=PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
        pseudo_target_repetitions=1,
    )


@pytest.fixture(scope="session")
def fim_data_mcmc(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> DataActivationsGradients:
    """Collect FIM data using MCMC strategy."""
    return _collect_fim_data(
        config,
        model_params_loss,
        dataset,
        suffix="_mcmc",
        pseudo_target_strategy=PseudoTargetGenerationStrategy.MCMC,
        pseudo_target_repetitions=5,
        rng_key=PRNGKey(42),
    )


@pytest.fixture(scope="session")
def fim_data_all_classes(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    dataset: Dataset,
) -> DataActivationsGradients:
    """Collect FIM data using ALL_CLASSES strategy."""
    return _collect_fim_data(
        config,
        model_params_loss,
        dataset,
        suffix="_all_classes",
        pseudo_target_strategy=PseudoTargetGenerationStrategy.ALL_CLASSES,
    )


# ---------------------------------------------------------------------
# Tests - Basic Functionality
# ---------------------------------------------------------------------


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_computation_empirical_fisher(
    fim_data_empirical: DataActivationsGradients,
):
    """Test that FIM block approximation matches full FIM for linear models (EMPIRICAL_FISHER)."""
    damping = 1e-5

    # Compute FIM block
    block_fim_computer = FIMBlockComputer(compute_context=fim_data_empirical).build()
    block_fim = block_fim_computer.estimate_hessian(damping=damping)

    # Compute full FIM for comparison
    fim_computer = FIMComputer(compute_context=fim_data_empirical).build()
    full_fim = fim_computer.estimate_hessian(damping=damping)

    assert FullMatrixMetric.RELATIVE_FROBENIUS.compute(full_fim, block_fim) < 1e-3, (
        "FIM block approximation does not match full FIM for linear model (EMPIRICAL_FISHER)."
    )


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_computation_mcmc(
    fim_data_mcmc: DataActivationsGradients,
):
    """Test that FIM block approximation works with MCMC strategy."""
    damping = 1e-5

    # Compute FIM block
    block_fim_computer = FIMBlockComputer(compute_context=fim_data_mcmc).build()
    block_fim = block_fim_computer.estimate_hessian(damping=damping)

    # Compute full FIM for comparison
    fim_computer = FIMComputer(compute_context=fim_data_mcmc).build()
    full_fim = fim_computer.estimate_hessian(damping=damping)

    # Should be close since it's still a linear model
    assert FullMatrixMetric.RELATIVE_FROBENIUS.compute(full_fim, block_fim) < 1e-3, (
        "FIM block approximation does not match full FIM for linear model (MCMC)."
    )


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_computation_all_classes(
    config: ModelConfig,
    fim_data_all_classes: DataActivationsGradients,
):
    """Test that FIM block approximation works with ALL_CLASSES strategy."""
    damping = 1e-5

    # Compute FIM block
    block_fim_computer = FIMBlockComputer(compute_context=fim_data_all_classes).build()
    block_fim = block_fim_computer.estimate_hessian(damping=damping)

    # Compute full FIM for comparison
    fim_computer = FIMComputer(compute_context=fim_data_all_classes).build()
    full_fim = fim_computer.estimate_hessian(damping=damping)

    # Should be close since it's still a linear model
    assert FullMatrixMetric.RELATIVE_FROBENIUS.compute(full_fim, block_fim) < 1e-3, (
        "FIM block approximation does not match full FIM for linear model (ALL_CLASSES)."
    )


@pytest.mark.parametrize("config", ["multi_layer"], indirect=True)
def test_fim_block_computation_multi_layer(
    fim_data_empirical: DataActivationsGradients,
):
    """Test FIM block approximation for multi-layer models."""
    damping = 1e-5

    # Compute FIM block
    block_fim_computer = FIMBlockComputer(compute_context=fim_data_empirical).build()
    block_fim = block_fim_computer.estimate_hessian(damping=damping)

    # Verify it's block diagonal by checking off-diagonal blocks are zero
    layer_names = fim_data_empirical.layer_names
    layer_sizes = []

    for layer_name in layer_names:
        act = fim_data_empirical.activations[layer_name]
        grad = fim_data_empirical.gradients[layer_name]
        layer_sizes.append(act.shape[1] * grad.shape[2])

    # Check that off-diagonal blocks are essentially zero
    start_idx = 0
    for i, size_i in enumerate(layer_sizes):
        for j, size_j in enumerate(layer_sizes):
            if i != j:
                block = block_fim[
                    start_idx : start_idx + size_i,
                    sum(layer_sizes[:j]) : sum(layer_sizes[:j]) + size_j,
                ]
                assert jnp.allclose(block, 0.0, atol=1e-4), (
                    f"Off-diagonal block ({i}, {j}) is not zero"
                )
        start_idx += size_i


# ---------------------------------------------------------------------
# Tests - HVP/IHVP Consistency
# ---------------------------------------------------------------------


@pytest.mark.parametrize("config", ["linear"], indirect=True)
@pytest.mark.parametrize(
    "strategy_fixture",
    ["fim_data_empirical", "fim_data_mcmc", "fim_data_all_classes"],
)
def test_fim_block_hvp_consistency(
    config: ModelConfig,
    strategy_fixture: str,
    request: pytest.FixtureRequest,
):
    """Test HVP consistency between FIMBlockComputer and FIMComputer for all strategies."""
    fim_data = request.getfixturevalue(strategy_fixture)
    damping = 1e-2

    block_fim = FIMBlockComputer(compute_context=fim_data).build()
    full_fim = FIMComputer(compute_context=fim_data).build()

    # Test with random vector
    v_rand = jax.random.normal(
        jax.random.PRNGKey(0),
        (
            fim_data.activations[fim_data.layer_names[0]].shape[1]
            * fim_data.gradients[fim_data.layer_names[0]].shape[2],
        ),
    )

    hvp_block = block_fim.estimate_hvp(v_rand, damping=damping)
    hvp_full = full_fim.estimate_hvp(v_rand, damping=damping)

    strategy_name = fim_data.pseudo_target_strategy.value
    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block, hvp_full) < 1e-3, (
        f"Block FIM HVP does not match full FIM HVP ({strategy_name})"
    )


@pytest.mark.parametrize("config", ["linear"], indirect=True)
@pytest.mark.parametrize(
    "strategy_fixture",
    ["fim_data_empirical", "fim_data_mcmc", "fim_data_all_classes"],
)
def test_fim_block_ihvp_consistency(
    config: ModelConfig,
    strategy_fixture: str,
    request: pytest.FixtureRequest,
):
    """Test IHVP consistency between FIMBlockComputer and FIMComputer for all strategies."""
    fim_data = request.getfixturevalue(strategy_fixture)
    damping = 1e-2

    block_fim = FIMBlockComputer(compute_context=fim_data).build()
    full_fim = FIMComputer(compute_context=fim_data).build()

    # Test with random vector
    v_rand = jax.random.normal(
        jax.random.PRNGKey(0),
        (
            fim_data.activations[fim_data.layer_names[0]].shape[1]
            * fim_data.gradients[fim_data.layer_names[0]].shape[2],
        ),
    )

    ihvp_block = block_fim.estimate_ihvp(v_rand, damping=damping)
    ihvp_full = full_fim.estimate_ihvp(v_rand, damping=damping)

    strategy_name = fim_data.pseudo_target_strategy.value
    assert VectorMetric.RELATIVE_ERROR.compute(ihvp_block, ihvp_full) < 1e-3, (
        f"Block FIM IHVP does not match full FIM IHVP ({strategy_name})"
    )


@pytest.mark.parametrize("config", ["linear"], indirect=True)
@pytest.mark.parametrize(
    "strategy_fixture",
    ["fim_data_empirical", "fim_data_mcmc", "fim_data_all_classes"],
)
def test_fim_block_roundtrip(
    config: ModelConfig,
    strategy_fixture: str,
    request: pytest.FixtureRequest,
):
    """Test round-trip H(H^{-1}v) ≈ v for all strategies."""
    fim_data = request.getfixturevalue(strategy_fixture)
    damping = 1e-2

    block_fim = FIMBlockComputer(compute_context=fim_data).build()

    # Test with random vector
    v_rand = jax.random.normal(
        jax.random.PRNGKey(0),
        (
            fim_data.activations[fim_data.layer_names[0]].shape[1]
            * fim_data.gradients[fim_data.layer_names[0]].shape[2],
        ),
    )

    ihvp = block_fim.estimate_ihvp(v_rand, damping=damping)
    roundtrip = block_fim.estimate_hvp(ihvp, damping=damping)

    strategy_name = fim_data.pseudo_target_strategy.value
    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip, v_rand) < 1e-3, (
        f"Block FIM round-trip H(H^{{-1}}v) failed ({strategy_name})"
    )


# ---------------------------------------------------------------------
# Tests - Batched Operations
# ---------------------------------------------------------------------


@pytest.mark.parametrize("config", ["linear"], indirect=True)
@pytest.mark.parametrize(
    "strategy_fixture",
    ["fim_data_empirical", "fim_data_mcmc", "fim_data_all_classes"],
)
def test_fim_block_hvp_batched(
    config: ModelConfig,
    strategy_fixture: str,
    request: pytest.FixtureRequest,
):
    """Test batched HVP computation for all strategies."""
    fim_data = request.getfixturevalue(strategy_fixture)
    damping = 1e-2

    block_fim = FIMBlockComputer(compute_context=fim_data).build()

    # Create batch of vectors
    n_vectors = 5
    dim = (
        fim_data.activations[fim_data.layer_names[0]].shape[1]
        * fim_data.gradients[fim_data.layer_names[0]].shape[2]
    )
    V = jax.random.normal(jax.random.PRNGKey(0), (n_vectors, dim))

    # Compute batched HVP
    hvp_batched = block_fim.estimate_hvp(V, damping=damping)

    # Compute individual HVPs
    hvp_individual = jnp.stack(
        [block_fim.estimate_hvp(V[i], damping=damping) for i in range(n_vectors)]
    )

    strategy_name = fim_data.pseudo_target_strategy.value
    for i in range(n_vectors):
        assert (
            VectorMetric.RELATIVE_ERROR.compute(hvp_batched[i], hvp_individual[i])
            < 1e-3
        ), f"Batched HVP does not match individual HVP for vector {i} ({strategy_name})"


@pytest.mark.parametrize("config", ["linear"], indirect=True)
@pytest.mark.parametrize(
    "strategy_fixture",
    ["fim_data_empirical", "fim_data_mcmc", "fim_data_all_classes"],
)
def test_fim_block_ihvp_batched(
    config: ModelConfig,
    strategy_fixture: str,
    request: pytest.FixtureRequest,
):
    """Test batched IHVP computation for all strategies."""
    fim_data = request.getfixturevalue(strategy_fixture)
    damping = 1e-2

    block_fim = FIMBlockComputer(compute_context=fim_data).build()

    # Create batch of vectors
    n_vectors = 5
    dim = (
        fim_data.activations[fim_data.layer_names[0]].shape[1]
        * fim_data.gradients[fim_data.layer_names[0]].shape[2]
    )
    V = jax.random.normal(jax.random.PRNGKey(0), (n_vectors, dim))

    # Compute batched IHVP
    ihvp_batched = block_fim.estimate_ihvp(V, damping=damping)

    # Compute individual IHVPs
    ihvp_individual = jnp.stack(
        [block_fim.estimate_ihvp(V[i], damping=damping) for i in range(n_vectors)]
    )

    strategy_name = fim_data.pseudo_target_strategy.value
    for i in range(n_vectors):
        assert (
            VectorMetric.RELATIVE_ERROR.compute(ihvp_batched[i], ihvp_individual[i])
            < 1e-3
        ), (
            f"Batched IHVP does not match individual IHVP for vector {i} ({strategy_name})"
        )


# ---------------------------------------------------------------------
# Tests - Strategy Comparison
# ---------------------------------------------------------------------


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_all_strategies_finite(
    config: ModelConfig,
    fim_data_empirical: DataActivationsGradients,
    fim_data_mcmc: DataActivationsGradients,
    fim_data_all_classes: DataActivationsGradients,
):
    """Test that all strategies produce finite results."""
    damping = 1e-2

    for name, fim_data in [
        ("EMPIRICAL_FISHER", fim_data_empirical),
        ("MCMC", fim_data_mcmc),
        ("ALL_CLASSES", fim_data_all_classes),
    ]:
        block_fim = FIMBlockComputer(compute_context=fim_data).build()
        H = block_fim.estimate_hessian(damping=damping)

        assert jnp.isfinite(H).all(), f"FIM contains non-finite values ({name})"
        assert H.shape[0] == H.shape[1], f"FIM is not square ({name})"


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_pseudo_inverse(
    config: ModelConfig,
    fim_data_empirical: DataActivationsGradients,
):
    """Test pseudo-inverse functionality."""
    pseudo_inverse_factor = 1e-5

    block_fim = FIMBlockComputer(compute_context=fim_data_empirical).build()

    dim = (
        fim_data_empirical.activations[fim_data_empirical.layer_names[0]].shape[1]
        * fim_data_empirical.gradients[fim_data_empirical.layer_names[0]].shape[2]
    )
    v = jax.random.normal(jax.random.PRNGKey(0), (dim,))

    # Compute IHVP with pseudo-inverse
    ihvp = block_fim.estimate_ihvp(v, pseudo_inverse_factor=pseudo_inverse_factor)

    assert jnp.isfinite(ihvp).all(), (
        "IHVP with pseudo-inverse contains non-finite values"
    )

    # Round-trip should still work reasonably well
    roundtrip = block_fim.estimate_hvp(ihvp)

    # Relaxed tolerance due to pseudo-inverse
    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip, v) < 1e-3, (
        "Round-trip with pseudo-inverse failed"
    )
