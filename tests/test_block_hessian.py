from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from src.config import (
    LossType,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
)
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.utils.data import BlockHessianData, ModelContext
from src.utils.data.data import Dataset, RandomClassificationDataset
from src.utils.loss import get_loss
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(params=["linear", "multi_layer"], scope="session")
def config(request, tmp_path_factory):
    """Create model configuration for testing."""
    base = tmp_path_factory.mktemp(request.param)

    # Set architecture and hidden dimensions based on parameter
    if request.param == "linear":
        architecture = ModelArchitecture.LINEAR
        hidden_dim = []
    else:
        architecture = ModelArchitecture.MLP
        hidden_dim = [10]

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

    # For Hessian computation, we don't need pseudo targets
    # ModelContext is created directly with the original dataset
    return ModelContext.create(
        dataset=dataset,
        model=model,
        params=params,
        loss_fn=loss,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_block_hessian_computation(
    model_context: ModelContext,
):
    """
    Block-diagonal Hessian must match the full Hessian
    for a linear model.
    """
    damping = 1e-6

    block_hessian = BlockHessianComputer(compute_context=model_context).build()
    full_hessian = HessianComputer(compute_context=model_context)

    H_block = block_hessian.estimate_hessian(damping=damping)
    H_full = full_hessian.compute_hessian(damping=damping)

    assert jnp.allclose(H_block, H_full, atol=1e-5), (
        "Block-diagonal Hessian does not match full Hessian for linear model."
    )


@pytest.mark.parametrize("config", ["multi_layer"], indirect=True)
def test_block_hessian_computation_multi_layer(
    model_context: ModelContext,
):
    """
    Test that block-diagonal Hessian matches full Hessian block-by-block
    for multi-layer models.
    """
    damping = 1e-5

    # Compute Hessians
    block_hessian = BlockHessianComputer(compute_context=model_context).build()
    full_hessian = HessianComputer(compute_context=model_context)

    H_block = block_hessian.estimate_hessian(damping=damping)
    H_full = full_hessian.compute_hessian(damping=damping)

    # Check each block matches
    assert isinstance(block_hessian.precomputed_data, BlockHessianData), (
        "Precomputed data should be of type BlockHessianData."
    )
    for block in block_hessian.precomputed_data.blocks:
        start, end = block
        # Extract block corresponding to layer
        H_full_block = H_full[start:end, start:end]
        H_block_block = H_block[start:end, start:end]

        assert jnp.allclose(H_block_block, H_full_block, atol=1e-5), (
            f"Block-diagonal Hessian block [{start}:{end}] does not match "
            f"full Hessian block for multi-layer model."
        )


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_block_hessian_hvp_ihvp_roundtrip_linear(
    model_context: ModelContext,
):
    """
    Check HVP / IHVP consistency and round trips between
    BlockHessianComputer and full HessianComputer for linear models.
    """
    damping = 1e-2

    # ------------------------------------------------------------------
    # Setup computers
    # ------------------------------------------------------------------
    block_hessian = BlockHessianComputer(compute_context=model_context).build()
    full_hessian = HessianComputer(compute_context=model_context)

    params_flat = model_context.params_flat
    v_ones = jnp.ones_like(params_flat)
    v_rand = jax.random.normal(jax.random.PRNGKey(0), params_flat.shape)

    # ------------------------------------------------------------------
    # HVP consistency
    # ------------------------------------------------------------------
    hvp_block = block_hessian.estimate_hvp(v_ones, damping=damping)
    hvp_full = full_hessian.compute_hvp(v_ones, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block, hvp_full) < 1e-4, (
        "Block Hessian HVP does not match full Hessian HVP (ones vector)"
    )

    hvp_block_r = block_hessian.estimate_hvp(v_rand, damping=damping)
    hvp_full_r = full_hessian.compute_hvp(v_rand, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block_r, hvp_full_r) < 1e-4, (
        "Block Hessian HVP does not match full Hessian HVP (random vector)"
    )

    # ------------------------------------------------------------------
    # IHVP consistency
    # ------------------------------------------------------------------
    ihvp_block = block_hessian.estimate_ihvp(v_ones, damping=damping)
    ihvp_full = full_hessian.compute_ihvp(v_ones, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(ihvp_block, ihvp_full) < 1e-4, (
        "Block Hessian IHVP does not match full Hessian IHVP (ones vector)"
    )

    ihvp_block_r = block_hessian.estimate_ihvp(v_rand, damping=damping)
    ihvp_full_r = full_hessian.compute_ihvp(v_rand, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(ihvp_block_r, ihvp_full_r) < 1e-4, (
        "Block Hessian IHVP does not match full Hessian IHVP (random vector)"
    )

    # ------------------------------------------------------------------
    # Round-trip sanity check:  H(H^{-1} v) ≈ v
    # ------------------------------------------------------------------
    roundtrip_block = block_hessian.estimate_hvp(ihvp_block_r, damping=damping)
    roundtrip_full = full_hessian.compute_hvp(ihvp_full_r, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_block, v_rand) < 1e-4, (
        "Block Hessian round-trip H(H^{-1}v) failed"
    )

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_full, v_rand) < 1e-4, (
        "Full Hessian round-trip H(H^{-1}v) failed"
    )
