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
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.utils.data import DataActivationsGradients, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_gradients
from src.utils.data.data import Dataset, RandomClassificationDataset
from src.utils.loss import cross_entropy_loss, get_loss
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

    # Set architecture and hidden dimensions based on parameter
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
    """Train a model and return it with its parameters."""
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
    """Create a ModelContext with pseudo targets."""
    model, params, loss = model_params_loss

    pseudo_targets = generate_pseudo_targets(
        model=model,
        params=params,
        inputs=dataset.inputs,
        loss_fn=loss,
        rng_key=jax.random.PRNGKey(seed=0),
    )
    return ModelContext.create(
        dataset=dataset.replace_targets(pseudo_targets),
        model=model,
        params=params,
        loss_fn=loss,
    )


def _collect_fim_data(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    model_context: ModelContext,
    suffix: str = "",
) -> DataActivationsGradients:
    """Helper function to collect FIM data with optional directory suffix."""
    base = config.directory
    assert base is not None, "Model directory must be set in config"

    collector_dir = f"{base}/collector{suffix}"

    def load_data():
        return CollectorActivationsGradients.load(collector_dir)

    collector_data: DataActivationsGradients | None = None

    try:
        collector_data = load_data()
    except (ValueError, FileNotFoundError):
        pass

    if collector_data is None:
        model, params, loss = model_params_loss
        collector = CollectorActivationsGradients(
            model=model,
            params=params,
            loss_fn=loss,
        )

        assert model_context.targets is not None, (
            "ModelContext targets must not be None"
        )

        collector_data = collector.collect(
            inputs=model_context.inputs,
            targets=model_context.targets,
            save_directory=collector_dir,
            try_load=True,
        )

        assert collector_data is not None, "Failed to collect FIM data."

    return collector_data


@pytest.fixture(scope="session")
def fim_data(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    model_context: ModelContext,
) -> DataActivationsGradients:
    """Collect FIM data for model (used by linear tests)."""
    return _collect_fim_data(config, model_params_loss, model_context)


@pytest.fixture(scope="session")
def fim_data_multi_layer(
    config: ModelConfig,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    model_context: ModelContext,
) -> DataActivationsGradients:
    """Collect FIM data for multi-layer model."""
    return _collect_fim_data(
        config, model_params_loss, model_context, suffix="_multi_layer"
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_computation(
    fim_data: DataActivationsGradients,
):
    """Test that FIM block approximation matches full FIM for linear models."""
    damping = 1e-5

    # Compute FIM block using FIMBlockComputer
    block_fim_computer = FIMBlockComputer(
        compute_context=(fim_data, DataActivationsGradients())
    ).build()

    block_fim = block_fim_computer.estimate_hessian(damping=damping)

    # Compute full FIM using standard method for comparison
    fim_computer = FIMComputer(
        compute_context=(fim_data, DataActivationsGradients())
    ).build()
    full_fim = fim_computer.estimate_hessian(damping=damping)

    assert FullMatrixMetric.RELATIVE_FROBENIUS.compute(full_fim, block_fim) < 1e-4, (
        "FIM block approximation does not match full FIM for linear model."
    )


@pytest.mark.parametrize("config", ["multi_layer"], indirect=True)
def test_fim_block_computation_multi_layer(
    fim_data_multi_layer: DataActivationsGradients,
):
    """Test that FIM block approximation matches full FIM block-by-block for multi-layer models."""
    damping = 1e-5

    # Compute FIM block using FIMBlockComputer
    block_fim_computer = FIMBlockComputer(
        compute_context=(fim_data_multi_layer, DataActivationsGradients())
    ).build()
    block_fim = block_fim_computer.estimate_hessian(damping=damping)

    # Compute full FIM using standard method for comparison
    fim_computer = FIMComputer(
        compute_context=(fim_data_multi_layer, DataActivationsGradients())
    ).build()
    full_fim = fim_computer.estimate_hessian(damping=damping)

    start_idx = 0
    for layer_name in fim_data_multi_layer.layer_names:
        # Extract block corresponding to layer
        end_idx = next(iter(fim_data_multi_layer.gradients.values())).shape[1]  # type: ignore

        block_fim_block = block_fim[
            start_idx : start_idx + end_idx, start_idx : start_idx + end_idx
        ]
        full_fim_block = full_fim[
            start_idx : start_idx + end_idx, start_idx : start_idx + end_idx
        ]
        start_idx += end_idx

        assert (
            FullMatrixMetric.RELATIVE_FROBENIUS.compute(full_fim_block, block_fim_block)
            < 1e-4
        ), f"FIM block approximation does not match full FIM for layer {layer_name}."


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_hvp_ihvp_roundtrip_linear(
    fim_data: DataActivationsGradients,
    model_params_loss: Tuple[ApproximationModel, Dict],
    dataset: Dataset,
):
    """
    Check HVP / IHVP consistency and round trips between
    FIMBlockComputer and full FIMComputer for linear models.
    """
    damping = 1e-2

    # ------------------------------------------------------------------
    # Setup computers
    # ------------------------------------------------------------------
    block_fim = FIMBlockComputer(
        compute_context=(fim_data, DataActivationsGradients())
    ).build()
    full_fim = FIMComputer(
        compute_context=(fim_data, DataActivationsGradients())
    ).build()

    gradients_concatenated = sample_gradients(
        model=model_params_loss[0],
        params=model_params_loss[1],
        inputs=dataset.inputs,
        targets=dataset.targets,
        loss_fn=cross_entropy_loss,
    )

    v_ones = jnp.ones_like(gradients_concatenated)
    v_rand = jax.random.normal(jax.random.PRNGKey(0), gradients_concatenated.shape)

    # ------------------------------------------------------------------
    # HVP consistency
    # ------------------------------------------------------------------
    hvp_block = block_fim._estimate_hvp(v_ones, damping=damping)
    hvp_full = full_fim._estimate_hvp(v_ones, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block, hvp_full) < 1e-3, (
        "Block FIM HVP does not match full FIM HVP (ones vector)"
    )

    hvp_block_r = block_fim._estimate_hvp(v_rand, damping=damping)
    hvp_full_r = full_fim._estimate_hvp(v_rand, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block_r, hvp_full_r) < 1e-3, (
        "Block FIM HVP does not match full FIM HVP (random vector)"
    )

    # ------------------------------------------------------------------
    # IHVP consistency
    # ------------------------------------------------------------------
    ihvp_block = block_fim._estimate_ihvp(v_ones, damping=damping)
    ihvp_full = full_fim._estimate_ihvp(v_ones, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(ihvp_block, ihvp_full) < 1e-3, (
        "Block FIM IHVP does not match full FIM IHVP (ones vector)"
    )

    ihvp_block_r = block_fim._estimate_ihvp(v_rand, damping=damping)
    ihvp_full_r = full_fim._estimate_ihvp(v_rand, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(ihvp_block_r, ihvp_full_r) < 1e-3, (
        "Block FIM IHVP does not match full FIM IHVP (random vector)"
    )

    # ------------------------------------------------------------------
    # Round-trip sanity check:  H(H^{-1} v) ≈ v
    # ------------------------------------------------------------------
    roundtrip_block = block_fim._estimate_hvp(ihvp_block_r, damping=damping)
    roundtrip_full = full_fim._estimate_hvp(ihvp_full_r, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_block, v_rand) < 1e-3, (
        "Block FIM round-trip H(H^{-1}v) failed"
    )

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_full, v_rand) < 1e-3, (
        "Full FIM round-trip H(H^{-1}v) failed"
    )
