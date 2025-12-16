from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from src.config import Config, HessianApproximationConfig, ModelConfig
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.utils.data import ModelContext
from src.utils.data.data import Dataset, RandomClassificationDataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.loss import cross_entropy_loss
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
from src.utils.models.linear_model import LinearModel
from src.utils.models.mlp import MLP
from src.utils.optimizers import optimizer
from src.utils.train import train_model

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(params=["linear", "multi_layer"], scope="session")
def config(request):
    hidden_dim = [] if request.param == "linear" else [10]
    return Config(
        dataset_path="random",
        seed=123,
        model=ModelConfig(
            model_name="linear" if request.param == "linear" else "multi_layer",
            metadata={"hidden_dim": hidden_dim},
        ),
        hessian_approximation=HessianApproximationConfig(
            method="BlockHessian",
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
def model_and_params(
    config: Config, dataset: Dataset
) -> Tuple[ApproximationModel, Dict]:
    config.model.metadata = config.model.metadata or {}
    hidden_dim = config.model.metadata["hidden_dim"]
    assert hidden_dim is not None, "hidden_dim must be specified in config metadata"

    if config.model.model_name == "linear":
        model = LinearModel(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=hidden_dim,
            seed=config.seed,
        )
    else:
        model = MLP(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=hidden_dim,
            seed=config.seed,
        )

    model, params, _ = train_model(
        model,
        dataset.get_dataloader(
            batch_size=JAXDataLoader.get_batch_size(),
            seed=config.seed,
        ),
        loss_fn=cross_entropy_loss,
        optimizer=optimizer("sgd", lr=1e-3),
        epochs=30,
    )

    return model, params


@pytest.fixture(scope="session")
def model_context(
    dataset: Dataset,
    model_and_params: Tuple[ApproximationModel, Dict],
) -> ModelContext:
    model, params = model_and_params
    return ModelContext.create(
        model=model,
        dataset=dataset,
        params=params,
        loss_fn=cross_entropy_loss,
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

    block_hessian = BlockHessianComputer(model_context)
    full_hessian = HessianComputer(model_context)

    H_block = block_hessian.estimate_hessian(damping=damping)
    H_full = full_hessian.compute_hessian(damping=damping)

    assert jnp.allclose(H_block, H_full, atol=1e-5), (
        "Block-diagonal Hessian does not match full Hessian for linear model."
    )


@pytest.mark.parametrize("config", ["multi_layer"], indirect=True)
def test_block_hessian_computation_multi_layer(
    model_context: ModelContext,
):
    damping = 1e-5

    # Precompute
    block_hessian = BlockHessianComputer(model_context)
    full_hessian = HessianComputer(model_context)

    H_block = block_hessian.estimate_hessian(damping=damping)
    H_full = full_hessian.compute_hessian(damping=damping)

    for block in block_hessian.blocks:
        start, end = block
        # extract block corresponding to layer
        H_full_block = H_full[start:end, start:end]
        H_block_block = H_block[start:end, start:end]

        assert jnp.allclose(H_block_block, H_full_block, atol=1e-5), (
            "Block-diagonal Hessian block does not match full Hessian block for multi-layer model."
        )


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_block_hessian_hvp_ihvp_roundtrip_linear(
    config: Config,
    model_context: ModelContext,
):
    """
    HVP / IHVP consistency and round-trip checks between
    BlockHessianComputer and true HessianComputer.
    """
    if config.model.metadata["hidden_dim"] != []:  # type: ignore
        pytest.skip("Only applicable for linear model")

    damping = 1e-2

    block_hessian = BlockHessianComputer(model_context)
    full_hessian = HessianComputer(model_context)

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
    # Round-trip sanity: H(H^{-1} v) ≈ v
    # ------------------------------------------------------------------
    roundtrip_block = block_hessian.estimate_hvp(ihvp_block_r, damping=damping)
    roundtrip_full = full_hessian.compute_hvp(ihvp_full_r, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_block, v_rand) < 1e-4, (
        "Block Hessian round-trip H(H^{-1}v) failed"
    )

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_full, v_rand) < 1e-4, (
        "Full Hessian round-trip H(H^{-1}v) failed"
    )
