from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import pytest

from src.config import Config, HessianApproximationConfig, ModelConfig
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.utils.data import ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets
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
def config(request, tmp_path_factory):
    base = tmp_path_factory.mktemp(request.param)

    hidden_dim = [] if request.param == "linear" else [10]

    return Config(
        dataset_path="random",
        seed=123,
        model=ModelConfig(
            model_name="test",
            directory=str(base / "model"),
            metadata={
                "hidden_dim": hidden_dim,
                "activation": "relu" if request.param == "multi_layer" else None,
            },
        ),
        hessian_approximation=HessianApproximationConfig(
            method="EKFAC",
            directory=str(base / "fim_block"),
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

    model = LinearModel(
        input_dim=dataset.input_dim(),
        output_dim=dataset.output_dim(),
        hidden_dim=hidden_dim,
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
def model_context(
    dataset: Dataset, model_and_params: Tuple[ApproximationModel, Dict]
) -> ModelContext:
    model, params = model_and_params

    pseudo_targets = generate_pseudo_targets(
        model=model,
        params=params,
        inputs=dataset.inputs,
        loss_fn=cross_entropy_loss,
        rng_key=jax.random.PRNGKey(seed=0),
    )
    return ModelContext.create(
        dataset=dataset.replace_targets(pseudo_targets),
        model=model,
        params=params,
        loss_fn=cross_entropy_loss,
    )


@pytest.fixture(scope="session")
def fim_block_data(
    config: Config, model_and_params: Tuple[MLP, Dict], model_context: ModelContext
):
    base = config.hessian_approximation.directory
    assert base is not None, "Hessian approximation directory must be set in config"

    try:
        return CollectorActivationsGradients.load(base)
    except (ValueError, FileNotFoundError):
        pass

    collector = CollectorActivationsGradients(
        model=model_and_params[0], params=model_and_params[1]
    )

    assert model_context.targets is not None, "ModelContext targets must not be None"

    collector.collect(
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=cross_entropy_loss,
        save_directory=base,
    )

    return CollectorActivationsGradients.load(base)


@pytest.fixture(scope="session")
def fim_block_data_multi_layer(
    config: Config, model_and_params: Tuple[MLP, Dict], model_context: ModelContext
):
    base = config.hessian_approximation.directory + "_multi_layer"  # type: ignore
    assert base is not None, "Hessian approximation directory must be set in config"

    try:
        return CollectorActivationsGradients.load(base)
    except (ValueError, FileNotFoundError):
        pass

    collector = CollectorActivationsGradients(
        model=model_and_params[0], params=model_and_params[1]
    )

    assert model_context.targets is not None, "ModelContext targets must not be None"

    collector.collect(
        inputs=model_context.inputs,
        targets=model_context.targets,
        loss_fn=cross_entropy_loss,
        save_directory=base,
    )

    return CollectorActivationsGradients.load(base)


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_computation(
    fim_block_data: Tuple[
        CollectorActivationsGradients, CollectorActivationsGradients, Dict
    ],
    model_context: ModelContext,
):
    damping = 1e-5

    # Compute FIM block using FIMBlockComputer
    block_fim_computer = FIMBlockComputer(compute_context=fim_block_data)
    block_fim = block_fim_computer.estimate_hessian(damping=damping)

    # Compute full FIM using standard method for comparison
    fim_computer = FIMComputer(compute_context=model_context)
    full_fim = fim_computer.estimate_hessian(damping=damping)

    assert jnp.allclose(block_fim, full_fim, atol=1e-5), (
        "FIM block approximation does not match full FIM for linear model."
    )


@pytest.mark.parametrize("config", ["multi_layer"], indirect=True)
def test_fim_block_computation_multi_layer(
    fim_block_data_multi_layer: Tuple[
        CollectorActivationsGradients, CollectorActivationsGradients, Dict
    ],
    model_context: ModelContext,
):
    damping = 1e-5

    # Compute FIM block using FIMBlockComputer
    block_fim_computer = FIMBlockComputer(compute_context=fim_block_data_multi_layer)
    block_fim = block_fim_computer.estimate_hessian(damping=damping)

    # Compute full FIM using standard method for comparison
    fim_computer = FIMComputer(compute_context=model_context)
    full_fim = fim_computer.estimate_hessian(damping=damping)

    start_idx = 0
    for layer_name in fim_block_data_multi_layer[2]:
        # extract block corresponding to layer
        end_idx = next(iter(fim_block_data_multi_layer[0].values())).shape[1]  # type: ignore

        block_fim_block = block_fim[
            start_idx : start_idx + end_idx, start_idx : start_idx + end_idx
        ]
        full_fim_block = full_fim[
            start_idx : start_idx + end_idx, start_idx : start_idx + end_idx
        ]
        start_idx += end_idx

        assert jnp.allclose(block_fim_block, full_fim_block, atol=1e-5), (
            f"FIM block approximation does not match full FIM for layer {layer_name}."
        )


@pytest.mark.parametrize("config", ["linear"], indirect=True)
def test_fim_block_hvp_ihvp_roundtrip_linear(
    fim_block_data: Tuple[
        CollectorActivationsGradients, CollectorActivationsGradients, Dict
    ],
    model_context: ModelContext,
):
    """
    Check HVP / IHVP consistency and round trips between
    FIMBlockComputer and full FIMComputer for linear models.
    """

    damping = 1e-2

    # ------------------------------------------------------------------
    # Setup computers
    # ------------------------------------------------------------------
    block_fim = FIMBlockComputer(compute_context=fim_block_data)
    full_fim = FIMComputer(compute_context=model_context)

    params_flat = model_context.params_flat

    v_ones = jnp.ones_like(params_flat)
    v_rand = jax.random.normal(jax.random.PRNGKey(0), params_flat.shape)

    # ------------------------------------------------------------------
    # HVP consistency
    # ------------------------------------------------------------------
    hvp_block = block_fim.estimate_hvp(v_ones, damping=damping)
    hvp_full = full_fim.estimate_hvp(v_ones, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block, hvp_full) < 1e-4, (
        "Block FIM HVP does not match full FIM HVP (ones vector)"
    )

    hvp_block_r = block_fim.estimate_hvp(v_rand, damping=damping)
    hvp_full_r = full_fim.estimate_hvp(v_rand, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block_r, hvp_full_r) < 1e-4, (
        "Block FIM HVP does not match full FIM HVP (random vector)"
    )

    # ------------------------------------------------------------------
    # IHVP consistency
    # ------------------------------------------------------------------
    ihvp_block = block_fim.estimate_ihvp(v_ones, damping=damping)
    ihvp_full = full_fim.estimate_ihvp(v_ones, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(ihvp_block, ihvp_full) < 1e-4, (
        "Block FIM IHVP does not match full FIM IHVP (ones vector)"
    )

    ihvp_block_r = block_fim.estimate_ihvp(v_rand, damping=damping)
    ihvp_full_r = full_fim.estimate_ihvp(v_rand, damping=damping)

    # ------------------------------------------------------------------
    # Round-trip sanity check: H(H^{-1} v) ≈ v
    # ------------------------------------------------------------------
    roundtrip_block = block_fim.estimate_hvp(ihvp_block_r, damping=damping)
    roundtrip_full = full_fim.estimate_hvp(ihvp_full_r, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_block, v_rand) < 1e-4, (
        "Block FIM round-trip H(H^{-1}v) failed"
    )

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_full, v_rand) < 1e-4, (
        "Full FIM round-trip H(H^{-1}v) failed"
    )
