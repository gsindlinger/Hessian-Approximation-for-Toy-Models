import jax
import jax.numpy as jnp
import pytest

from src.config import ModelConfig
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.utils.data import ModelContext
from src.utils.metrics.vector_metrics import VectorMetric

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("block_test_config", ["linear"], indirect=True)
def test_block_hessian_computation(
    block_test_config: ModelConfig,
    block_test_model_context: ModelContext,
):
    """
    Block-diagonal Hessian must match the full Hessian
    for a linear model.
    """
    damping = 1e-6

    block_hessian = BlockHessianComputer(
        compute_context=block_test_model_context
    ).build()
    full_hessian = HessianComputer(
        compute_context=block_test_model_context
    ).build()

    H_block = block_hessian.estimate_hessian(damping=damping)
    H_full = full_hessian.estimate_hessian(damping=damping)

    assert jnp.allclose(H_block, H_full, atol=1e-5), (
        "Block-diagonal Hessian does not match full Hessian for linear model."
    )


@pytest.mark.parametrize("block_test_config", ["multi_layer"], indirect=True)
def test_block_hessian_computation_multi_layer(
    block_test_config: ModelConfig,
    block_test_model_context: ModelContext,
):
    """
    Test that block-diagonal Hessian matches full Hessian block-by-block
    for multi-layer models.
    """
    damping = 1e-5

    # Compute Hessians
    block_hessian = BlockHessianComputer(
        compute_context=block_test_model_context
    ).build()
    full_hessian = HessianComputer(
        compute_context=block_test_model_context
    ).build()

    H_block = block_hessian.estimate_hessian(damping=damping)
    H_full = full_hessian.estimate_hessian(damping=damping)

    # Check each per-layer block matches
    layer_matrix = block_hessian.layer_matrix
    assert layer_matrix is not None
    offset = 0
    for layer in layer_matrix.param_groups:
        I, O = layer_matrix.layer_shapes[layer]
        start, end = offset, offset + I * O
        offset = end
        H_full_block = H_full[start:end, start:end]
        H_block_block = H_block[start:end, start:end]

        assert jnp.allclose(H_block_block, H_full_block, atol=1e-4), (
            f"Block-diagonal Hessian block [{start}:{end}] does not match "
            f"full Hessian block for multi-layer model."
        )


@pytest.mark.parametrize("block_test_config", ["linear"], indirect=True)
def test_block_hessian_hvp_ihvp_roundtrip_linear(
    block_test_config: ModelConfig,
    block_test_model_context: ModelContext,
):
    """
    Check HVP / IHVP consistency and round trips between
    BlockHessianComputer and full HessianComputer for linear models.
    """
    damping = 1e-2

    # ------------------------------------------------------------------
    # Setup computers
    # ------------------------------------------------------------------
    block_hessian = BlockHessianComputer(
        compute_context=block_test_model_context
    ).build()
    full_hessian = HessianComputer(
        compute_context=block_test_model_context
    ).build()

    params_flat = block_test_model_context.params_flat
    v_ones = jnp.ones_like(params_flat)
    v_rand = jax.random.normal(jax.random.PRNGKey(0), params_flat.shape)

    # ------------------------------------------------------------------
    # HVP consistency
    # ------------------------------------------------------------------
    hvp_block = block_hessian.estimate_hvp(v_ones, damping=damping)
    hvp_full = full_hessian.estimate_hvp(v_ones, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block, hvp_full) < 1e-3, (
        "Block Hessian HVP does not match full Hessian HVP (ones vector)"
    )

    hvp_block_r = block_hessian.estimate_hvp(v_rand, damping=damping)
    hvp_full_r = full_hessian.estimate_hvp(v_rand, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(hvp_block_r, hvp_full_r) < 1e-3, (
        "Block Hessian HVP does not match full Hessian HVP (random vector)"
    )

    # ------------------------------------------------------------------
    # IHVP consistency
    # ------------------------------------------------------------------
    ihvp_block = block_hessian.estimate_ihvp(v_ones, damping=damping)
    ihvp_full = full_hessian.estimate_ihvp(v_ones, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(ihvp_block, ihvp_full) < 1e-3, (
        "Block Hessian IHVP does not match full Hessian IHVP (ones vector)"
    )

    ihvp_block_r = block_hessian.estimate_ihvp(v_rand, damping=damping)
    ihvp_full_r = full_hessian.estimate_ihvp(v_rand, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(ihvp_block_r, ihvp_full_r) < 1e-3, (
        "Block Hessian IHVP does not match full Hessian IHVP (random vector)"
    )

    # ------------------------------------------------------------------
    # Round-trip sanity check:  H(H^{-1} v) ≈ v
    # ------------------------------------------------------------------
    roundtrip_block = block_hessian.estimate_hvp(ihvp_block_r, damping=damping)
    roundtrip_full = full_hessian.estimate_hvp(ihvp_full_r, damping=damping)

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_block, v_rand) < 1e-4, (
        "Block Hessian round-trip H(H^{-1}v) failed"
    )

    assert VectorMetric.RELATIVE_ERROR.compute(roundtrip_full, v_rand) < 1e-4, (
        "Full Hessian round-trip H(H^{-1}v) failed"
    )
