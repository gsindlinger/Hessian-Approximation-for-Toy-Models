"""Analysis precision is whole-tree, opt-in, and cache-isolated."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from experiments.analyze_hessians import _apply_analysis_precision, _precision_cache_dir
from experiments.provenance import config_hash
from src.config import HessianAnalysisConfig, HessianComputationConfig
from src.utils.data.data import Dataset


def test_analysis_precision_promotes_every_leaf_before_unravel():
    params = {"params": {
        "embedding": jnp.ones((5, 4), dtype=jnp.float32),
        "norm": {"scale": jnp.ones(4, dtype=jnp.float64), "bias": jnp.zeros(4, dtype=jnp.float32)},
    }}
    train = Dataset(jnp.array([[1, 2]], dtype=jnp.int32), jnp.array([3], dtype=jnp.int32), jnp.array([17]))
    test = Dataset(jnp.ones((2, 4), dtype=jnp.float32), jnp.ones((2, 1), dtype=jnp.float32))
    promoted, train64, test64 = _apply_analysis_precision(params, train, test, "float64")
    flat, unravel = ravel_pytree(promoted)
    assert flat.dtype == jnp.float64
    assert all(x.dtype == jnp.float64 for x in jax.tree.leaves(unravel(flat)))
    assert params["params"]["embedding"].dtype == jnp.float32
    assert train64.inputs.dtype == jnp.int32
    assert train64.targets.dtype == jnp.int32
    np.testing.assert_array_equal(train64.row_indices, train.row_indices)
    assert test64.inputs.dtype == jnp.float64
    assert test64.targets.dtype == jnp.float64


def test_analysis_precision_none_preserves_legacy_objects_and_paths():
    params = {"x": jnp.ones(3, dtype=jnp.float32)}
    data = Dataset(jnp.ones((1, 3)), jnp.ones((1,)))
    p, train, test = _apply_analysis_precision(params, data, data, None)
    assert p is params and train is data and test is data
    path = "/tmp/collector_mcmc_r1_s42"
    assert _precision_cache_dir(path, None) == path
    assert _precision_cache_dir(path, "float64") == path + "_dtypefloat64"
    assert _precision_cache_dir(path, "float32") != _precision_cache_dir(path, "float64")


def test_analysis_precision_changes_result_identity():
    dataset = {"name": "modular_addition", "split_id": "unit_test"}
    hashes = {
        config_hash(dataset, HessianAnalysisConfig(
            computation_config=HessianComputationConfig(parameter_dtype=dtype)
        ), 42, "test_model", None)
        for dtype in (None, "float32", "float64")
    }
    assert len(hashes) == 3


def test_invalid_analysis_precision_is_rejected():
    with pytest.raises(ValueError, match="parameter_dtype"):
        HessianComputationConfig(parameter_dtype="float16")
