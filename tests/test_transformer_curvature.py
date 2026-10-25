"""Tiny, untrained full-parameter transformer checks against direct autodiff."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from src.config import PseudoTargetGenerationStrategy
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.emac import EMacComputer
from src.hessians.computer.eidentity import EIdentityComputer
from src.hessians.computer.eshampoo import EShampooComputer
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.utils.data import ModelContext, layer_shapes_from_model_context, parameter_gradients_from_factors
from src.utils.data.data import Dataset
from src.utils.loss import cross_entropy_loss
from src.utils.models.transformer import TinyTransformer


@pytest.fixture(scope="module")
def tiny_context():
    model = TinyTransformer(
        input_dim=3, output_dim=5, transformer_dim=4,
        transformer_num_layers=1, transformer_num_heads=2,
        transformer_vocab_size=5, transformer_max_length=4,
    )
    inputs = jnp.array([[0, 1, 2], [1, 1, 2]], dtype=jnp.int32)
    dataset = Dataset(inputs, jnp.array([2, 3]))
    params = model.init(jax.random.PRNGKey(8), inputs)
    params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
    context = ModelContext.create(model, params, dataset, cross_entropy_loss)
    return model, params, dataset, context


@pytest.fixture(scope="module")
def all_classes_data(tiny_context, tmp_path_factory):
    model, params, dataset, ctx = tiny_context
    directory = tmp_path_factory.mktemp("transformer_cache")
    collector = CollectorActivationsGradients(
        model, params, cross_entropy_loss,
        pseudo_target_strategy=PseudoTargetGenerationStrategy.ALL_CLASSES,
        pseudo_target_repetitions=-1,
    )
    data = collector.collect(dataset, batch_size=3, save_directory=str(directory))
    return data, directory


def test_transformer_flat_layout_covers_embeddings_norms_biases(tiny_context):
    model, params, dataset, ctx = tiny_context
    shapes = layer_shapes_from_model_context(ctx)
    assert shapes == model.parameter_block_shapes()
    assert sum(i * o for i, o in shapes.values()) == ctx.params_flat.size
    expected_order = [
        "/".join(key.key for key in path)
        for path, _ in jax.tree_util.tree_flatten_with_path(params["params"])[0]
    ]
    assert list(shapes) == expected_order
    assert "position_embeddings/kernel" in shapes
    assert "token_embeddings/kernel" in shapes
    assert "block_0_ln_1/scale" in shapes
    assert "block_0_attn_qkv/bias" in shapes


def test_transformer_collected_gradients_equal_all_parameter_autodiff(tiny_context, all_classes_data):
    model, params, dataset, ctx = tiny_context
    data, _ = all_classes_data
    reconstructed = jnp.concatenate([
        parameter_gradients_from_factors(data.activations[name], data.gradients[name]).reshape(2, -1, 5)
        for name in data.layer_names
    ], axis=1)

    def one_gradient(x, target):
        return jax.grad(lambda p: cross_entropy_loss(model.apply(ctx.unravel_fn(p), x), target))(ctx.params_flat)

    reference = jax.vmap(lambda x: jax.vmap(lambda c: one_gradient(x, c))(jnp.arange(5)))(dataset.inputs)
    reference = reference.transpose(0, 2, 1)
    np.testing.assert_allclose(reconstructed, reference, rtol=2e-10, atol=2e-10)


def test_transformer_collector_cache_round_trip(all_classes_data):
    data, directory = all_classes_data
    loaded = CollectorActivationsGradients.load(str(directory))
    assert loaded.layer_names == data.layer_names
    for name in data.layer_names:
        np.testing.assert_array_equal(loaded.activations[name], data.activations[name])
        np.testing.assert_array_equal(loaded.gradients[name], data.gradients[name])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["cache_format_version"] == 2
    assert "expand" in manifest["shared_parameter_convention"]["kfac"]


def test_transformer_full_hessian_is_all_parameter_autodiff(tiny_context):
    model, params, dataset, ctx = tiny_context
    reference = jax.hessian(
        lambda p: cross_entropy_loss(model.apply(ctx.unravel_fn(p), dataset.inputs), dataset.targets)
    )(ctx.params_flat)
    computer = HessianComputer(ctx, derivative_batch_size=47).build()
    actual = computer.estimate_hessian()
    assert actual.shape == (ctx.params_flat.size, ctx.params_flat.size)
    np.testing.assert_allclose(actual, reference, rtol=2e-9, atol=2e-9)
    # Explicitly retain nonzero cross-module interactions, not just blocks.
    first_size = np.prod(next(iter(computer.layer_matrix.layer_shapes.values())))
    assert np.linalg.norm(np.asarray(actual[:first_size, first_size:])) > 1e-6


def test_transformer_full_gnh_equals_all_classes_fisher(tiny_context, all_classes_data):
    model, params, dataset, ctx = tiny_context
    data, _ = all_classes_data
    actual_gnh = GNHComputer(ctx).build().estimate_hessian()
    actual_fim = FIMComputer(data, batch_size=1).build().estimate_hessian()
    assert actual_gnh.shape == (ctx.params_flat.size, ctx.params_flat.size)
    np.testing.assert_allclose(actual_gnh, actual_fim, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("computer", [EKFACComputer, EMacComputer, EShampooComputer, EIdentityComputer])
@pytest.mark.parametrize("corrected", [False, True])
def test_transformer_all_default_kronecker_variants(tiny_context, all_classes_data, computer, corrected):
    _, _, _, ctx = tiny_context
    data, _ = all_classes_data
    estimate = computer(data, corr_context=data, apply_eigenvalue_correction=corrected, batch_size=1).build()
    matrix = estimate.estimate_hessian()
    assert matrix.shape == (ctx.params_flat.size, ctx.params_flat.size)
    assert np.isfinite(matrix).all()
    assert estimate.layer_matrix.layer_shapes == layer_shapes_from_model_context(ctx)


def test_transformer_block_estimators_keep_every_parameter(tiny_context, all_classes_data):
    _, _, _, ctx = tiny_context
    data, _ = all_classes_data
    for computer in (BlockHessianComputer(ctx), FIMBlockComputer(data, batch_size=1)):
        computer.build()
        matrix = computer.estimate_hessian()
        assert matrix.shape == (ctx.params_flat.size, ctx.params_flat.size)
        assert np.isfinite(matrix).all()
        assert computer.layer_matrix.layer_shapes == layer_shapes_from_model_context(ctx)
