"""Focused architecture/cache checks; these tests do not train a model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from src.config import ActivationFunction, ModelArchitecture, ModelConfig
from src.utils.models.registry import ModelRegistry
from src.utils.models.transformer import TinyTransformer


def _small_model(activation=ActivationFunction.RELU):
    model = TinyTransformer(
        input_dim=4, output_dim=11, transformer_dim=8,
        transformer_num_layers=2, transformer_num_heads=4,
        transformer_vocab_size=11, transformer_max_length=5,
        activation=activation,
    )
    x = jnp.array([[1, 10, 1, 9], [4, 10, 6, 9]], dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(17), x)
    params = jax.tree.map(lambda a: a.astype(jnp.float64), params)
    return model, params, x


def _loss(logits, labels):
    return -jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels].sum()


def test_parameter_count_registry_and_all_leaf_layout():
    config = ModelConfig(architecture=ModelArchitecture.TRANSFORMER, input_dim=4, output_dim=99)
    model = ModelRegistry.get_model(config)
    params = model.init(jax.random.PRNGKey(0), jnp.array([[1, 98, 2, 97]]))
    flat, _ = ravel_pytree(params)
    assert model.num_params == flat.size == 9840
    leaves = jax.tree_util.tree_flatten_with_path(params["params"])[0]
    names = ["/".join(key.key for key in path) for path, _ in leaves]
    assert model.get_layer_names() == names
    assert len(names) == 29
    shapes = model.parameter_block_shapes()
    assert sum(i * o for i, o in shapes.values()) == flat.size
    assert shapes["token_embeddings/kernel"] == (99, 16)
    assert shapes["position_embeddings/kernel"] == (5, 16)
    assert shapes["block_0_ln_1/scale"] == (1, 16)
    assert shapes["block_1_attn_qkv/bias"] == (1, 48)
    assert ModelRegistry.get_model(ModelConfig.from_dict(config.serialize())).num_params == 9840


@pytest.mark.parametrize("activation", [ActivationFunction.RELU, ActivationFunction.GELU])
def test_cache_reconstructs_every_per_example_gradient(activation):
    model, params, x = _small_model(activation)
    labels = jnp.array([2, 3])

    class Recorder:
        def __init__(self):
            self.data = {}

        def backward_collector_fn(self, name, a, g):
            assert name not in self.data
            self.data[name] = (np.asarray(a), np.asarray(g))

    recorder = Recorder()

    def cached_loss(p):
        return _loss(model.apply(p, x, recorder, method=model.collector_apply), labels)

    actual_value, actual_grad = jax.value_and_grad(cached_loss)(params)
    expected_value, expected_grad = jax.value_and_grad(lambda p: _loss(model.apply(p, x), labels))(params)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ravel_pytree(actual_grad)[0], ravel_pytree(expected_grad)[0], rtol=1e-10, atol=1e-10)

    per_example = jax.vmap(jax.grad(lambda p, a, y: -jax.nn.log_softmax(model.apply(p, a))[y]), in_axes=(None, 0, 0))(
        params, x, labels
    )
    assert set(recorder.data) == set(model.get_layer_names())
    for path, leaf in jax.tree_util.tree_flatten_with_path(per_example["params"])[0]:
        name = "/".join(key.key for key in path)
        a, g = recorder.data[name]
        assert a.ndim == g.ndim == 3
        reconstructed = np.einsum("bti,bto->bio", a, g).reshape(leaf.shape)
        np.testing.assert_allclose(reconstructed, leaf, rtol=1e-9, atol=1e-10, err_msg=name)

    # Position 4 is a genuine trainable parameter, but unused by 4-token data.
    assert np.all(np.asarray(per_example["params"]["position_embeddings"]["kernel"][:, 4, :]) == 0)


def test_causal_mask_and_single_example_api():
    model, params, x = _small_model()
    sequence = model.apply(params, x, method=model.sequence_logits)
    assert sequence.shape == (2, 4, 11)
    changed = x.at[:, 2:].set(jnp.array([[2, 3], [7, 8]], dtype=x.dtype))
    changed_sequence = model.apply(params, changed, method=model.sequence_logits)
    np.testing.assert_allclose(sequence[:, :2], changed_sequence[:, :2], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(model.apply(params, x), sequence[:, -1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(model.apply(params, x[0]), sequence[0, -1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(jax.vmap(lambda a: model.apply(params, a))(x), sequence[:, -1], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("activation", [ActivationFunction.RELU, ActivationFunction.GELU])
def test_matches_author_pytorch_architecture_forward_and_backward(activation):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    nn = torch.nn
    model, params, x = _small_model(activation)
    d = model.transformer_dim
    heads = model.transformer_num_heads

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(d)
            self.ln_2 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.up = nn.Linear(d, d * heads)
            self.down = nn.Linear(d * heads, d)

        def forward(self, h):
            h = self.ln_1(h)
            mask = torch.ones(h.shape[1], h.shape[1], dtype=torch.bool).triu(1)
            attention, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            h = h + attention
            m = self.up(self.ln_2(h))
            m = torch.relu(m) if activation == ActivationFunction.RELU else torch.nn.functional.gelu(m)
            return h + self.down(m)

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embeddings = nn.Embedding(11, d)
            self.position_embeddings = nn.Embedding(5, d)
            self.layers = nn.ModuleList([Block() for _ in range(2)])
            self.ln_f = nn.LayerNorm(d)
            self.head = nn.Linear(d, 11, bias=False)

        def forward(self, tokens):
            h = self.token_embeddings(tokens) + self.position_embeddings(torch.arange(tokens.shape[-1]))
            for block in self.layers:
                h = block(h)
            return self.head(self.ln_f(h))[:, -1, :]

    reference = Reference().double()
    mapping = {}
    for name in ("token_embeddings", "position_embeddings"):
        mapping[f"{name}/kernel"] = (getattr(reference, name).weight, False)
    for i, block in enumerate(reference.layers):
        for name in ("ln_1", "ln_2"):
            mapping[f"block_{i}_{name}/scale"] = (getattr(block, name).weight, False)
            mapping[f"block_{i}_{name}/bias"] = (getattr(block, name).bias, False)
        mapping[f"block_{i}_attn_qkv/kernel"] = (block.attn.in_proj_weight, True)
        mapping[f"block_{i}_attn_qkv/bias"] = (block.attn.in_proj_bias, False)
        mapping[f"block_{i}_attn_out/kernel"] = (block.attn.out_proj.weight, True)
        mapping[f"block_{i}_attn_out/bias"] = (block.attn.out_proj.bias, False)
        for name in ("up", "down"):
            mapping[f"block_{i}_mlp_{name}/kernel"] = (getattr(block, name).weight, True)
            mapping[f"block_{i}_mlp_{name}/bias"] = (getattr(block, name).bias, False)
    mapping["ln_f/scale"] = (reference.ln_f.weight, False)
    mapping["ln_f/bias"] = (reference.ln_f.bias, False)
    mapping["head/kernel"] = (reference.head.weight, True)
    assert set(mapping) == set(model.get_layer_names())

    with torch.no_grad():
        for path, leaf in jax.tree_util.tree_flatten_with_path(params["params"])[0]:
            name = "/".join(key.key for key in path)
            target, transpose = mapping[name]
            array = np.asarray(leaf)
            target.copy_(torch.tensor(array.T.copy() if transpose else array.copy()))

    torch_logits = reference(torch.tensor(np.asarray(x), dtype=torch.int64))
    np.testing.assert_allclose(model.apply(params, x), torch_logits.detach().numpy(), rtol=1e-10, atol=1e-10)
    torch.nn.functional.cross_entropy(torch_logits, torch.tensor([2, 3]), reduction="sum").backward()
    grads = jax.grad(lambda p: _loss(model.apply(p, x), jnp.array([2, 3])))(params)
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads["params"])[0]:
        name = "/".join(key.key for key in path)
        target, transpose = mapping[name]
        expected = target.grad.numpy().T if transpose else target.grad.numpy()
        np.testing.assert_allclose(leaf, expected, rtol=1e-8, atol=1e-10, err_msg=name)


def test_invalid_head_configuration_fails_early():
    with pytest.raises(ValueError, match="divisible"):
        TinyTransformer(input_dim=4, output_dim=99, transformer_dim=10, transformer_num_heads=4)
