"""Tiny-batch checks only; no training, dataset loading, or experiment outputs."""
import numpy as np
import jax
import jax.numpy as jnp

from experiments.unrolling.transformer import make_transformer


def test_full_transformer_cache_and_exact_factors():
    theta, forward, shapes, factors = make_transformer(
        {"activation": "relu", "factor_batch_size": 1}, 42, 4, 99,
    )
    assert theta.size == 9840 and len(shapes) == 29
    assert all(leaf.dtype == jnp.float64 for leaf in forward.unravel(theta))
    tokens = jnp.array([[0, 98, 0, 97], [1, 98, 2, 97]], dtype=jnp.int32)
    cached = [factors.example_data(theta, x) for x in tokens]

    @jax.jit
    def exact_root_gradients(x):
        logits, pullback = jax.vjp(lambda p: forward(p, x[None])[0], theta)
        probs = jax.nn.softmax(logits)
        root = (jnp.eye(99) - probs[:, None]) * jnp.sqrt(probs)[None, :]
        return jax.vmap(lambda r: pullback(r)[0], in_axes=1)(root).T

    exact = np.stack([np.asarray(exact_root_gradients(x)) for x in tokens])
    reconstructed = np.stack([
        np.concatenate([np.einsum("ti,tok->iok", np.asarray(a), np.asarray(g)).reshape(-1, 99)
                        for a, g in data]) for data in cached
    ])
    np.testing.assert_allclose(reconstructed, exact, rtol=2e-9, atol=2e-10)

    statistics = factors(theta, tokens)
    offset = 0
    for index, (shape, stats) in enumerate(zip(shapes, statistics)):
        a, g, qa, qg, correction = map(np.asarray, stats)
        incoming = np.stack([np.asarray(data[index][0]) for data in cached])
        cotangents = np.stack([np.asarray(data[index][1]) for data in cached])
        expected_a = np.einsum("nti,ntj->ij", incoming, incoming) / len(tokens)
        expected_g = np.einsum("ntok,ntpk->op", cotangents, cotangents) / (len(tokens) * incoming.shape[1])
        np.testing.assert_allclose(a, expected_a, rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(g, expected_g, rtol=2e-9, atol=2e-10)
        size = np.prod(shape)
        block = exact[:, offset:offset + size].reshape((len(tokens), *shape, 99))
        rotated = np.einsum("ia,niok,ob->nabk", qa, block, qg, optimize=True)
        expected_correction = np.sum(rotated**2, axis=(0, 3)) / len(tokens)
        np.testing.assert_allclose(correction, expected_correction, rtol=2e-8, atol=2e-10)
        offset += size
    assert offset == theta.size


def test_forward_matches_native_architecture_with_identical_parameters():
    # Reference import is confined to this test, never the standalone adapter.
    from src.config import ActivationFunction
    from src.utils.models.transformer import TinyTransformer

    theta, forward, _, _ = make_transformer({"activation": "relu"}, 42, 4, 99)
    native = TinyTransformer(input_dim=4, output_dim=99, activation=ActivationFunction.RELU)
    params = {}
    for name, value in zip(forward.parameter_names, forward.unravel(theta)):
        module, leaf = name.split("/")
        params.setdefault(module, {})[leaf] = value
    tokens = jnp.array([[0, 98, 0, 97], [1, 98, 2, 97]], dtype=jnp.int32)
    reference = jax.jit(lambda p: native.apply({"params": p}, tokens))(params)
    actual = jax.jit(forward)(theta, tokens)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=2e-11, atol=2e-12)
