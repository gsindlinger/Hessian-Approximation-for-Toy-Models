"""Hessian-sized KernelSM decoder with complete, token-aware gradient caches.

Architecture: VirtuosoResearch/Kernel-surrogate-models commit
f8f6c1802e2e289f76c8124304eeaab13f3aff16,
src/exp_modular_arithmetic/src/models/transformer.py.

Preserves the author's nonstandard first residual: normalize the stream,
then add attention to that normalized stream. No embeddings, norms, or
biases are frozen. Default configuration has 9,840 trainable parameters.
Cache groups follow individual parameter leaves in JAX flattening order;
sum_t a[t] outer g[t] is the complete per-example parameter derivative.
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.tree_util import tree_flatten_with_path

from src.config import ActivationFunction
from src.utils.models.approximation_model import ApproximationModel


def _linear_uniform(key, shape, dtype=jnp.float32):
    """PyTorch Linear default distribution, in (input, output) order."""
    bound = 1.0 / math.sqrt(shape[0])
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class _LeafCollector:
    def __init__(self, collector, *, kind: str, use_bias: bool = False):
        self.collector = collector
        self.kind = kind
        self.use_bias = use_bias

    def backward_collector_fn(self, name, a, g):
        ones = jnp.ones((*a.shape[:-1], 1), dtype=a.dtype)
        if self.kind == "normalization":
            self.collector.backward_collector_fn(f"{name}/scale", ones, a * g)
            self.collector.backward_collector_fn(f"{name}/bias", ones, g)
        else:
            self.collector.backward_collector_fn(f"{name}/kernel", a, g)
            if self.use_bias:
                self.collector.backward_collector_fn(f"{name}/bias", ones, g)


class _CachedDense(nn.Module):
    features: int
    use_bias: bool = True
    qkv: bool = False
    zero_bias: bool = False

    @nn.compact
    def __call__(self, x, collector=None):
        kernel_init = nn.initializers.xavier_uniform() if self.qkv else _linear_uniform
        params = {"kernel": self.param("kernel", kernel_init, (x.shape[-1], self.features))}
        if self.use_bias:
            if self.zero_bias or self.qkv:
                bias_init = nn.initializers.zeros
            else:
                bound = 1.0 / math.sqrt(x.shape[-1])

                def bias_init(key, shape, dtype=jnp.float32):
                    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

            params["bias"] = self.param("bias", bias_init, (self.features,))

        def affine(p, a):
            result = a @ p["kernel"]
            return result + p["bias"] if self.use_bias else result

        if collector is None:
            return affine(params, x)
        from src.hessians.collector import layer_wrapper_vjp

        return layer_wrapper_vjp(
            affine, params, x, self.name,
            _LeafCollector(collector, kind="dense", use_bias=self.use_bias),
        )


class _CachedEmbedding(nn.Module):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, indices, collector=None):
        kernel = self.param(
            "kernel", nn.initializers.normal(stddev=1.0),
            (self.num_embeddings, self.features),
        )
        if collector is None:
            return jnp.take(kernel, indices, axis=0)
        from src.hessians.collector import layer_wrapper_vjp

        # One-hot inputs describe the exact embedding derivative, including
        # repeated tokens. Ordinary training uses a gather for efficiency.
        inputs = jax.nn.one_hot(indices, self.num_embeddings, dtype=kernel.dtype)
        return layer_wrapper_vjp(
            lambda p, a: a @ p["kernel"], {"kernel": kernel}, inputs,
            self.name, _LeafCollector(collector, kind="embedding"),
        )


class _CachedLayerNorm(nn.Module):
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, collector=None):
        # Normalization stays outside the affine hook, so its full Jacobian
        # propagates upstream through ordinary autodiff.
        centered = x - jnp.mean(x, axis=-1, keepdims=True)
        normalized = centered * jax.lax.rsqrt(
            jnp.mean(jnp.square(centered), axis=-1, keepdims=True) + self.epsilon
        )
        params = {
            "scale": self.param("scale", nn.initializers.ones, (x.shape[-1],)),
            "bias": self.param("bias", nn.initializers.zeros, (x.shape[-1],)),
        }

        def affine(p, a):
            return a * p["scale"] + p["bias"]

        if collector is None:
            return affine(params, normalized)
        from src.hessians.collector import layer_wrapper_vjp

        return layer_wrapper_vjp(
            affine, params, normalized, self.name,
            _LeafCollector(collector, kind="normalization"),
        )


class TinyTransformer(ApproximationModel):
    """Causal decoder returning logits for the final input token.

    Input shape is (..., sequence_length) with one optional batch axis;
    output is (..., output_dim). ``sequence_logits`` exposes all tokens.
    Dropout is absent, as in the source modular-arithmetic configurations.
    """

    hidden_dim: list[int] | None = field(default_factory=list)
    activation: ActivationFunction = ActivationFunction.RELU
    transformer_dim: int = 16
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_vocab_size: int = 99
    transformer_max_length: int = 5

    @property
    def input_dtype(self):
        return jnp.int32

    def __post_init__(self):
        super().__post_init__()
        if self.transformer_dim <= 0 or self.transformer_num_heads <= 0:
            raise ValueError("Transformer width and number of heads must be positive")
        if self.transformer_dim % self.transformer_num_heads:
            raise ValueError("Transformer width must be divisible by the number of heads")
        if self.transformer_num_layers <= 0:
            raise ValueError("Transformer must contain at least one layer")
        if not 0 < self.input_dim <= self.transformer_max_length:
            raise ValueError("input_dim must be a positive sequence length within max_length")
        if self.transformer_vocab_size <= 0 or self.output_dim <= 0:
            raise ValueError("Transformer vocabulary and output dimensions must be positive")
        if self.activation not in (ActivationFunction.RELU, ActivationFunction.GELU):
            raise ValueError("KernelSM transformer supports ReLU or GELU activation")

    @nn.compact
    def __call__(self, x, collector=None, *, return_sequence: bool = False):
        x = jnp.asarray(x)
        if x.ndim not in (1, 2):
            raise ValueError("Transformer inputs need a token axis and optional batch axis")
        unbatched = x.ndim == 1
        if unbatched:
            x = x[None, :]
        length = x.shape[-1]
        if length < 1 or length > self.transformer_max_length:
            raise ValueError("Input length is outside the position embedding range")
        # Shape utilities initialize with floating-point zeros; the modular
        # dataset adapter validates actual token values before training.
        tokens = x.astype(jnp.int32)
        d = self.transformer_dim
        heads = self.transformer_num_heads
        h = _CachedEmbedding(self.transformer_vocab_size, d, name="token_embeddings")(
            tokens, collector
        )
        positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), tokens.shape)
        h = h + _CachedEmbedding(self.transformer_max_length, d, name="position_embeddings")(
            positions, collector
        )
        causal_mask = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]

        for i in range(self.transformer_num_layers):
            h = _CachedLayerNorm(name=f"block_{i}_ln_1")(h, collector)
            qkv = _CachedDense(3 * d, qkv=True, name=f"block_{i}_attn_qkv")(h, collector)
            q, k, v = [
                a.reshape(a.shape[0], length, heads, d // heads).transpose(0, 2, 1, 3)
                for a in jnp.split(qkv, 3, axis=-1)
            ]
            scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(d // heads)
            attention = jax.nn.softmax(jnp.where(causal_mask, scores, -jnp.inf), axis=-1)
            context = jnp.einsum("bhts,bhsd->bhtd", attention, v)
            context = context.transpose(0, 2, 1, 3).reshape(h.shape)
            h = h + _CachedDense(d, zero_bias=True, name=f"block_{i}_attn_out")(
                context, collector
            )
            m = _CachedLayerNorm(name=f"block_{i}_ln_2")(h, collector)
            m = _CachedDense(d * heads, name=f"block_{i}_mlp_up")(m, collector)
            m = jax.nn.relu(m) if self.activation == ActivationFunction.RELU else jax.nn.gelu(m, approximate=False)
            h = h + _CachedDense(d, name=f"block_{i}_mlp_down")(m, collector)

        h = _CachedLayerNorm(name="ln_f")(h, collector)
        # Keep a singleton token axis for consistent sequence-aware caches.
        if not return_sequence:
            h = h[:, -1:, :]
        logits = _CachedDense(self.output_dim, use_bias=False, name="head")(h, collector)
        if not return_sequence:
            logits = logits[:, 0, :]
        return logits[0] if unbatched else logits

    def collector_apply(self, x, collector) -> Any:
        return self(x, collector=collector)

    def sequence_logits(self, x):
        return self(x, return_sequence=True)

    def _parameter_shape_leaves(self):
        variables = jax.eval_shape(
            self.init, jax.random.PRNGKey(0), jnp.zeros((1, self.input_dim), dtype=jnp.int32)
        )
        return tree_flatten_with_path(variables["params"])[0]

    def get_layer_names(self) -> list[str]:
        """Every trainable leaf, in the same order as ravel_pytree."""
        return ["/".join(key.key for key in path) for path, _ in self._parameter_shape_leaves()]

    def parameter_block_shapes(self) -> dict[str, tuple[int, int]]:
        """Native matrix layout; vectors are represented by a single row."""
        result = {}
        for path, leaf in self._parameter_shape_leaves():
            name = "/".join(key.key for key in path)
            result[name] = (1, leaf.shape[0]) if len(leaf.shape) == 1 else leaf.shape
        return result
