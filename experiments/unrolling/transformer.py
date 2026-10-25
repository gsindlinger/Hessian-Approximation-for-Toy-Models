"""Local Hessian-sized KernelSM transformer and exact-output-curvature factors.

No repository model, collector, cache, or database imports. The first residual
adds attention to the NORMALIZED stream, matching the author/native model.
All 29 parameter leaves are trainable and explicitly initialized in float64.
"""


def make_transformer(config, seed, inputlength, output_dim):
    """Return (flat parameters, forward, block shapes, factor function).

    Factors are in sorted native leaf order, with vectors represented as (1,D).
    ``factors(theta, x)`` streams two passes over small batches: covariance,
    then exact Fisher/GNH diagonals in the Kronecker eigenbasis. Shared-token
    gradients are summed BEFORE squaring for the correction. No label draws.
    """
    import math
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64
    d = int(config.get("transformer_dim", 16))
    layers = int(config.get("transformer_num_layers", 2))
    heads = int(config.get("transformer_num_heads", 4))
    vocab = int(config.get("transformer_vocab_size", 99))
    maxlen = int(config.get("transformer_max_length", 5))
    activation = config.get("activation", "relu")
    loss_name = config.get("loss", "cross_entropy")
    chunk_size = int(config.get("factor_batch_size", 16))
    if (d < 1 or heads < 1 or d % heads or layers < 1 or vocab < 1
            or not 0 < inputlength <= maxlen or output_dim < 1 or chunk_size < 1):
        raise ValueError("Invalid local transformer dimensions or factor batch size.")
    if activation not in ("relu", "gelu") or loss_name not in ("cross_entropy", "mse"):
        raise ValueError("Transformer activation must be relu/gelu; loss must be CE/MSE.")

    # Module metadata also fixes each affine-output intervention's shape.
    modules = []
    specs = {}

    def add(name, kind, shape, *, bias=True, qkv=False, zero_bias=False):
        modules.append((name, kind, shape, 1 if name == "head" else inputlength))
        if kind == "norm":
            specs[f"{name}/scale"] = ((shape[-1],), "ones", 0.0)
            specs[f"{name}/bias"] = ((shape[-1],), "zeros", 0.0)
        else:
            init = "normal" if kind == "embedding" else "uniform"
            bound = math.sqrt(6 / sum(shape)) if qkv else 1 / math.sqrt(shape[0])
            specs[f"{name}/kernel"] = (shape, init, bound)
            if kind == "dense" and bias:
                specs[f"{name}/bias"] = ((shape[-1],), "zeros" if qkv or zero_bias else "uniform", bound)

    add("token_embeddings", "embedding", (vocab, d))
    add("position_embeddings", "embedding", (maxlen, d))
    for i in range(layers):
        add(f"block_{i}_ln_1", "norm", (1, d))
        add(f"block_{i}_attn_qkv", "dense", (d, 3 * d), qkv=True)
        add(f"block_{i}_attn_out", "dense", (d, d), zero_bias=True)
        add(f"block_{i}_ln_2", "norm", (1, d))
        add(f"block_{i}_mlp_up", "dense", (d, heads * d))
        add(f"block_{i}_mlp_down", "dense", (heads * d, d))
    add("ln_f", "norm", (1, d))
    add("head", "dense", (d, output_dim), bias=False)
    names = tuple(sorted(specs))
    keys = jax.random.split(jax.random.PRNGKey(seed), len(names))
    leaves = []
    for key, name in zip(keys, names):
        shape, kind, bound = specs[name]
        if kind == "normal":
            value = jax.random.normal(key, shape, dtype=dtype)
        elif kind == "uniform":
            value = jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)
        else:
            value = jnp.ones(shape, dtype=dtype) if kind == "ones" else jnp.zeros(shape, dtype=dtype)
        leaves.append(value)
    theta_init, unravel = ravel_pytree(tuple(leaves))
    block_shapes = tuple((1, *specs[n][0]) if len(specs[n][0]) == 1 else specs[n][0] for n in names)
    module_index = {name: i for i, (name, _, _, _) in enumerate(modules)}
    zero_shifts = tuple(jnp.zeros((length, shape[-1]), dtype=dtype)
                        for _, _, shape, length in modules)
    causal = jnp.arange(inputlength)[:, None] >= jnp.arange(inputlength)[None, :]

    def single(theta, tokens, shifts=None, cache=False):
        params = dict(zip(names, unravel(theta)))
        incoming = []

        def affine(name, a):
            index = module_index[name]
            kind = modules[index][1]
            if kind == "norm":
                centered = a - a.mean(-1, keepdims=True)
                a = centered * jax.lax.rsqrt(jnp.mean(centered**2, axis=-1, keepdims=True) + 1e-5)
                out = a * params[f"{name}/scale"] + params[f"{name}/bias"]
            elif kind == "embedding":
                token_ids = a
                out = params[f"{name}/kernel"][token_ids]
                a = jax.nn.one_hot(token_ids, modules[index][2][0], dtype=dtype)
            else:
                out = a @ params[f"{name}/kernel"]
                if f"{name}/bias" in params:
                    out = out + params[f"{name}/bias"]
            incoming.append(a)
            return out if shifts is None else out + shifts[index]

        tokens = tokens.astype(jnp.int32)
        h = affine("token_embeddings", tokens)
        h = h + affine("position_embeddings", jnp.arange(inputlength))
        for i in range(layers):
            h = affine(f"block_{i}_ln_1", h)
            qkv = affine(f"block_{i}_attn_qkv", h)
            q, k, v = [a.reshape(inputlength, heads, d // heads).transpose(1, 0, 2)
                       for a in jnp.split(qkv, 3, axis=-1)]
            scores = jnp.einsum("htd,hsd->hts", q, k) / math.sqrt(d // heads)
            attention = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
            context = jnp.einsum("hts,hsd->htd", attention, v).transpose(1, 0, 2).reshape(inputlength, d)
            h = h + affine(f"block_{i}_attn_out", context)
            m = affine(f"block_{i}_ln_2", h)
            m = affine(f"block_{i}_mlp_up", m)
            m = jax.nn.relu(m) if activation == "relu" else jax.nn.gelu(m, approximate=False)
            h = h + affine(f"block_{i}_mlp_down", m)
        h = affine("ln_f", h)
        logits = affine("head", h[-1:, :])[0]
        return (logits, tuple(incoming)) if cache else logits

    def forward(theta, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != inputlength:
            raise ValueError("Transformer inputs require shape (batch, inputlength).")
        return jax.vmap(single, in_axes=(None, 0))(theta, inputs)

    def example_data(theta, tokens):
        logits, pullback, incoming = jax.vjp(
            lambda shifts: single(theta, tokens, shifts, cache=True), zero_shifts,
            has_aux=True,
        )
        if loss_name == "mse":
            root = math.sqrt(2 / output_dim) * jnp.eye(output_dim, dtype=dtype)
        else:
            p = jax.nn.softmax(logits)
            # Columns sqrt(p_k)(e_k-p), whose outer products sum to C.
            root = (jnp.eye(output_dim, dtype=dtype) - p[:, None]) * jnp.sqrt(p)[None, :]
        cotangents = jax.vmap(lambda r: pullback(r)[0], in_axes=1)(root)
        data = {}
        for (name, kind, _, _), a, g in zip(modules, incoming, cotangents):
            g = g.transpose(1, 2, 0)  # (tokens, output width, output-root column)
            ones = jnp.ones((a.shape[0], 1), dtype=dtype)
            if kind == "norm":
                data[f"{name}/scale"] = (ones, a[:, :, None] * g)
                data[f"{name}/bias"] = (ones, g)
            else:
                data[f"{name}/kernel"] = (a, g)
                if f"{name}/bias" in specs:
                    data[f"{name}/bias"] = (ones, g)
        return tuple(data[name] for name in names)

    @jax.jit
    def covariance_sums(theta, bx):
        data = jax.vmap(example_data, in_axes=(None, 0))(theta, bx)
        return tuple((jnp.einsum("nti,ntj->ij", a, a),
                      jnp.einsum("ntok,ntpk->op", g, g) / a.shape[1]) for a, g in data)

    @jax.jit
    def correction_sums(theta, bx, bases):
        data = jax.vmap(example_data, in_axes=(None, 0))(theta, bx)
        sums = []
        for (a, g), (qa, qg) in zip(data, bases):
            # A shared leaf's complete gradient includes cross-token terms.
            grads = jnp.einsum("nti,ntok->niok", a, g)
            rotated = jnp.einsum("ia,niok,ob->nabk", qa, grads, qg)
            sums.append(jnp.sum(rotated**2, axis=(0, 3)))
        return tuple(sums)

    def reduce_chunks(fn, theta, bx, *extra):
        total = None
        for start in range(0, len(bx), chunk_size):
            current = fn(theta, bx[start:start + chunk_size], *extra)
            total = current if total is None else jax.tree.map(jnp.add, total, current)
            # Bound live accelerator work/memory across full-dataset passes.
            jax.block_until_ready(total)
        return total

    def factors(theta, bx):
        if len(bx) < 1:
            raise ValueError("Cannot factor an empty input batch.")
        sums = reduce_chunks(covariance_sums, theta, bx)
        covs = tuple((a / len(bx), g / len(bx)) for a, g in sums)
        bases = tuple((jnp.linalg.eigh((a + a.T) / 2)[1],
                       jnp.linalg.eigh((g + g.T) / 2)[1]) for a, g in covs)
        corrections = reduce_chunks(correction_sums, theta, bx, bases)
        return tuple((a, g, qa, qg, jnp.maximum(c / len(bx), 0))
                     for (a, g), (qa, qg), c in zip(covs, bases, corrections))

    # Pure, local inspection hooks used by the accompanying tiny-batch checks.
    forward.parameter_names = names
    forward.unravel = unravel
    factors.example_data = jax.jit(example_data)
    return theta_init, forward, block_shapes, factors
