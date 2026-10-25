# %% [markdown]
# # Initial-gradient perturbations through SGD
#
# Freeze v_i = grad loss_i(theta_start), inject theta_start +/- epsilon v_i,
# and replay identical minibatches. Along the unperturbed trajectory propagate
# u_(t+1) = u_t - lr H_hat(theta_t) u_t, with u_0 = v_i.
# Read out with q = grad SUM_test loss(theta_T): predict q dot u_T.
#
# This is a fresh, local plain-SGD experiment, not replay of a saved optimizer.
# It tests an INITIAL weight perturbation, not data deletion or SOURCE's
# repeated data-weight forcing. Zero curvature and identity curvature differ.

# %%
from pathlib import Path
import argparse
import csv
import json


def experiment_directory():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidate = parent / "experiments" / "unrolling"
        if (candidate / "unrolling.py").is_file():
            return candidate
    raise RuntimeError("Open the notebook from inside the repository.")


HERE = experiment_directory()
CONFIG = {
    "architecture": "mlp",
    "split_dir": "experiments/datasets/digits/splits/default",
    "loss": "cross_entropy",       # or mse; reads existing NPZ files, no downloads
    "widths": [16] * 8,            # bias-free Small MLP8 architecture
    "activation": "tanh",          # or relu; tanh makes finite-epsilon checks smooth
    "seed": 42,
    "learning_rate": 0.03,
    "batch_size": 32,
    "start_step": 0,               # optional local SGD warmup, not an external checkpoint
    "horizons": [0, 1, 10, 50, 100],
    "num_probes": 16,
    "normalize_probes": False,     # default: inject RAW per-example gradients
    "epsilons": [1e-4, 1e-3, 1e-2],
    "methods": [
        "zero", "identity", "h_batch", "h_full", "gnh_batch", "gnh_full",
        "block_h_batch", "kfac_full", "ekfac_full", "foof_full",
    ],                            # kfac_batch / ekfac_batch / foof_batch also supported
}
RUN = False                       # notebook execution gate; deliberately closed
DEFAULT_CONFIG = CONFIG


def preset(name, seed=42):
    """Two requested model families; still one explicit run, never a sweep."""
    cfg = dict(DEFAULT_CONFIG, seed=seed)
    if name == "mlp3":
        cfg.update(architecture="mlp", widths=[16] * 3, activation="tanh",
                   learning_rate=0.03)
    elif name == "resnet3":
        cfg.update(architecture="resnet", widths=[16] * 3, activation="relu",
                   learning_rate=0.003, epsilons=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    elif name == "transformer":
        cfg.update(architecture="transformer", widths=[], activation="relu",
                   split_dir="experiments/datasets/modular_addition/splits/v2_4e589aa773582474",
                   output_dim=99, transformer_dim=16, transformer_num_layers=2,
                   transformer_num_heads=4, transformer_vocab_size=99,
                   transformer_max_length=5, factor_batch_size=16,
                   learning_rate=0.001, epsilons=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    else:
        raise ValueError(f"Unknown model preset: {name}")
    return cfg


MODEL = "resnet3"                  # alternatively "transformer" or plain "mlp3"
CONFIG = preset(MODEL, seed=42)
OUTPUT_NAME = f"{MODEL}_seed42"     # writes only HERE/outputs/NAME; refuses overwrite

# %% [markdown]
# ## Local model and curvature operators
#
# All numerical imports are deferred until run_experiment is explicitly called.
# There are no imports from src, the experiment runner, collectors, or database.
# KFAC/EKFAC use exact output curvature: diag(p)-pp^T for cross entropy,
# 2 I / output_dimension for mean-squared error. No empirical-Fisher labels
# or sampled pseudo-targets. Full means the full training partition at the
# CURRENT baseline weights, not curvature frozen at initialization.

# %%
def run_experiment(config, *, output_name=None):
    import os
    import time
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from jax.flatten_util import ravel_pytree

    cfg = dict(config)
    architecture = cfg.get("architecture", "mlp")
    if architecture not in ("mlp", "resnet", "transformer"):
        raise ValueError("Expected mlp, resnet, or transformer architecture.")
    methods = list(cfg["methods"])
    allowed = {"zero", "identity", "h_batch", "h_full", "gnh_batch", "gnh_full",
               "block_h_batch", "block_h_full",
               *[f"{m}_{s}" for m in ("kfac", "ekfac", "foof") for s in ("batch", "full")]}
    if not {"zero", "h_batch"}.issubset(methods) or not set(methods) <= allowed:
        raise ValueError("Include zero and h_batch; use only the documented methods.")
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate methods.")
    horizons = sorted(set(cfg["horizons"]))
    eps = np.asarray(cfg["epsilons"], dtype=float)
    lr = float(cfg["learning_rate"])
    if (not horizons or any(not isinstance(h, int) or h < 0 for h in horizons)
            or max(horizons) < 1 or not isinstance(cfg["start_step"], int) or cfg["start_step"] < 0
            or cfg["batch_size"] < 1 or cfg["num_probes"] < 3
            or not np.isfinite(lr) or lr <= 0
            or len(eps) < 2 or not np.all(np.isfinite(eps) & (eps > 0))):
        raise ValueError("Check horizons, warmup, batch/probe counts, lr and epsilon grid.")
    if cfg["activation"] not in ("tanh", "relu") or cfg["loss"] not in ("cross_entropy", "mse"):
        raise ValueError("Supported activations: tanh/relu; losses: cross_entropy/mse.")
    if any(not isinstance(w, int) or w < 1 for w in cfg["widths"]):
        raise ValueError("Widths must be positive integers.")
    split = Path(cfg["split_dir"])
    if not split.is_absolute():
        split = HERE.parents[1] / split
    with np.load(split / "train" / "data.npz", allow_pickle=False) as data:
        x, y = data["inputs"].astype(float), data["targets"]
    with np.load(split / "test" / "data.npz", allow_pickle=False) as data:
        xq, yq = data["inputs"].astype(float), data["targets"]
    if (x.ndim != 2 or xq.ndim != 2 or not len(x) or not len(xq)
            or x.shape[1] != xq.shape[1] or len(y) != len(x) or len(yq) != len(xq)):
        raise ValueError("Expected nonempty tabular/image-vector train and test arrays.")
    if cfg["num_probes"] > len(x) or cfg["batch_size"] > len(x):
        raise ValueError("Probe count and batch size cannot exceed the training size.")
    normalization = {}
    if cfg["loss"] == "mse":
        y, yq = y.reshape(len(y), -1).astype(float), yq.reshape(len(yq), -1).astype(float)
        for name, train, query in (("x", x, xq), ("y", y, yq)):
            mean, std = train.mean(0), train.std(0)
            std = np.where(std > 1e-12, std, 1.0)
            normalization[name] = {"mean": mean.tolist(), "std": std.tolist()}
            train -= mean
            train /= std
            query -= mean
            query /= std
        output_dim = y.shape[1]
    else:
        # Stored Digits/MNIST inputs already have their pixel normalization.
        if (y.size != len(x) or yq.size != len(xq)
                or not np.isfinite(y).all() or not np.isfinite(yq).all()
                or not np.equal(y, np.floor(y)).all() or not np.equal(yq, np.floor(yq)).all()):
            raise ValueError("Classification requires one finite integer label per example.")
        y, yq = y.reshape(-1).astype(np.int32), yq.reshape(-1).astype(np.int32)
        output_dim = int(cfg.get("output_dim", int(y.max()) + 1))
        if y.min() < 0 or yq.min() < 0 or yq.max() >= output_dim:
            raise ValueError("Classification targets must be contiguous nonnegative labels.")
    if not all(np.isfinite(a).all() for a in (x, y, xq, yq)):
        raise ValueError("Nonfinite dataset values.")
    if architecture == "transformer":
        if cfg["loss"] != "cross_entropy":
            raise ValueError("The modular-addition transformer uses cross entropy.")
        if not all(np.equal(a, np.floor(a)).all() for a in (x, xq)):
            raise ValueError("Transformer inputs must be integer tokens.")
        x, xq = x.astype(np.int32), xq.astype(np.int32)
    x, y, xq, yq = map(jnp.asarray, (x, y, xq, yq))
    if architecture == "resnet":
        if not cfg["widths"] or len(set(cfg["widths"])) != 1 or cfg["activation"] != "relu":
            raise ValueError("Local residual MLP supports equal-width, two-affine ReLU blocks.")
        widths = [x.shape[1], *([cfg["widths"][0]] * (1 + 2 * len(cfg["widths"]))), output_dim]
    else:
        widths = [x.shape[1], *cfg["widths"], output_dim]
    keys = jax.random.split(jax.random.PRNGKey(cfg["seed"]), len(widths) - 1)
    initializer = jax.nn.initializers.xavier_uniform() if architecture == "resnet" else jax.nn.initializers.lecun_normal()
    weights = tuple(initializer(key, (a, b), jnp.float64)
                    for key, a, b in zip(keys, widths[:-1], widths[1:]))
    theta_init, unravel = ravel_pytree(weights)
    block_shapes = tuple(w.shape for w in weights)
    offsets = np.cumsum([0, *[w.size for w in weights]])
    activation = jnp.tanh if cfg["activation"] == "tanh" else jax.nn.relu

    def forward(theta, inputs, cache=False):
        incoming, preacts = [], []
        ws = unravel(theta)
        if architecture == "resnet":
            incoming.append(inputs)
            inputs = inputs @ ws[0]
            preacts.append(inputs)
            for layer in range(1, len(ws) - 1, 2):
                incoming.append(inputs)
                z = inputs @ ws[layer]
                preacts.append(z)
                hidden = jax.nn.relu(z)
                incoming.append(hidden)
                branch = hidden @ ws[layer + 1]
                preacts.append(branch)
                inputs = inputs + branch
            incoming.append(inputs)
            inputs = inputs @ ws[-1]
            preacts.append(inputs)
            return (inputs, incoming, preacts) if cache else inputs
        for layer, w in enumerate(ws):
            incoming.append(inputs)
            z = inputs @ w
            preacts.append(z)
            inputs = activation(z) if layer < len(ws) - 1 else z
        return (inputs, incoming, preacts) if cache else inputs

    external_factors = None
    if architecture == "transformer":
        # Sibling-only adapter: no import through src or the shared experiment package.
        import importlib.util
        spec = importlib.util.spec_from_file_location("unrolling_transformer", HERE / "transformer.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        theta_init, forward, block_shapes, external_factors = adapter.make_transformer(
            cfg, cfg["seed"], x.shape[1], output_dim)
        offsets = np.cumsum([0, *[int(np.prod(shape)) for shape in block_shapes]])

    def losses(logits, targets):
        if cfg["loss"] == "mse":
            return jnp.mean((logits - targets) ** 2, axis=-1)
        return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None], axis=1)[:, 0]

    def loss(theta, bx, by):
        return losses(forward(theta, bx), by).mean()

    grad = jax.grad(loss)
    step = jax.jit(lambda theta, bx, by: theta - lr * grad(theta, bx, by))
    multi_step = jax.jit(jax.vmap(step, in_axes=(0, None, None)))
    query = jax.jit(lambda theta: losses(forward(theta, xq), yq).sum())
    query_grad = jax.jit(jax.grad(query))
    multi_query = jax.jit(jax.vmap(query))

    @jax.jit
    def h_products(theta, vectors, bx, by):
        _, multiply = jax.linearize(lambda p: grad(p, bx, by), theta)
        return jax.vmap(multiply)(vectors)

    @jax.jit
    def gnh_products(theta, vectors, bx, by):
        z, jv = jax.linearize(lambda p: forward(p, bx), theta)
        _, jt = jax.vjp(lambda p: forward(p, bx), theta)
        _, cv = jax.linearize(jax.grad(lambda z: losses(z, by).mean()), z)
        return jax.vmap(lambda v: jt(cv(jv(v)))[0])(vectors)

    @jax.jit
    def block_products(theta, vectors, bx, by):
        result = jnp.zeros_like(vectors)
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            masked = jnp.zeros_like(vectors).at[:, lo:hi].set(vectors[:, lo:hi])
            result = result.at[:, lo:hi].set(h_products(theta, masked, bx, by)[:, lo:hi])
        return result

    @jax.jit
    def factors(theta, bx):
        z, incoming, preacts = forward(theta, bx, cache=True)
        ws, n = unravel(theta), len(bx)
        if cfg["loss"] == "mse":
            output_cov = jnp.broadcast_to(2 * jnp.eye(output_dim) / output_dim,
                                         (n, output_dim, output_dim))
        else:
            p = jax.nn.softmax(z)
            output_cov = jax.vmap(jnp.diag)(p) - p[:, :, None] * p[:, None, :]
        # B[n,c,o] = derivative of logit c wrt this layer's preactivation o.
        back = jnp.broadcast_to(jnp.eye(output_dim), (n, output_dim, output_dim))
        if architecture == "resnet":
            backs = [None] * len(ws)
            backs[-1] = back
            downstream = jnp.einsum("nco,io->nci", back, ws[-1])
            for layer in range(len(ws) - 3, 0, -2):
                backs[layer + 1] = downstream
                backs[layer] = jnp.einsum("nco,io->nci", downstream, ws[layer + 1]) * (preacts[layer] > 0)[:, None, :]
                downstream = downstream + jnp.einsum("nco,io->nci", backs[layer], ws[layer])
            backs[0] = downstream
        result = []
        for layer in range(len(ws) - 1, -1, -1):
            if architecture == "resnet":
                back = backs[layer]
            a = incoming[layer]
            per_example_g = jnp.einsum("nco,ncd,ndp->nop", back, output_cov, back)
            acov, gcov = a.T @ a / n, per_example_g.mean(0)
            _, qa = jnp.linalg.eigh((acov + acov.T) / 2)
            _, qg = jnp.linalg.eigh((gcov + gcov.T) / 2)
            gdiag = jnp.einsum("oa,nop,pa->na", qg, per_example_g, qg)
            corrected = ((a @ qa) ** 2).T @ gdiag / n  # no division by class count
            result.append((acov, gcov, qa, qg, jnp.maximum(corrected, 0)))
            if layer and architecture != "resnet":
                back = jnp.einsum("nco,io->nci", back, ws[layer])
                previous = preacts[layer - 1]
                derivative = 1 - jnp.tanh(previous) ** 2 if cfg["activation"] == "tanh" else (previous > 0)
                back = back * derivative[:, None, :]
        return tuple(reversed(result))

    if external_factors is not None:
        factors = external_factors

    def factor_products(vectors, stats, kind):
        pieces = []
        for shape, lo, hi, (a, g, qa, qg, corrected) in zip(block_shapes, offsets[:-1], offsets[1:], stats):
            v = vectors[:, lo:hi].reshape((-1, *shape))
            if kind == "foof":
                product = jnp.einsum("ij,njk->nik", a, v)
            elif kind == "kfac":
                product = jnp.einsum("ij,njk,kl->nil", a, v, g)
            else:
                rotated = jnp.einsum("ia,nij,jb->nab", qa, v, qg)
                product = jnp.einsum("ia,nab,jb->nij", qa, corrected * rotated, qg)
            pieces.append(product.reshape((len(vectors), -1)))
        return jnp.concatenate(pieces, axis=1)

    factor_apply = {m: jax.jit(lambda v, f, kind=m: factor_products(v, f, kind))
                    for m in ("kfac", "ekfac", "foof")}
    batch_seed, probe_seed = np.random.SeedSequence(cfg["seed"]).spawn(2)
    rng = np.random.default_rng(batch_seed)
    count = (cfg["start_step"] + max(horizons)) * cfg["batch_size"]
    # Fixed random-reshuffling stream; a batch may straddle permutation boundaries.
    stream = np.concatenate([rng.permutation(len(x)) for _ in range((count + len(x) - 1) // len(x))])
    all_batches = jnp.asarray(stream[:count].reshape((-1, cfg["batch_size"])))

    @jax.jit
    def replay(initial, batches):
        def update(theta, ids):
            return theta - lr * grad(theta, x[ids], y[ids]), None
        return jax.lax.scan(update, initial, batches)[0]

    theta0 = replay(theta_init, all_batches[:cfg["start_step"]])
    batches = all_batches[cfg["start_step"]:]
    probe_ids = np.random.default_rng(probe_seed).choice(len(x), cfg["num_probes"], replace=False)
    raw_probes = jax.vmap(lambda xi, yi: grad(theta0, xi[None, :], yi[None]))(x[probe_ids], y[probe_ids])
    probe_norms = jnp.linalg.norm(raw_probes, axis=1)
    if not bool(jnp.all(jnp.isfinite(probe_norms) & (probe_norms > 0))):
        raise ValueError("Zero/nonfinite probe gradient: inspect the setup; no automatic resampling.")
    probes = raw_probes / probe_norms[:, None] if cfg["normalize_probes"] else raw_probes
    tangent = {method: probes for method in methods}
    plus = theta0 + jnp.asarray(eps)[:, None, None] * probes[None, :, :]
    minus = theta0 - jnp.asarray(eps)[:, None, None] * probes[None, :, :]
    theta, snapshots, rows = theta0, {}, []
    result = {
        "config": cfg, "normalization": normalization, "rows": rows, "snapshots": snapshots,
        "status": "running",
        "protocol": {"theta_init": np.asarray(theta_init), "theta_start": np.asarray(theta0),
                     "batch_indices": np.asarray(all_batches), "probe_ids": probe_ids,
                     "raw_probes": np.asarray(raw_probes), "probes": np.asarray(probes),
                     "probe_norms": np.asarray(probe_norms),
                     "block_shapes": np.asarray(block_shapes)},
    }
    destination = result_directory(output_name) if output_name else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for t in range(max(horizons) + 1):
        if t in horizons:
            # Independent differentiation of the entire replay, on two fixed probes.
            exact_jvp = jax.jit(jax.vmap(lambda v: jax.jvp(
                lambda p: replay(p, batches[:t]), (theta0,), (v,))[1]))(probes[:2])
            if not all(np.isfinite(np.asarray(v)).all() for v in (exact_jvp, tangent["h_batch"])):
                raise FloatingPointError(f"Nonfinite exact derivative at step {t}.")
            np.testing.assert_allclose(np.asarray(tangent["h_batch"][:2]), np.asarray(exact_jvp),
                                       rtol=1e-7, atol=1e-9, err_msg=f"Replay JVP failed at step {t}")
            np.testing.assert_allclose(np.asarray(theta), np.asarray(replay(theta0, batches[:t])),
                                       rtol=1e-9, atol=1e-10)
            qp = multi_query(plus.reshape((-1, theta0.size))).reshape(plus.shape[:2])
            qm = multi_query(minus.reshape((-1, theta0.size))).reshape(minus.shape[:2])
            snap = {key: np.asarray(value) for key, value in {
                "theta": theta, "query_gradient": query_grad(theta), "query_loss": query(theta),
                "train_loss": loss(theta, x, y), "responses": jnp.stack([tangent[m] for m in methods]),
                "plus_delta": plus - theta, "minus_delta": minus - theta,
                "query_plus": qp, "query_minus": qm, "replay_jvp": exact_jvp,
            }.items()}
            if not all(np.isfinite(snap[key]).all() for key in ("query_loss", "train_loss", "query_gradient")):
                raise FloatingPointError(f"Nonfinite baseline query/evaluation at step {t}.")
            bad = [m for m, u in zip(methods, snap["responses"]) if not np.isfinite(u).all()]
            if bad:
                print(f"WARNING step={t}: nonfinite surrogate responses: {bad}; see valid counts.", flush=True)
            snapshots[t] = snap
            rows.extend(measure(snap, np.asarray(probes), methods, eps, t))
            print(f"step={t}: train loss={float(snap['train_loss']):.6g}; "
                  f"mean query loss={float(snap['query_loss']) / len(xq):.6g}; "
                  f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
            result["status"] = "complete" if t == max(horizons) else "running"
            if destination is not None:
                save_results(result, output_name, _owned_directory=destination)
        if t == max(horizons):
            break
        bx, by = x[batches[t]], y[batches[t]]
        stats = {}
        for scope in ("batch", "full"):
            if any(m in methods for m in (f"kfac_{scope}", f"ekfac_{scope}", f"foof_{scope}")):
                stage = time.perf_counter()
                stats[scope] = factors(theta, x if scope == "full" else bx)
                if t == 0:
                    jax.block_until_ready(stats[scope])
                    print(f"first-step {scope} factors: {time.perf_counter() - stage:.2f}s", flush=True)
        for method, u in tangent.items():
            if method in ("zero", "identity"):
                product = jnp.zeros_like(u) if method == "zero" else u
            else:
                kind, scope = method.rsplit("_", 1)
                cx, cy = (x, y) if scope == "full" else (bx, by)
                stage = time.perf_counter()
                if kind in factor_apply:
                    product = factor_apply[kind](u, stats[scope])
                else:
                    product = {"h": h_products, "gnh": gnh_products, "block_h": block_products}[kind](theta, u, cx, cy)
                if t == 0:
                    product.block_until_ready()
                    print(f"first-step {method}: {time.perf_counter() - stage:.2f}s", flush=True)
            tangent[method] = u - lr * product
        theta = step(theta, bx, by)
        plus = multi_step(plus.reshape((-1, theta0.size)), bx, by).reshape(plus.shape)
        minus = multi_step(minus.reshape((-1, theta0.size)), bx, by).reshape(minus.shape)
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in (theta, plus, minus)):
            raise FloatingPointError(f"Nonfinite baseline/perturbed training at step {t + 1}.")
    return result

# %% [markdown]
# ## Metrics and optional local output
#
# Scores rank TRAINING probes for one fixed summed-test-loss query. Cosine is
# supplementary: zero and identity can have identical directional rankings.
# Report finite-epsilon validity separately from curvature-approximation error.
# Undefined ratios/cosines (notably zero curvature's correction cosine) are NaN
# in arrays/CSV, with explicit valid counts; they are not silently replaced by 0.

# %%
def measure(snap, probes, methods, epsilons, horizon):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    def divide(a, b):
        a, b = np.broadcast_arrays(a, b)
        return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float),
                         where=np.isfinite(a) & np.isfinite(b) & (b > 0))

    def cosine(a, b):
        return divide(np.sum(a * b, axis=-1), np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))

    def average(a):
        valid = np.isfinite(a)
        return float(np.mean(a[valid])) if np.any(valid) else float("nan")

    def correlation(a, b, kind):
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() < 3 or np.ptp(a[good]) == 0 or np.ptp(b[good]) == 0:
            return float("nan")
        return float((spearmanr if kind == "spearman" else pearsonr)(a[good], b[good]).statistic)

    exact = snap["responses"][methods.index("h_batch")]
    q = snap["query_gradient"]
    rows = []
    for epsilon_index, epsilon in enumerate(epsilons):
        forward = snap["plus_delta"][epsilon_index] / epsilon
        backward = -snap["minus_delta"][epsilon_index] / epsilon
        fd = (forward + backward) / 2
        score_fd = (snap["query_plus"][epsilon_index] - snap["query_minus"][epsilon_index]) / (2 * epsilon)
        exact_norm = np.linalg.norm(exact, axis=1)
        fd_norm = np.linalg.norm(fd, axis=1)
        correction_norm = np.linalg.norm(exact - probes, axis=1)
        for method, response in zip(methods, snap["responses"]):
            score = response @ q
            correction_cos = cosine(response - probes, exact - probes)
            rows.append({
                "horizon": horizon, "epsilon": float(epsilon), "method": method,
                "status": "finite" if np.isfinite(response).all() else "nonfinite_response",
                "total_probes": len(probes),
                "valid_responses": int(np.isfinite(response).all(axis=1).sum()),
                "valid_score_pairs": int((np.isfinite(score) & np.isfinite(score_fd)).sum()),
                "valid_exact_norms": int((np.isfinite(exact_norm) & (exact_norm > 0)).sum()),
                "valid_fd_norms": int((np.isfinite(fd_norm) & (fd_norm > 0)).sum()),
                "valid_correction_norms": int((np.isfinite(correction_norm) & (correction_norm > 0)).sum()),
                "response_rel_l2_exact": average(divide(np.linalg.norm(response - exact, axis=1), exact_norm)),
                "correction_rel_error": average(divide(np.linalg.norm(response - exact, axis=1), correction_norm)),
                "correction_cosine": average(correction_cos),
                "valid_correction_cosines": int(np.isfinite(correction_cos).sum()),
                "response_rel_l2_fd": average(divide(np.linalg.norm(response - fd, axis=1), fd_norm)),
                "response_cosine_fd": average(cosine(response, fd)),
                "query_cosine": average(cosine(response, q)),
                "query_cosine_mae_fd": average(np.abs(cosine(response, q) - cosine(fd, q))),
                "score_rel_l2_exact": float(divide(np.linalg.norm(score - exact @ q), np.linalg.norm(exact @ q))),
                "score_rel_l2_fd": float(divide(np.linalg.norm(score - score_fd), np.linalg.norm(score_fd))),
                "score_spearman_exact": correlation(score, exact @ q, "spearman"),
                "score_spearman_fd": correlation(score, score_fd, "spearman"),
                "score_pearson_fd": correlation(score, score_fd, "pearson"),
                "forward_backward_rel_gap": average(divide(np.linalg.norm(forward - backward, axis=1), fd_norm)),
            })
    return rows


def result_directory(name):
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("Use a simple output folder name, not a path.")
    destination = HERE / "outputs" / name
    if not destination.resolve().is_relative_to(HERE.resolve()):
        raise ValueError("Output symlinks must not redirect writes outside this experiment folder.")
    return destination


def save_results(result, name, *, _owned_directory=None):
    import numpy as np
    destination = result_directory(name)
    if _owned_directory is None:
        destination.mkdir(parents=True, exist_ok=False)
    elif Path(_owned_directory) != destination or not destination.is_dir():
        raise ValueError("Incremental writes require this run's newly reserved directory.")
    (destination / "config.json").write_text(json.dumps({
        "config": result["config"], "normalization": result["normalization"],
        "query": "sum of ALL saved test-example losses",
        "optimizer": "local constant-lr SGD; no momentum, weight decay, or dropout",
        "precision": "float64", "jvp_validation": "first two fixed probes at every horizon",
        "status": result.get("status", "complete"),
    }, indent=2) + "\n")
    if not (destination / "protocol.npz").exists():
        np.savez_compressed(destination / "protocol.npz", **result["protocol"])
    for horizon, snapshot in result["snapshots"].items():
        np.savez_compressed(destination / f"step_{horizon:05d}.npz", **snapshot)
    with (destination / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(f"Saved only to {destination}")
    return destination


def plot_results(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    epsilon = min(result["config"]["epsilons"])
    for method in result["config"]["methods"]:
        rows = [r for r in result["rows"] if r["method"] == method and r["epsilon"] == epsilon]
        for axis, metric, title in zip(axes,
                ("correction_rel_error", "response_rel_l2_fd", "score_spearman_fd"),
                ("Error / exact curvature correction", "Endpoint error vs finite difference", "Query-score Spearman vs finite difference")):
            axis.plot([r["horizon"] for r in rows], [r[metric] for r in rows], marker=".", label=method)
            axis.set(xlabel="SGD steps after injection", title=title)
            axis.grid(alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"Smallest epsilon = {epsilon:g}; inspect the FULL epsilon grid before interpreting")
    fig.tight_layout(rect=(0, .15, 1, .9))
    return fig

# %% [markdown]
# ## Run only after approval
#
# Set RUN = True in the configuration cell to execute. Reusing an output name
# is refused; choose a new name deliberately. Inspect all epsilon values in
# metrics.csv, particularly the h_batch row, before interpreting approximations.

# %%
if RUN and "__file__" not in globals():
    if (HERE / "outputs" / OUTPUT_NAME).exists():
        raise FileExistsError("Choose a new OUTPUT_NAME before running.")
    result = run_experiment(CONFIG, output_name=OUTPUT_NAME)
    output = result_directory(OUTPUT_NAME)
    figure = plot_results(result)
    figure.savefig(output / "summary.png", dpi=160)

# %% cli
def write_notebook():
    """Convert percent cells using stdlib JSON only; NEVER execute a cell."""
    cells, lines, kind = [], [], None

    def append():
        if kind is None or kind == "cli" or not "".join(lines).strip():
            return
        source = "".join(lines)
        if kind == "markdown":
            source = "".join(line[2:] if line.startswith("# ") else
                             line[1:] if line.startswith("#") else line
                             for line in lines)
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in (HERE / "unrolling.py").read_text().splitlines(keepends=True):
        if line.startswith("# %%"):
            append()
            kind = "markdown" if "[markdown]" in line else "cli" if "cli" in line else "code"
            lines = []
        else:
            lines.append(line)
    append()
    notebook = {"nbformat": 4, "nbformat_minor": 4, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }}
    destination = HERE / "unrolling.ipynb"
    if destination.exists():
        old = json.loads(destination.read_text())
        if any(c.get("outputs") or c.get("execution_count") is not None for c in old["cells"] if c["cell_type"] == "code"):
            raise FileExistsError("Refusing to replace an executed notebook; preserve it first.")
    destination.write_text(json.dumps(notebook, indent=1) + "\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone unrolling POC. No execution by default.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--notebook", action="store_true", help="Generate an UNEXECUTED notebook; stdlib only.")
    mode.add_argument("--run", action="store_true", help="Explicitly execute the CONFIG above and save locally.")
    parser.add_argument("--model", choices=("mlp3", "resnet3", "transformer"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true", help="With --run: only horizons 0 and 1, same setup.")
    args = parser.parse_args()
    if args.notebook:
        write_notebook()
    elif args.run:
        cfg = preset(args.model, args.seed) if args.model else dict(CONFIG, seed=args.seed)
        name = f"{args.model}_seed{args.seed}" if args.model else OUTPUT_NAME
        if args.check:
            cfg = dict(cfg, horizons=[0, 1])
            name += "_check"
        if (HERE / "outputs" / name).exists():
            raise FileExistsError("Choose a new OUTPUT_NAME before running.")
        result = run_experiment(cfg, output_name=name)
        output = result_directory(name)
        plot_results(result).savefig(output / "summary.png", dpi=160)
    elif not args.run:
        parser.print_help()
