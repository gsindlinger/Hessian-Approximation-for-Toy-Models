#!/usr/bin/env python
"""One PBRF-MNIST replicate: a single (architecture, seed) run.

Ports the logic of experiments/notebooks/pbrf_mnist_reproduction.ipynb for ONE
config and writes a JSON with the raw per-example arrays + convenience
correlations. Seeded so that `seed` alone fixes: weight init, warm-train data
order, the draw of the N_DELETED points, the retrain order, and the MCMC Fisher
samples (they move together — a coupled replicate).

Meant to be launched one-per-GPU by sweep.py (via CUDA_VISIBLE_DEVICES), but is
runnable standalone:  python run_one.py --arch_idx 0 --seed 0 --out r.json
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import json
import argparse
import tempfile
import io
import contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.stats import pearsonr, spearmanr

from src.config import (
    ModelConfig,
    TrainingConfig,
    ModelArchitecture,
    ActivationFunction,
    LossType,
    OptimizerType,
    LRSchedule,
    DatasetEnum,
    PseudoTargetGenerationStrategy as PTS,
    HessianApproximationMethod as HAM,
)
from src.utils.data.data import DownloadableDataset, Dataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.train import train_model
from src.utils.optimizers import optimizer as make_optimizer
from src.utils.loss import get_loss, loss_wrapper_with_apply_fn
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.registry import HessianComputerRegistry
from src.hessians.utils.data import ModelContext
from src.utils.influence import compute_per_example_flat_grads, compute_influence_matrix

# ---- shared knobs (identical to the notebook) ----
LAMBDA = 1e-2
BATCH = 64
RETRAIN_EPOCHS = 150
N_DELETED = 320
RETRAIN_KEY = 7
MCMC_REPS = 1
METHODS = [HAM.GNH, HAM.KFAC, HAM.EKFAC, HAM.SHAMPOO, HAM.ESHAMPOO, HAM.IDENTITY]
# (name, architecture, hidden_dim, activation, lr, epochs)
ARCHS = [
    ("MLP [16x8]", ModelArchitecture.MLP, [16] * 8, ActivationFunction.TANH, 0.10, 50),
    ("ResNet 3x[16,16]", ModelArchitecture.RESNETMLP, [[16, 16]] * 3, ActivationFunction.RELU, 0.02, 3),
    ("ResNet 8x[8,16]", ModelArchitecture.RESNETMLP, [[8, 16]] * 8, ActivationFunction.RELU, 0.02, 3),
]
loss_fn = get_loss(LossType.CROSS_ENTROPY)


def load_data():
    "MNIST 10% subset, standardized; fixed across seeds (the dataset, not a nuisance var)."
    full = DownloadableDataset.load(
        DatasetEnum.MNIST,
        directory=str(PROJECT_ROOT / "experiments/datasets/mnist"),
        store_on_disk=False,
    )
    Xall = np.asarray(full.inputs, np.float64)
    Yall = np.asarray(full.targets).reshape(-1).astype(int)
    rng = np.random.RandomState(0)
    idx = rng.choice(Xall.shape[0], 7000, replace=False)  # ~10%
    Xs, Ys = Xall[idx], Yall[idx]
    Xs = (Xs - Xs.mean(0, keepdims=True)) / (Xs.std(0, keepdims=True) + 1e-6)
    NTR = 6000
    X, Y = jnp.asarray(Xs[:NTR]), jnp.asarray(Ys[:NTR])
    Xq, Yq = jnp.asarray(Xs[NTR:]), jnp.asarray(Ys[NTR:])
    return X, Y, Xq, Yq, int(X.shape[0]), 784, 10


def gnh_direct(flat, unravel, model, X, N):
    "Direct full-Jacobian GNH (exact; fast at ~15k params)."

    def single(x):
        lo = lambda fp: model.apply(unravel(fp), x[None, :]).reshape(-1)
        J = jax.jacrev(lo)(flat)
        p = jax.nn.softmax(lo(flat))
        return J.T @ ((jnp.diag(p) - jnp.outer(p, p)) @ J)

    G, _ = jax.lax.scan(
        lambda a, x: (a + single(x), None), jnp.zeros((flat.size,) * 2), X
    )
    return G / N


def run_arch(name, arch, hidden, act, lr, epochs, seed, data):
    X, Y, Xq, Yq, N, IN, OUT = data
    mc = ModelConfig(
        architecture=arch,
        input_dim=IN,
        hidden_dim=hidden,
        output_dim=OUT,
        activation=act,
        loss=LossType.CROSS_ENTROPY,
        training=TrainingConfig(
            learning_rate=lr,
            weight_decay=0.0,
            optimizer=OptimizerType.SGD,
            epochs=epochs,
            batch_size=BATCH,
            lr_schedule=LRSchedule.COSINE,
        ),
        directory=None,
    )
    opt = make_optimizer(
        OptimizerType.SGD,
        lr=lr,
        weight_decay=0.0,
        lr_schedule=LRSchedule.COSINE,
        total_steps=(N // BATCH) * epochs,
    )
    dl = JAXDataLoader(X, Y, batch_size=BATCH, shuffle=True, rng_key=jax.random.PRNGKey(seed))
    with contextlib.redirect_stderr(io.StringIO()):
        model, params, _ = train_model(
            mc, dl, loss_fn, opt, epochs=epochs, seed=seed, save_checkpoints=False
        )
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), params)
    flat, unravel = ravel_pytree(params)
    EPS = 1.0 / N
    logits = model.apply(params, X)
    soft = jax.lax.stop_gradient(logits)
    tracc = float(jnp.mean(jnp.argmax(logits, -1) == Y))
    teacc = float(jnp.mean(jnp.argmax(model.apply(params, Xq), -1) == Yq))
    rs = np.random.RandomState(seed)
    didx = jnp.asarray(rs.choice(N, N_DELETED, replace=False), dtype=jnp.int32)

    def phi(fp):
        return loss_wrapper_with_apply_fn(model.apply, fp, unravel, loss_fn, Xq, Yq)

    phi0 = float(phi(flat))
    lper = lambda f, y: optax.softmax_cross_entropy_with_integer_labels(f, y)

    def bper(f, fs):
        fs = jax.lax.stop_gradient(fs)
        ps = jax.nn.softmax(fs, -1)
        return jnp.sum(ps * (jax.nn.log_softmax(fs, -1) - jax.nn.log_softmax(f, -1)), -1)

    nfull = N // BATCH

    def retrain(mode, rlr, rlam):
        o = optax.sgd(rlr)

        def one(t):
            def bl(p, idx):
                f = model.apply(unravel(p), X[idx])
                is_t = (idx == t).astype(f.dtype)
                if mode == "warm":
                    c = (1.0 - is_t) * lper(f, Y[idx])
                    return jnp.sum(c) / jnp.maximum(jnp.sum(1.0 - is_t), 1.0)
                c = (1.0 - is_t) * bper(f, soft[idx]) + is_t * (-lper(f, Y[idx]))
                return jnp.mean(c) + 0.5 * rlam * jnp.sum((p - flat) ** 2)

            def ep(carry, ek):
                p, st = carry
                perm = jax.random.permutation(ek, N)[: nfull * BATCH].reshape(nfull, BATCH)

                def bs(c2, ii):
                    p, st = c2
                    g = jax.grad(bl)(p, ii)
                    u, st = o.update(g, st, p)
                    return (optax.apply_updates(p, u), st), None

                (p, st), _ = jax.lax.scan(bs, (p, st), perm)
                return (p, st), None

            keys = jax.random.split(jax.random.PRNGKey(RETRAIN_KEY + seed), RETRAIN_EPOCHS)
            (p, _), _ = jax.lax.scan(ep, (flat, o.init(flat)), keys)
            return phi(p)

        return np.asarray(jax.jit(jax.vmap(one))(didx)) - phi0

    dwarm = retrain("warm", lr, 0.0)
    dpbrf = retrain("pbrf", lr / 10.0, LAMBDA)
    qg = jax.grad(phi)(flat)
    tg = compute_per_example_flat_grads(model, params, X[didx], Y[didx], loss_fn)
    Gd = gnh_direct(flat, unravel, model, X, N)
    if_gnh = EPS * np.asarray(
        tg @ jnp.linalg.solve(Gd + LAMBDA * jnp.eye(flat.size), qg)
    ).reshape(-1)
    # approximators: Kronecker via repo; GNH via the direct build above.
    # MCMC pseudo-targets (seed-dependent key) so the Fisher targets the GNH.
    mctx = ModelContext.create(
        model=model, params=params, dataset=Dataset(inputs=X, targets=Y), loss_fn=loss_fn
    )

    def _collect(key):
        return CollectorActivationsGradients(
            model=model,
            params=params,
            loss_fn=loss_fn,
            pseudo_target_strategy=PTS.MCMC,
            pseudo_target_repetitions=MCMC_REPS,
        ).collect(
            dataset=Dataset(inputs=X, targets=Y),
            save_directory=tempfile.mkdtemp(),
            try_load=False,
            rng_key=jax.random.PRNGKey(key),
        )

    cdata, cdata_corr = _collect(1000 + 2 * seed), _collect(1001 + 2 * seed)
    delta_if = {HAM.GNH: if_gnh}
    for m in METHODS:
        if m == HAM.GNH:
            continue
        ctx = HessianComputerRegistry.get_compute_context(m, cdata, mctx)
        comp = HessianComputerRegistry.get_computer(m, ctx, corr_context=cdata_corr).build(
            base_directory=None
        )
        delta_if[m] = EPS * np.asarray(
            compute_influence_matrix(qg[None, :], tg, comp, damping=float(LAMBDA))
        ).reshape(-1)

    # convenience correlations per method
    corr = {}
    for m in METHODS:
        di = delta_if[m]
        corr[m.value] = dict(
            pbrf_P=float(pearsonr(di, dpbrf)[0]),
            pbrf_S=float(spearmanr(di, dpbrf)[0]),
            warm_P=float(pearsonr(di, dwarm)[0]),
            warm_S=float(spearmanr(di, dwarm)[0]),
            ratio=float(np.linalg.norm(di) / np.linalg.norm(dpbrf)),
        )
    return dict(
        name=name,
        seed=int(seed),
        params=int(flat.size),
        tracc=tracc,
        teacc=teacc,
        dwarm=dwarm.tolist(),
        dpbrf=dpbrf.tolist(),
        delta_if={m.value: np.asarray(delta_if[m]).tolist() for m in METHODS},
        corr=corr,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch_idx", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    data = load_data()
    spec = ARCHS[args.arch_idx]
    r = run_arch(*spec, seed=args.seed, data=data)
    r["arch_idx"] = args.arch_idx
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f)
    c = r["corr"]["gnh"]
    print(
        f"[done] arch={args.arch_idx} ({r['name']}) seed={args.seed} "
        f"trAcc={r['tracc']:.3f} teAcc={r['teacc']:.3f} | "
        f"GNH-IF PBRF P/S={c['pbrf_P']:+.2f}/{c['pbrf_S']:+.2f} "
        f"warm P/S={c['warm_P']:+.2f}/{c['warm_S']:+.2f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
