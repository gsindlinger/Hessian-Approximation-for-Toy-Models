#!/usr/bin/env python
"""Ground-truth noise ceiling for the PBRF/warm retrains (diagnosis step 2).

Recomputes the PBRF (and warm) ground truth twice with independent retrain
batch-order keys and correlates the two replicates per architecture. The
PBRF self-correlation is the ceiling any IF estimate can reach: if the ResNet
PBRF only agrees with itself at Spearman ~0.8, then GNH-IF at 0.75-0.90 is at
the SGD-retrain noise floor and nothing is broken. Also reports GNH-IF vs each
PBRF replicate, so IF's distance to the ceiling is read off the same table.

Matches the notebook (pbrf_mnist_reproduction.ipynb): seed 0, 40 deleted
points, key 7 for replicate A (reproduces the notebook's ground truth exactly).
Skips the Kronecker approximators entirely. ~4 retrain passes + 1 dense GNH
per arch, sequential on one GPU (~15 min for all three archs).

  python noise_ceiling.py                 # all three archs, keys 7 107
  python noise_ceiling.py --archs 1 2     # ResNets only
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import json
import argparse
import io
import contextlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.stats import pearsonr, spearmanr

from run_one import (
    ARCHS,
    BATCH,
    LAMBDA,
    RETRAIN_EPOCHS,
    gnh_direct,
    load_data,
    loss_fn,
)
from src.config import LossType, ModelConfig, OptimizerType, LRSchedule, TrainingConfig
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.train import train_model
from src.utils.optimizers import optimizer as make_optimizer
from src.utils.loss import loss_wrapper_with_apply_fn
from src.utils.influence import compute_per_example_flat_grads


def _corr(a, b):
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def run_ceiling(name, arch, hidden, act, lr, epochs, seed, rkeys, n_deleted, data):
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
    soft = jax.lax.stop_gradient(model.apply(params, X))
    tracc = float(jnp.mean(jnp.argmax(soft, -1) == Y))
    rs = np.random.RandomState(seed)
    didx = jnp.asarray(rs.choice(N, n_deleted, replace=False), dtype=jnp.int32)

    def phi(fp):
        return loss_wrapper_with_apply_fn(model.apply, fp, unravel, loss_fn, Xq, Yq)

    phi0 = float(phi(flat))
    lper = lambda f, y: optax.softmax_cross_entropy_with_integer_labels(f, y)

    def bper(f, fs):
        fs = jax.lax.stop_gradient(fs)
        ps = jax.nn.softmax(fs, -1)
        return jnp.sum(ps * (jax.nn.log_softmax(fs, -1) - jax.nn.log_softmax(f, -1)), -1)

    nfull = N // BATCH

    def retrain(mode, rlr, rlam, rkey):
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

            keys = jax.random.split(jax.random.PRNGKey(rkey), RETRAIN_EPOCHS)
            (p, _), _ = jax.lax.scan(ep, (flat, o.init(flat)), keys)
            return phi(p)

        return np.asarray(jax.jit(jax.vmap(one))(didx)) - phi0

    dpbrf = {k: retrain("pbrf", lr / 10.0, LAMBDA, k) for k in rkeys}
    dwarm = {k: retrain("warm", lr, 0.0, k) for k in rkeys}

    qg = jax.grad(phi)(flat)
    tg = compute_per_example_flat_grads(model, params, X[didx], Y[didx], loss_fn)
    Gd = gnh_direct(flat, unravel, model, X, N)
    if_gnh = (1.0 / N) * np.asarray(
        tg @ jnp.linalg.solve(Gd + LAMBDA * jnp.eye(flat.size), qg)
    ).reshape(-1)

    kA, kB = rkeys[0], rkeys[1]
    ceil_pbrf = _corr(dpbrf[kA], dpbrf[kB])
    ceil_warm = _corr(dwarm[kA], dwarm[kB])
    if_vs = {k: _corr(if_gnh, dpbrf[k]) for k in rkeys}
    print(f"\n### {name}  (trAcc={tracc:.3f}, seed={seed}, keys={kA}/{kB}, n_del={n_deleted})")
    print(f"  PBRF ceiling  PBRF_A<->PBRF_B : P={ceil_pbrf[0]:+.3f}  S={ceil_pbrf[1]:+.3f}")
    print(f"  warm ceiling  warm_A<->warm_B : P={ceil_warm[0]:+.3f}  S={ceil_warm[1]:+.3f}")
    for k in rkeys:
        p, s = if_vs[k]
        print(f"  GNH-IF <-> PBRF_key{k:<4}        : P={p:+.3f}  S={s:+.3f}")
    print(
        f"  scales: |dpbrf| std A/B = {dpbrf[kA].std():.2e}/{dpbrf[kB].std():.2e}   "
        f"dwarm mean+/-std A = {dwarm[kA].mean():+.3f}+/-{dwarm[kA].std():.3f}"
    )
    return dict(
        name=name,
        seed=seed,
        keys=list(rkeys),
        n_deleted=n_deleted,
        tracc=tracc,
        ceil_pbrf=ceil_pbrf,
        ceil_warm=ceil_warm,
        if_vs_pbrf={str(k): if_vs[k] for k in rkeys},
        dpbrf={str(k): v.tolist() for k, v in dpbrf.items()},
        dwarm={str(k): v.tolist() for k, v in dwarm.items()},
        if_gnh=if_gnh.tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keys", type=int, nargs=2, default=[7, 107],
                    help="two retrain batch-order keys; 7 = the notebook's ground truth")
    ap.add_argument("--n_deleted", type=int, default=40, help="40 = notebook; sweep used 320")
    ap.add_argument("--out", type=str, default=str(HERE / "results_pbrf" / "noise_ceiling.json"))
    args = ap.parse_args()

    data = load_data()
    results = [
        run_ceiling(*ARCHS[a], seed=args.seed, rkeys=args.keys,
                    n_deleted=args.n_deleted, data=data)
        for a in args.archs
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"\nsaved -> {args.out}")
    print("read-out: if GNH-IF<->PBRF ~= the PBRF ceiling, IF is noise-limited, not broken.")


if __name__ == "__main__":
    main()
