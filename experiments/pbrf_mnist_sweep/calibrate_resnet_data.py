#!/usr/bin/env python
"""Calibrate the capacity-limited ResNet config (diagnosis step 4a).

The noise-ceiling run showed the PBRF ground truth is ~deterministic
(self-Spearman 0.998+), so the ResNets' IF gap is real and traces to the
3-epoch early stop (non-stationary theta_s, immature low-rank GNH). Fix: give
the same ResNets enough data that they become capacity-limited and can train
long, like the MLP. This script sweeps (arch, NTR, lr) x 50 epochs and records
per-epoch train/test acc + FULL-batch grad norm, to pick the config where
train acc plateaus at ~0.94-0.96 with a small grad norm (MLP reference at 10%
data: trAcc 0.955, grad norm 0.27). Flags memorization (trAcc > 0.99 = bad).

  python calibrate_resnet_data.py                     # both ResNets, NTR {12k,18k}, lr {0.02,0.05}
  python calibrate_resnet_data.py --ntrs 12000 --lrs 0.02

Round-1 verdict (2026-07-23): more MNIST does NOT help — 12k and 18k are both
fully memorized by epoch ~22 (interpolation time is flat in NTR), lr 0.05
diverges (NaN). Hence round 2, two alternative levers:
  # (A) harder task, same nets (the digits->MNIST trick again):
  python calibrate_resnet_data.py --dataset fashion_mnist --archs 1 2 --ntrs 6000 --lrs 0.02
  python calibrate_resnet_data.py --dataset fashion_mnist --archs 0 --ntrs 6000 --lrs 0.1 \
      --out results_pbrf/calibrate_fashion_mlp.json
  # (B) narrower residual stream (the 784->stream embedding is the capacity):
  python calibrate_resnet_data.py --specs 3:16,8 3:16,6 8:8,8 --ntrs 6000 --lrs 0.02
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import json
import argparse
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

from run_one import ARCHS, BATCH, loss_fn
from src.config import (
    ActivationFunction,
    DatasetEnum,
    LossType,
    LRSchedule,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    TrainingConfig,
)
from src.utils.data.data import DownloadableDataset
from src.utils.data.jax_dataloader import JAXDataLoader
from src.utils.models.registry import ModelRegistry
from src.utils.optimizers import optimizer as make_optimizer
from src.utils.train import create_train_state, initialize_model, train_step

NQ = 2000  # query/test points, disjoint from train


def parse_spec(s):
    "'3:16,8' -> ('ResNet 3x[16,8]', RESNETMLP, [[16,8]]*3, RELU)"
    nblocks, dims = s.split(":")
    up, down = (int(d) for d in dims.split(","))
    return (
        f"ResNet {nblocks}x[{up},{down}]",
        ModelArchitecture.RESNETMLP,
        [[up, down]] * int(nblocks),
        ActivationFunction.RELU,
    )


def load_subset(ntr, dataset):
    "Dataset subset of ntr train + NQ query, standardized (RandomState(0) draw)."
    full = DownloadableDataset.load(
        dataset,
        directory=str(PROJECT_ROOT / f"experiments/datasets/{dataset.value}"),
        store_on_disk=False,
    )
    Xall = np.asarray(full.inputs, np.float64)
    Yall = np.asarray(full.targets).reshape(-1).astype(int)
    rng = np.random.RandomState(0)
    idx = rng.choice(Xall.shape[0], ntr + NQ, replace=False)
    Xs, Ys = Xall[idx], Yall[idx]
    Xs = (Xs - Xs.mean(0, keepdims=True)) / (Xs.std(0, keepdims=True) + 1e-6)
    return (
        jnp.asarray(Xs[:ntr]),
        jnp.asarray(Ys[:ntr]),
        jnp.asarray(Xs[ntr:]),
        jnp.asarray(Ys[ntr:]),
    )


def calibrate(name, arch, hidden, act, lr, epochs, X, Y, Xq, Yq, seed):
    N = int(X.shape[0])
    mc = ModelConfig(
        architecture=arch,
        input_dim=784,
        hidden_dim=hidden,
        output_dim=10,
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
    # mirror train_model()'s loop, but record per-epoch metrics locally
    model = ModelRegistry.get_model(model_config=mc, seed=seed)
    params = initialize_model(model, input_shape=model.input_dim, key=jax.random.PRNGKey(seed))
    state = create_train_state(model, params, opt)
    dl = JAXDataLoader(X, Y, batch_size=BATCH, shuffle=True, rng_key=jax.random.PRNGKey(seed))

    @jax.jit
    def full_metrics(p):
        def mean_loss(pp, xx, yy):
            return loss_fn(model.apply(pp, xx), yy)

        g = jax.grad(mean_loss)(p, X, Y)
        gnorm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(g)))
        tracc = jnp.mean(jnp.argmax(model.apply(p, X), -1) == Y)
        teacc = jnp.mean(jnp.argmax(model.apply(p, Xq), -1) == Yq)
        return gnorm, tracc, teacc

    hist = []
    for epoch in range(1, epochs + 1):
        for bx, by in dl:
            state, _, _ = train_step(state, bx, by, loss_fn)
        gnorm, tracc, teacc = full_metrics(state.params)
        hist.append(dict(epoch=epoch, tracc=float(tracc), teacc=float(teacc), gnorm=float(gnorm)))
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    ep {epoch:3d}  trAcc={float(tracc):.3f}  teAcc={float(teacc):.3f}  "
                f"|grad|={float(gnorm):.3f}",
                flush=True,
            )
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", type=int, nargs="+", default=None, help="indices into run_one.ARCHS")
    ap.add_argument("--specs", type=str, nargs="+", default=None,
                    help="custom ResNet widths, e.g. 3:16,8 = 3 blocks of [up=16, down=8]")
    ap.add_argument("--dataset", type=str, default="mnist",
                    choices=[d.value for d in DatasetEnum])
    ap.add_argument("--ntrs", type=int, nargs="+", default=[12000, 18000])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.02, 0.05])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="default: results_pbrf/calibrate_<dataset>.json")
    args = ap.parse_args()
    dataset = DatasetEnum(args.dataset)
    out = args.out or str(HERE / "results_pbrf" / f"calibrate_{dataset.value}.json")

    if args.archs is None and args.specs is None:
        args.archs = [1, 2]
    configs = [ARCHS[a][:4] for a in (args.archs or [])]
    configs += [parse_spec(s) for s in (args.specs or [])]

    results = []
    for ntr in args.ntrs:
        X, Y, Xq, Yq = load_subset(ntr, dataset)
        for name, arch, hidden, act in configs:
            for lr in args.lrs:
                print(f"\n== {name}  {dataset.value}  NTR={ntr}  lr={lr}  epochs={args.epochs}")
                hist = calibrate(name, arch, hidden, act, lr, args.epochs,
                                 X, Y, Xq, Yq, args.seed)
                results.append(dict(name=name, dataset=dataset.value, ntr=ntr, lr=lr,
                                    epochs=args.epochs, seed=args.seed,
                                    hidden=hidden, hist=hist))

    print("\n===== summary (final epoch) =====")
    print(f"{'arch':18} {'NTR':>6} {'lr':>5} {'trAcc':>6} {'teAcc':>6} {'|grad|':>7}  verdict")
    for r in results:
        f = r["hist"][-1]
        if f["tracc"] > 0.99:
            v = "MEMORIZED (bad)"
        elif 0.88 <= f["tracc"] <= 0.97:
            v = "capacity-limited (good)"
        else:
            v = "undertrained/other"
        print(f"{r['name']:18} {r['ntr']:6d} {r['lr']:5.2f} {f['tracc']:6.3f} "
              f"{f['teacc']:6.3f} {f['gnorm']:7.3f}  {v}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh)
    print(f"\nsaved -> {out}")
    print("pick: trAcc plateau ~0.94-0.96 with the smallest |grad| (MLP ref: 0.955 / 0.27).")


if __name__ == "__main__":
    main()
