#!/usr/bin/env python
"""Fan out the PBRF-MNIST sweep over GPUs, then summarize mean +/- std.

Runs every (arch_idx, seed) as a separate run_one.py subprocess, one per GPU
(CUDA_VISIBLE_DEVICES), at most len(GPUS) at a time. Each writes a JSON to
--outdir; at the end we read them all back and print + save a mean/std summary.

  python sweep.py                       # run full sweep on GPUs 0..7, then summarize
  python sweep.py --gpus 0 1 2 3        # restrict to a GPU subset
  python sweep.py --seeds 0 1 2         # fewer seeds
  python sweep.py --aggregate-only      # just re-summarize existing JSONs
"""
import os
import sys
import json
import queue
import argparse
import threading
import subprocess
from glob import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_ONE = HERE / "run_one.py"
METHODS = ["gnh", "kfac", "ekfac", "shampoo", "eshampoo", "identity"]
METRICS = ["pbrf_P", "pbrf_S", "warm_P", "warm_S", "ratio"]


def launch(jobs, gpus, outdir, logdir):
    "Run jobs (arch_idx, seed) across `gpus`, <=len(gpus) concurrent."
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    lock = threading.Lock()
    state = {"done": 0, "fail": 0}
    total = len(jobs)

    def worker(arch_idx, seed):
        g = gpu_q.get()  # blocks until a GPU frees up -> caps concurrency at len(gpus)
        try:
            out = outdir / f"arch{arch_idx}_seed{seed}.json"
            log = logdir / f"arch{arch_idx}_seed{seed}.log"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            with open(log, "w") as lf:
                rc = subprocess.run(
                    [sys.executable, str(RUN_ONE),
                     "--arch_idx", str(arch_idx), "--seed", str(seed), "--out", str(out)],
                    env=env, stdout=lf, stderr=subprocess.STDOUT,
                ).returncode
            with lock:
                if rc == 0:
                    state["done"] += 1
                    tag = "ok"
                else:
                    state["fail"] += 1
                    tag = f"FAIL rc={rc} (see {log})"
                print(f"[{state['done']+state['fail']}/{total}] gpu{g} arch{arch_idx} seed{seed} {tag}",
                      flush=True)
        finally:
            gpu_q.put(g)

    threads = [threading.Thread(target=worker, args=(a, s)) for a, s in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return state["fail"]


def aggregate(outdir):
    "Read every JSON, group by arch, print + save mean+/-std of the correlations."
    files = sorted(glob(str(outdir / "*.json")))
    runs = []
    for f in files:
        if Path(f).name == "summary.json":
            continue
        with open(f) as fh:
            runs.append(json.load(fh))
    if not runs:
        print("no run JSONs found in", outdir)
        return

    by_arch = {}
    for r in runs:
        by_arch.setdefault((r["arch_idx"], r["name"]), []).append(r)

    summary = {}
    for (arch_idx, name), rs in sorted(by_arch.items()):
        seeds = sorted(r["seed"] for r in rs)
        print(f"\n### [{arch_idx}] {name}  (mean+/-std over {len(rs)} seeds: {seeds})")
        print(f"    trAcc={np.mean([r['tracc'] for r in rs]):.3f}+/-{np.std([r['tracc'] for r in rs]):.3f}"
              f"  teAcc={np.mean([r['teacc'] for r in rs]):.3f}+/-{np.std([r['teacc'] for r in rs]):.3f}")
        header = "  method   " + "  ".join(f"{m:>13}" for m in METRICS)
        print(header)
        arch_sum = {}
        for m in METHODS:
            cells, entry = [], {}
            for metric in METRICS:
                vals = np.array([r["corr"][m][metric] for r in rs], float)
                mu, sd = float(vals.mean()), float(vals.std())
                entry[metric] = [mu, sd]
                cells.append(f"{mu:+.3f}+/-{sd:.3f}")
            arch_sum[m] = entry
            print(f"  {m:8} " + "  ".join(f"{c:>13}" for c in cells))
        summary[name] = dict(arch_idx=arch_idx, n_seeds=len(rs), seeds=seeds, methods=arch_sum)

    out = outdir / "summary.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsaved summary ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--archs", type=int, nargs="+", default=[0])
    ap.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--outdir", type=str, default=str(HERE / "results_pbrf"))
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    logdir = outdir / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        jobs = [(a, s) for a in args.archs for s in args.seeds]
        print(f"launching {len(jobs)} jobs on GPUs {args.gpus} "
              f"(archs={args.archs}, seeds={args.seeds})")
        nfail = launch(jobs, args.gpus, outdir, logdir)
        if nfail:
            print(f"\n!! {nfail} job(s) failed; summary below covers only the successful JSONs")

    aggregate(outdir)


if __name__ == "__main__":
    main()
