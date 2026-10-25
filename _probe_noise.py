"""One-off probe: grokking regimes where minibatch noise is visible in weight space.
For each config: does it grok, and how big is mean ||w_i - wbar_i|| relative to the
diameter of the averaged trajectory? Mirrors the notebook RNG structure (split rng(seed),
torch.manual_seed(seed), batch rng(seed+1), choice without replacement) so winners transfer 1:1."""
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
DEVICE = os.environ.get("PROBE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
WINDOW = 50
P = 31


class Square(nn.Module):
    def forward(self, x):
        return x * x


def run(name, act, loss_kind, optname, h, frac, lr, wd, batch, steps,
        seed=0, eval_every=250, snap_every=200, post_grok=2000, mom=0.0):
    rng = np.random.default_rng(seed)
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    a, b = a.ravel(), b.ravel()
    y_all = (a + b) % P
    eye = np.eye(P, dtype=np.float32)
    X_all = np.concatenate([eye[a], eye[b]], axis=1)
    perm = rng.permutation(P * P)
    n_tr = int(frac * P * P)
    tr, va = perm[:n_tr], perm[n_tr:]
    X_tr = torch.tensor(X_all[tr], device=DEVICE)
    y_tr = torch.tensor(y_all[tr], device=DEVICE)
    X_va = torch.tensor(X_all[va], device=DEVICE)
    y_va = torch.tensor(y_all[va], device=DEVICE)
    T_tr = torch.eye(P, device=DEVICE)[y_tr]

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2 * P, h), Square() if act == "quad" else nn.ReLU(),
                          nn.Linear(h, P)).to(DEVICE)
    D = sum(q.numel() for q in model.parameters())
    if optname == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=wd)
    else:  # plain minibatch SGD; coupled L2 so per-step shrink = lr * wd
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    brng = np.random.default_rng(seed + 1)

    buf = torch.empty(WINDOW, D, device=DEVICE)
    snaps_raw, snaps_avg = [], []
    mem = grok = None
    vacc = 0.0
    stop_at = steps
    for i in range(steps):
        idx = brng.choice(n_tr, batch, replace=False)
        out = model(X_tr[idx])
        loss = ce(out, y_tr[idx]) if loss_kind == "ce" else mse(out, T_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w = torch.nn.utils.parameters_to_vector(model.parameters())
        if not torch.isfinite(w).all():
            print(f"{name}: DIVERGED at step {i + 1}", flush=True)
            return
        buf[i % WINDOW] = w
        if (i + 1) % snap_every == 0:
            k = min(i + 1, WINDOW)
            snaps_raw.append(w.cpu())
            snaps_avg.append(buf[:k].mean(0).cpu())
        if (i + 1) % eval_every == 0:
            with torch.no_grad():
                ta = (model(X_tr).argmax(1) == y_tr).float().mean().item()
                vacc = (model(X_va).argmax(1) == y_va).float().mean().item()
            if mem is None and ta >= 0.99:
                mem = i + 1
            if grok is None and vacc >= 0.90:
                grok = i + 1
                stop_at = min(steps, i + 1 + post_grok)  # keep a post-grok tail for noise stats
        if i + 1 >= stop_at:
            break

    Wr, Wa = torch.stack(snaps_raw).double(), torch.stack(snaps_avg).double()
    dev = Wr - Wa
    noise = dev.norm(dim=1).mean().item()
    diam = torch.cdist(Wa, Wa).max().item()
    V3 = torch.linalg.svd(Wa - Wa.mean(0), full_matrices=False).Vh[:3]
    inplane = (dev @ V3.T).norm(dim=1).mean().item()  # what the PCA plots can actually show
    print(f"{name}: mem@{mem}, grok@{grok}, last val {vacc:.1%} | "
          f"noise {noise:.3f} ({noise / diam:.2%} of diam {diam:.1f}), "
          f"in-plane {inplane:.3f} ({inplane / diam:.2%})", flush=True)


t0 = time.time()
print("device:", DEVICE, flush=True)
# round 4: raise Adam lr further and shrink width (less room for noise to hide orthogonally)
run("H1 adamw h256 lr1.5e-2 b256", "relu", "ce", "adamw", 256, 0.6, 1.5e-2, 2.0, 256, 20000, snap_every=100, post_grok=3000)
run("H2 adamw h256 lr2e-2   b256", "relu", "ce", "adamw", 256, 0.6, 2e-2, 2.0, 256, 20000, snap_every=100, post_grok=3000)
run("H3 adamw h256 lr3e-2   b256", "relu", "ce", "adamw", 256, 0.6, 3e-2, 2.0, 256, 20000, snap_every=100, post_grok=3000)
run("H4 adamw h128 lr1e-2   b256", "relu", "ce", "adamw", 128, 0.6, 1e-2, 2.0, 256, 20000, snap_every=100, post_grok=3000)
run("H5 adamw h128 lr2e-2   b256", "relu", "ce", "adamw", 128, 0.6, 2e-2, 2.0, 256, 20000, snap_every=100, post_grok=3000)
run("H6 adamw h64  lr1e-2   b256", "relu", "ce", "adamw", 64, 0.6, 1e-2, 2.0, 256, 20000, snap_every=100, post_grok=3000)
print(f"probe total {time.time() - t0:.0f}s", flush=True)
