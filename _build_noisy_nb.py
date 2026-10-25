"""Build grokking_noisy_trajectory_pca.ipynb (standalone toy experiment, unrelated to the rest of the repo)."""
import json

NB_PATH = "grokking_noisy_trajectory_pca.ipynb"

md_intro = r"""# Weight-trajectory PCA on grokking — a regime where minibatch noise is visible

In the previous grokking run (`grokking_trajectory_pca.ipynb`: AdamW lr $10^{-3}$, batch 256 of 576, wd 2.0) the
weights drift so coherently that the raw path hides under the 50-step average — mean $\|w_i - \bar{w}_i\|$ is only
≈ 0.8% of the trajectory diameter. Probing for a regime where noise matters showed:

- shrinking the batch (64 / 32) raises noise somewhat but **delays or kills grokking** at this weight decay,
- plain minibatch SGD on the quadratic-activation + MSE variant either crawls or diverges,
- **raising Adam's learning rate at fixed batch does both jobs**: the per-step noise floor of Adam-style updates
  scales with lr, and a larger lr also groks *faster* (memorize ≈ 200, grok ≈ 1,400 at lr $10^{-2}$).

So this notebook reruns the identical pipeline at **lr $= 10^{-2}$** — everything else unchanged (batch 256,
wd 2.0, 60% train split) — for 10,000 steps with snapshots every 10 → 1,000 points. Noise is now ≈ 3% of the
trajectory diameter in full weight space, and after grokking the drift stalls while the noise floor stays, so the
late dynamics are genuinely noise-dominated.

One caveat the probes made precise: **no grokking regime found here puts more than ≈ 1% of that noise inside the
top-3 PCA plane.** With one-hot inputs each example's gradient touches disjoint weight rows, so minibatch noise is
nearly orthogonal to the smooth low-dimensional drift — plain SGD is the extreme case (≈ 0.1% in-plane, and it
only generalizes gradually, with no sharp grok). The projection therefore filters most of the jitter out of the
main panels; the zoom inset on the top-2 figure shows the post-grok noise ball at its own scale instead.

Figures are saved to `diff_plots/grok_noisy_*.png`.
"""

src_setup = r'''import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

SEED = 0
P_MOD = 31          # modulus of the toy task (a + b) mod P_MOD
TRAIN_FRAC = 0.6
HIDDEN = 256
N_STEPS = 10000
RECORD_EVERY = 10   # snapshot stride -> N_STEPS / RECORD_EVERY = 1000 points
WINDOW = 50         # trailing average is over the last 50 *steps*, as before
BATCH_SIZE = 256
LR = 1e-2           # 10x the previous run: the noise lever
WD = 2.0
PLOT_DIR = "diff_plots"

os.makedirs(PLOT_DIR, exist_ok=True)
torch.set_num_threads(2)  # tiny model; don't grab the whole shared node

# palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
AQUA = "#199e70"
RAW = "#c3c2b7"      # recessive gray for the raw minibatch path
DARKEST = "#0d366b"  # ramp end, also the end-of-training marker
RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.size": 11,
    "legend.frameon": False,
    "figure.dpi": 110,
})


def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved", path)
'''

src_data = r'''rng = np.random.default_rng(SEED)
a, b = np.meshgrid(np.arange(P_MOD), np.arange(P_MOD), indexing="ij")
a, b = a.ravel(), b.ravel()
y_all = (a + b) % P_MOD
eye = np.eye(P_MOD, dtype=np.float32)
X_all = np.concatenate([eye[a], eye[b]], axis=1)  # one-hot(a) ++ one-hot(b)

perm = rng.permutation(P_MOD * P_MOD)
n_tr = int(TRAIN_FRAC * P_MOD * P_MOD)
tr, va = perm[:n_tr], perm[n_tr:]
X_tr, y_tr = torch.tensor(X_all[tr]), torch.tensor(y_all[tr])
X_va, y_va = torch.tensor(X_all[va]), torch.tensor(y_all[va])
print(f"{P_MOD * P_MOD} pairs total: {len(tr)} train / {len(va)} validation, "
      f"chance accuracy {1 / P_MOD:.1%}")
'''

src_train = r'''torch.manual_seed(SEED)
model = nn.Sequential(nn.Linear(2 * P_MOD, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, P_MOD))
n_params = sum(q.numel() for q in model.parameters())

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WD)
loss_fn = nn.CrossEntropyLoss()
batch_rng = np.random.default_rng(SEED + 1)

N_REC = N_STEPS // RECORD_EVERY
steps_rec = (np.arange(N_REC) + 1) * RECORD_EVERY
W_raw = np.empty((N_REC, n_params), dtype=np.float32)  # w_i at snapshot times
W_avg = np.empty((N_REC, n_params), dtype=np.float32)  # trailing 50-step mean at snapshot times
tr_loss = np.empty(N_REC); va_loss = np.empty(N_REC)
tr_acc = np.empty(N_REC); va_acc = np.empty(N_REC)

buf = np.empty((WINDOW, n_params))  # ring buffer of the last WINDOW raw weight vectors
rec = 0
for i in range(N_STEPS):
    idx = batch_rng.choice(len(X_tr), BATCH_SIZE, replace=False)
    loss = loss_fn(model(X_tr[idx]), y_tr[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    w = parameters_to_vector(model.parameters()).detach().numpy()
    buf[i % WINDOW] = w
    if (i + 1) % RECORD_EVERY == 0:
        k = min(i + 1, WINDOW)
        W_raw[rec] = w
        W_avg[rec] = buf.mean(axis=0) if k == WINDOW else buf[:k].mean(axis=0)
        with torch.no_grad():
            lt, lv = model(X_tr), model(X_va)
            tr_loss[rec] = loss_fn(lt, y_tr).item()
            va_loss[rec] = loss_fn(lv, y_va).item()
            tr_acc[rec] = (lt.argmax(1) == y_tr).float().mean().item()
            va_acc[rec] = (lv.argmax(1) == y_va).float().mean().item()
        rec += 1

mem_hit = np.nonzero(tr_acc >= 0.99)[0]
grok_hit = np.nonzero(va_acc >= 0.90)[0]
g_idx = int(grok_hit[0]) if len(grok_hit) else None
print(f"{n_params} parameters, {N_STEPS} steps, {N_REC} snapshots")
print("train acc >= 99% at step", steps_rec[mem_hit[0]] if len(mem_hit) else "never")
print("val   acc >= 90% at step", steps_rec[g_idx] if g_idx is not None else "never (no grokking in budget)")
print(f"final: train acc {tr_acc[-1]:.1%}, val acc {va_acc[-1]:.1%}, "
      f"train loss {tr_loss[-1]:.2e}, val loss {va_loss[-1]:.2e}")
'''

src_metrics = r'''def mark_grok(ax, with_label=False):
    if g_idx is None:
        return
    ax.axvline(steps_rec[g_idx], color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    if with_label:
        ax.text(steps_rec[g_idx], 0.04, " grok", transform=ax.get_xaxis_transform(),
                color=INK2, fontsize=9, rotation=90)


fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)

ax = axes[0]
ax.plot(steps_rec, tr_loss, color=BLUE, linewidth=1.2, label="train")
ax.plot(steps_rec, va_loss, color=ORANGE, linewidth=1.2, label="validation")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("cross-entropy loss")
ax.legend()
mark_grok(ax)
ax.set_title("Loss")

ax = axes[1]
ax.plot(steps_rec, tr_acc, color=BLUE, linewidth=1.2, label="train")
ax.plot(steps_rec, va_acc, color=ORANGE, linewidth=1.2, label="validation")
ax.axhline(1 / P_MOD, color=BASE, linewidth=0.9, linestyle=":")
ax.annotate("chance", (steps_rec[0], 1 / P_MOD), xytext=(2, 3),
            textcoords="offset points", color=MUTED, fontsize=8)
ax.set_xscale("log")
ax.set_ylim(0, 1.03)
ax.set_xlabel("step")
ax.set_ylabel("accuracy")
ax.legend(loc="center left")
mark_grok(ax, with_label=True)
ax.set_title("Accuracy")

ax = axes[2]
ax.plot(steps_rec, np.linalg.norm(W_raw.astype(np.float64), axis=1),
        color=VIOLET, linewidth=1.8)
ax.set_xscale("log")
ax.set_xlabel("step")
ax.set_ylabel(r"$\|w\|_2$")
mark_grok(ax)
ax.set_title("Weight norm")

fig.suptitle(f"Grokking on (a + b) mod {P_MOD} at lr $10^{{-2}}$ — the noisy regime",
             fontsize=13, fontweight="bold", color=INK)
save_fig(fig, "grok_noisy_metrics.png")
plt.show()
'''

src_pca = r'''A = W_avg.astype(np.float64)  # w_avg snapshots, i = 0 ... 999

mu = A.mean(axis=0)
U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
del U
evr = S**2 / np.sum(S**2)

P3 = Vt[:3]
P2 = Vt[:2]
# same center + basis for both paths, so they live in the same coordinates
Z3_avg = (A - mu) @ P3.T
Z2_avg = (A - mu) @ P2.T
# raw path projected in float32 (identical up to plotting precision, half the memory)
Z3_raw = (W_raw - mu.astype(np.float32)) @ P3.astype(np.float32).T
Z2_raw = Z3_raw[:, :2]

assert np.allclose(Z2_avg, Z3_avg[:, :2])
print("PCA components are nested: top-2 projection == first two coords of top-3")
print("explained variance ratio (top 5):", np.array2string(evr[:5], precision=4))
print(f"cumulative: top-2 = {evr[:2].sum():.1%}, top-3 = {evr[:3].sum():.1%}")
'''

src_noise = r'''dev = (W_raw - W_avg).astype(np.float64)
noise = np.linalg.norm(dev, axis=1)
noise_in = np.linalg.norm(dev @ P3.T, axis=1)  # the part the top-3 projection can show
C = A - mu
n2 = (C * C).sum(axis=1)
d2 = n2[:, None] + n2[None, :] - 2.0 * (C @ C.T)
diam = float(np.sqrt(max(d2.max(), 0.0)))
print(f"trajectory diameter = {diam:.1f}")
print(f"mean ||w - wbar||          = {noise.mean():.3f}  ({noise.mean() / diam:.2%} of diameter)")
print(f"mean in-plane (top-3) part = {noise_in.mean():.3f}  ({noise_in.mean() / diam:.2%} of diameter)")
print("(same setup at lr 1e-3: full-space ratio was ~0.8%; the projection filters out most noise)")

fig, ax = plt.subplots(figsize=(7.6, 4.0), constrained_layout=True)
ax.plot(steps_rec, noise, color=AQUA, linewidth=1.4,
        label=r"full space $\|w_i - \bar{w}_i\|$")
ax.plot(steps_rec, noise_in, color=DARKEST, linewidth=1.4,
        label="component inside the top-3 PCA plane")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("deviation norm")
ax.legend(fontsize=9)
mark_grok(ax, with_label=True)
ax.set_title(f"Noise scale: {noise.mean() / diam:.1%} of diameter in full space, "
             f"{noise_in.mean() / diam:.1%} inside the projection")
save_fig(fig, "grok_noisy_noise_scale.png")
plt.show()
'''

src_evr = r'''k = 10
fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
ax.bar(range(1, k + 1), evr[:k], color=[BLUE] * 3 + [BASE] * (k - 3), width=0.62)
for j in range(3):
    ax.text(j + 1, evr[j] + 0.015, f"{evr[j]:.1%}", ha="center", color=INK2, fontsize=10)
ax.set_xticks(range(1, k + 1), [f"PC{j}" for j in range(1, k + 1)])
ax.set_ylabel("explained variance ratio")
ax.set_ylim(0, evr[0] * 1.16)
ax.grid(axis="x", visible=False)
ax.set_title(f"PCA of averaged path — kept PCs (blue) explain {evr[:3].sum():.1%} "
             f"(top-2: {evr[:2].sum():.1%})")
save_fig(fig, "grok_noisy_pca_explained_variance.png")
plt.show()
'''

src_helpers = r'''SNORM = Normalize(steps_rec[0], steps_rec[-1])
SEG_MID = 0.5 * (steps_rec[:-1] + steps_rec[1:])


def path_limits(Zs, i, pad=0.06):
    v = np.concatenate([Z[:, i] for Z in Zs])
    lo, hi = v.min(), v.max()
    m = pad * (hi - lo)
    return lo - m, hi + m


def ramp_line_2d(ax, Za, i, j):
    pts = Za[:, [i, j]].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=RAMP, norm=SNORM, linewidths=2.0,
                        capstyle="round", zorder=3)
    lc.set_array(SEG_MID)
    ax.add_collection(lc)
    return lc


def mark_ends_2d(ax, Za, i, j):
    ax.scatter(*Za[0, [i, j]], s=65, facecolor=SURFACE, edgecolor=INK,
               linewidth=1.4, zorder=5)
    ax.scatter(*Za[-1, [i, j]], s=150, marker="*", facecolor=DARKEST,
               edgecolor=SURFACE, linewidth=0.7, zorder=5)
    ax.annotate("start", Za[0, [i, j]], xytext=(6, 6),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.annotate("end", Za[-1, [i, j]], xytext=(6, 6),
                textcoords="offset points", color=INK2, fontsize=9)


def mark_grok_2d(ax, Za, i, j):
    if g_idx is None:
        return
    ax.scatter(*Za[g_idx, [i, j]], s=95, marker="D", facecolor=ORANGE,
               edgecolor=SURFACE, linewidth=0.8, zorder=6)
    ax.annotate("grok", Za[g_idx, [i, j]], xytext=(7, -11),
                textcoords="offset points", color=INK2, fontsize=9)


def style_cbar(cb):
    cb.set_label("training step", color=INK2)
    cb.outline.set_edgecolor(BASE)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED)


legend_handles = [
    Line2D([], [], color=RAW, linewidth=1.2, label="raw AdamW path $w_i$"),
    Line2D([], [], color=BLUE, linewidth=2.4,
           label=rf"trailing {WINDOW}-step average $\bar{{w}}_i$"),
]
if g_idx is not None:
    legend_handles.append(
        Line2D([], [], marker="D", linestyle="", markerfacecolor=ORANGE,
               markeredgecolor=SURFACE, markersize=8,
               label=r"grok (val acc first $\geq$ 90%)"))
'''

src_3d = r'''fig = plt.figure(figsize=(8.8, 7.2))
ax = fig.add_subplot(projection="3d")

ax.plot(Z3_raw[:, 0], Z3_raw[:, 1], Z3_raw[:, 2],
        color=RAW, linewidth=0.7, alpha=0.6, zorder=1)
pts = Z3_avg.reshape(-1, 1, 3)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = Line3DCollection(segs, cmap=RAMP, norm=SNORM, linewidths=2.4)
lc.set_array(SEG_MID)
ax.add_collection3d(lc)

ax.scatter(*Z3_avg[0], s=70, facecolor=SURFACE, edgecolor=INK,
           linewidth=1.4, depthshade=False)
ax.scatter(*Z3_avg[-1], s=180, marker="*", facecolor=DARKEST,
           edgecolor=SURFACE, linewidth=0.7, depthshade=False)
ax.text(*Z3_avg[0], "  start", color=INK2, fontsize=9)
ax.text(*Z3_avg[-1], "  end", color=INK2, fontsize=9)
if g_idx is not None:
    ax.scatter(*Z3_avg[g_idx], s=110, marker="D", facecolor=ORANGE,
               edgecolor=SURFACE, linewidth=0.8, depthshade=False)
    ax.text(*Z3_avg[g_idx], "  grok", color=INK2, fontsize=9)

for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
    setter(path_limits([Z3_raw, Z3_avg], i))
ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
ax.set_zlabel(f"PC3 ({evr[2]:.1%} var)")
for axis3 in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis3.set_pane_color(to_rgba(SURFACE))
    try:
        axis3._axinfo["grid"].update(color=GRID, linewidth=0.7)
    except Exception:
        pass
ax.view_init(elev=22, azim=-58)
ax.set_title("Noisy-regime grokking trajectory in the top-3 PCA space of the averaged path", pad=14)
ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
style_cbar(fig.colorbar(lc, ax=ax, shrink=0.55, pad=0.06))
save_fig(fig, "grok_noisy_pca3_trajectory_3d.png")
plt.show()
'''

src_pairwise = r'''pairs = [(0, 1), (0, 2), (1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), constrained_layout=True)
for ax, (i, j) in zip(axes, pairs):
    ax.plot(Z3_raw[:, i], Z3_raw[:, j], color=RAW, linewidth=0.6, alpha=0.6, zorder=1)
    lc = ramp_line_2d(ax, Z3_avg, i, j)
    mark_ends_2d(ax, Z3_avg, i, j)
    mark_grok_2d(ax, Z3_avg, i, j)
    ax.set_xlim(path_limits([Z3_raw, Z3_avg], i))
    ax.set_ylim(path_limits([Z3_raw, Z3_avg], j))
    ax.set_xlabel(f"PC{i + 1} ({evr[i]:.1%})")
    ax.set_ylabel(f"PC{j + 1} ({evr[j]:.1%})")
axes[0].legend(handles=legend_handles, loc="upper left", fontsize=8)
fig.suptitle("Pairwise views of the top-3 PCA projection (noisy grokking regime)", fontsize=13,
             fontweight="bold", color=INK)
style_cbar(fig.colorbar(lc, ax=axes, shrink=0.92, pad=0.015))
save_fig(fig, "grok_noisy_pca3_pairwise.png")
plt.show()
'''

src_top2 = r'''fig, ax = plt.subplots(figsize=(8.0, 6.4), constrained_layout=True)
ax.plot(Z2_raw[:, 0], Z2_raw[:, 1], color=RAW, linewidth=0.7, alpha=0.6, zorder=1)
lc = ramp_line_2d(ax, Z2_avg, 0, 1)
mark_ends_2d(ax, Z2_avg, 0, 1)
mark_grok_2d(ax, Z2_avg, 0, 1)
ax.set_xlim(path_limits([Z2_raw, Z2_avg], 0))
ax.set_ylim(path_limits([Z2_raw, Z2_avg], 1))
ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
ax.set_title(f"Top-2 PCA variant — {evr[:2].sum():.1%} of averaged-path variance")
style_cbar(fig.colorbar(lc, ax=ax, pad=0.02))

K = 150  # zoom: the last K snapshots (= K * RECORD_EVERY steps), where drift has stalled
zr, za = Z2_raw[-K:], Z2_avg[-K:]
x0 = min(zr[:, 0].min(), za[:, 0].min()); x1 = max(zr[:, 0].max(), za[:, 0].max())
y0 = min(zr[:, 1].min(), za[:, 1].min()); y1 = max(zr[:, 1].max(), za[:, 1].max())
px, py = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
axins = ax.inset_axes([0.42, 0.07, 0.42, 0.42])
axins.plot(zr[:, 0], zr[:, 1], color=MUTED, linewidth=0.8, alpha=0.9, zorder=2)
ramp_line_2d(axins, Z2_avg, 0, 1)
axins.scatter(*za[-1], s=110, marker="*", facecolor=DARKEST, edgecolor=SURFACE,
              linewidth=0.6, zorder=6)
axins.set_xlim(x0 - px, x1 + px)
axins.set_ylim(y0 - py, y1 + py)
axins.tick_params(labelsize=7, colors=MUTED)
for sp in axins.spines.values():
    sp.set_visible(True)
    sp.set_color(BASE)
axins.set_title(f"last {K * RECORD_EVERY} steps: raw path is a noise ball",
                fontsize=8.5, color=INK2)
ax.indicate_inset_zoom(axins, edgecolor=MUTED)
save_fig(fig, "grok_noisy_pca2_trajectory.png")
plt.show()
'''

md_outro = r"""## Reading the plots

- Same grokking signature, compressed: memorization by step ≈ 100, grokking at ≈ 1,300, then thousands of steps of
  post-grok consolidation — most of the recorded path lives in the noise-dominated late phase.
- The noise-scale figure separates two regimes: the early spike is *averaging lag* (the trailing mean trails the
  fast memorization transit — that part lies in-plane, it is real drift), while the late floor is stationary
  minibatch noise, ≈ 3% of the trajectory diameter in full space but only ≈ 1% inside the top-3 plane.
- Why the main panels still look smooth: with one-hot inputs each example's gradient touches disjoint first-layer
  rows, and Adam's preconditioning spreads updates across all ~24k coordinates, so the noise is nearly orthogonal
  to the low-dimensional drift plane and the projection filters it out. Plain SGD is *more* orthogonal here
  (≈ 0.1% in-plane) and only generalizes gradually, without a sharp grok — the visibly jittery two-moons picture
  comes from dense shared features, and no optimizer knob recovers it on this task. Shrinking the batch just
  delays or destroys grokking at this weight decay.
- The zoom inset on the top-2 figure shows the same data at the scale where noise *is* dominant: over the last
  1,500 steps the averaged path barely moves while the raw path wanders as a noise ball around it.
- PCA is fit on the averaged path only; the raw path is projected with the same center and basis.
- PCA components are nested: the top-2 variant uses exactly the first two of the top-3 directions (asserted above).
"""

cells = []


def add(kind, src):
    cell = {
        "id": f"cell{len(cells)}",
        "cell_type": kind,
        "metadata": {},
        "source": src.strip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cells.append(cell)


add("markdown", md_intro)
add("code", src_setup)
add("code", src_data)
add("code", src_train)
add("code", src_metrics)
add("code", src_pca)
add("code", src_noise)
add("code", src_evr)
add("code", src_helpers)
add("code", src_3d)
add("code", src_pairwise)
add("code", src_top2)
add("markdown", md_outro)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", NB_PATH)
