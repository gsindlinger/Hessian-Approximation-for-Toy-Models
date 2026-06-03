"""Plot layer for the runs.db explorer.

All matplotlib figure functions. Pure presentation: data is loaded/sliced/
computed by `hessian_data` (imported as `D`) and handed in here to be drawn.
Dependency direction is one-way: plots → data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import LogNorm

import hessian_data as D

plt.rcParams.update({"font.size": 11})

# ── Style ─────────────────────────────────────────────────────────────
COLORS = {
    "kfac": "#1f77b4", "ekfac": "#ff7f0e", "gnh": "#d62728", "fim": "#2ca02c",
    "block_fim": "#9467bd", "block_hessian": "#8c564b", "shampoo": "#e377c2",
    "eshampoo": "#7f7f7f", "identity": "#17becf", "eidentity": "#9edae5",
    "exact": "#bcbd22", "mac": "#393b79", "emac": "#637939",
}
LABELS = {
    "kfac": "K-FAC", "ekfac": "EK-FAC", "gnh": "GNH", "fim": "FIM",
    "block_fim": "Block FIM", "block_hessian": "Block Hessian", "shampoo": "Shampoo",
    "eshampoo": "E-Shampoo", "identity": "Identity", "eidentity": "E-Identity",
    "exact": "Exact", "mac": "MAC", "emac": "E-MAC",
}
CATEGORY_DISPLAY = {"matrix": "Matrix", "hvp": "HVP", "ihvp": "IHVP", "round_trip": "Round-trip"}


def _format_metric_name(metric: str) -> str:
    return (
        metric.replace("_", " ").title()
        .replace("L2", r"$L_2$").replace("Hvp", "HVP").replace("Ihvp", "IHVP")
    )


def category_label(cat: tuple[str, str]) -> str:
    """Display label for a (computation_type, metric) pair."""
    ct, metric = cat
    return f"{CATEGORY_DISPLAY.get(ct, ct)}  {_format_metric_name(metric)}"


def _method_color(m: str) -> tuple:
    return COLORS.get(m, "#444444")


def _damping_color(d: float, *, vmin: float, vmax: float, cmap_name: str = "viridis") -> tuple:
    return colormaps[cmap_name](LogNorm(vmin=vmin, vmax=vmax)(d))


_LINESTYLES = ["-", "--", ":", "-."]


def _strategy_styles(strategies: list[str]) -> dict[str, str]:
    return {s: _LINESTYLES[i % len(_LINESTYLES)] for i, s in enumerate(sorted(strategies))}


# =====================================================================
# LDS sweeps
# =====================================================================
def _lds_lines_panel(
    ax, sub: pd.DataFrame, *, x_col, group_col, color_for, label_for,
    log_x: bool = False, show_band: bool = True,
    style_col: Optional[str] = "damping_strategy", style_map: Optional[dict] = None,
):
    """One panel: one line per group, x = `x_col`, y = lds_mean (with CI band)."""
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_axis_off()
        return []

    use_style = style_map is not None and len(style_map) > 1 and style_col in sub.columns
    for g, gdf in sub.groupby(group_col, sort=False):
        gdf = gdf.sort_values(x_col)
        color = color_for(g)
        inner_iter = gdf.groupby(style_col, sort=False) if use_style else [(None, gdf)]
        for s, sdf in inner_iter:
            sdf = sdf.sort_values(x_col)
            ls = style_map.get(s, "-") if use_style else "-"
            x = sdf[x_col].to_numpy()
            y = sdf["lds_mean"].to_numpy()
            label = f"{label_for(g)} · {s}" if use_style else label_for(g)
            ax.plot(x, y, color=color, lw=1.6, marker="o", ms=3.5, ls=ls, label=label)
            if show_band and "lds_ci_low" in sdf and "lds_ci_high" in sdf:
                lo, hi = sdf["lds_ci_low"].to_numpy(), sdf["lds_ci_high"].to_numpy()
                if np.isfinite(lo).any() and np.isfinite(hi).any():
                    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)

    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if log_x:
        ax.set_xscale("log")


def _x_tick_label(x_col: str, v) -> str:
    if x_col == "epoch":
        return str(int(v))
    return f"{v:.0e}" if v not in (0, 1) else f"{v:g}"


def _lds_bars_panel(
    ax, sub: pd.DataFrame, *, x_col, group_col, color_for, label_for,
    show_band: bool = True, **_ignored,  # absorb log_x / style_* (line-panel only)
):
    """One panel: grouped bars, x = `x_col` categories, y = lds_mean. CI shown as
    capped (I-shaped) error bars. Multiple rows per (group, x) are averaged."""
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_axis_off()
        return

    xs = sorted(sub[x_col].dropna().unique())
    groups = list(dict.fromkeys(sub[group_col].dropna().tolist()))
    xpos = np.arange(len(xs))
    n_groups = max(len(groups), 1)
    bw = 0.8 / n_groups

    for gi, g in enumerate(groups):
        gdf = sub[sub[group_col] == g]
        means, lo_err, hi_err = [], [], []
        for xv in xs:
            r = gdf[gdf[x_col] == xv]
            if r.empty:
                means.append(np.nan); lo_err.append(0.0); hi_err.append(0.0)
                continue
            m = r["lds_mean"].mean()
            means.append(m)
            lo = m - r["lds_ci_low"].mean() if "lds_ci_low" in r else np.nan
            hi = r["lds_ci_high"].mean() - m if "lds_ci_high" in r else np.nan
            lo_err.append(lo if np.isfinite(lo) and lo > 0 else 0.0)
            hi_err.append(hi if np.isfinite(hi) and hi > 0 else 0.0)
        offset = (gi - (n_groups - 1) / 2) * bw
        yerr = np.array([lo_err, hi_err]) if show_band else None
        ax.bar(xpos + offset, means, width=bw * 0.95, color=color_for(g),
               label=label_for(g), yerr=yerr, capsize=2.5,
               error_kw={"elinewidth": 0.8, "alpha": 0.7})

    ax.set_xticks(xpos)
    ax.set_xticklabels([_x_tick_label(x_col, v) for v in xs], rotation=35, ha="right")
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _lds_panel(kind: str):
    return _lds_bars_panel if kind == "bar" else _lds_lines_panel


def plot_lds_methodfix(df_sub, method, *, ax=None, show_legend=True, show_band=True, kind="line"):
    """Fix method, vary epoch (x) × damping (color/bar)."""
    sub = df_sub[df_sub["approximator"] == method]
    fig, ax = _ax_or_new(ax, (7.2, 4.4))
    dvals = sorted(sub["damping_value"].dropna().unique().tolist())
    dmin = max(min(dvals), 1e-12) if dvals else 1e-12
    dmax = max(dvals) if dvals else 1.0
    style_map = _strategy_styles(sub["damping_strategy"].dropna().unique().tolist())
    _lds_panel(kind)(
        ax, sub, x_col="epoch", group_col="damping_value",
        color_for=lambda d: _damping_color(d, vmin=dmin, vmax=dmax),
        label_for=lambda d: f"λ={d:.0e}" if d not in (0, 1) else f"λ={d:g}",
        log_x=False, show_band=show_band,
        style_map=style_map if len(style_map) > 1 else None,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("LDS mean")
    ax.set_title(LABELS.get(method, method), fontsize=11, fontweight="bold")
    if show_legend:
        ax.legend(loc="best", fontsize=8, ncol=2 if len(dvals) > 4 else 1, frameon=False)
    return fig


def plot_lds_dampingfix(df_sub, damping, *, strategy=None, ax=None, show_legend=True,
                        show_band=True, methods=None, kind="line"):
    """Fix damping, vary epoch (x) × method (color/bar)."""
    sub = df_sub[np.isclose(df_sub["damping_value"], damping)]
    if strategy is not None:
        sub = sub[sub["damping_strategy"] == strategy]
    if methods is not None:
        sub = sub[sub["approximator"].isin(methods)]
    methods_here = D.order_methods(sub["approximator"].dropna().unique().tolist())
    fig, ax = _ax_or_new(ax, (7.6, 4.6))
    style_map = _strategy_styles(sub["damping_strategy"].dropna().unique().tolist())
    _lds_panel(kind)(
        ax, sub, x_col="epoch", group_col="approximator",
        color_for=_method_color, label_for=lambda m: LABELS.get(m, m),
        log_x=False, show_band=show_band,
        style_map=style_map if len(style_map) > 1 else None,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("LDS mean")
    ax.set_title(f"λ = {damping:g}", fontsize=11, fontweight="bold")
    if show_legend and methods_here:
        ax.legend(loc="best", fontsize=8, ncol=2 if len(methods_here) > 6 else 1, frameon=False)
    return fig


def plot_lds_epochfix(df_sub, epoch, *, strategy=None, ax=None, show_legend=True,
                      show_band=True, methods=None, kind="line"):
    """Fix epoch, vary damping (x, log) × method (color/bar)."""
    sub = df_sub[df_sub["epoch"] == epoch]
    if strategy is not None:
        sub = sub[sub["damping_strategy"] == strategy]
    if methods is not None:
        sub = sub[sub["approximator"].isin(methods)]
    methods_here = D.order_methods(sub["approximator"].dropna().unique().tolist())
    fig, ax = _ax_or_new(ax, (7.6, 4.6))
    style_map = _strategy_styles(sub["damping_strategy"].dropna().unique().tolist())
    _lds_panel(kind)(
        ax, sub, x_col="damping_value", group_col="approximator",
        color_for=_method_color, label_for=lambda m: LABELS.get(m, m),
        log_x=True, show_band=show_band,
        style_map=style_map if len(style_map) > 1 else None,
    )
    ax.set_xlabel("damping λ")
    ax.set_ylabel("LDS mean")
    ax.set_title(f"epoch = {epoch}", fontsize=11, fontweight="bold")
    if show_legend and methods_here:
        ax.legend(loc="best", fontsize=8, ncol=2 if len(methods_here) > 6 else 1, frameon=False)
    return fig


def _ax_or_new(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        return fig, ax
    return ax.get_figure(), ax


def _lds_pivot(df_sub, method, strategy=None):
    sub = df_sub[df_sub["approximator"] == method]
    if strategy is not None:
        sub = sub[sub["damping_strategy"] == strategy]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.pivot_table(index="damping_value", columns="epoch", values="lds_mean", aggfunc="mean")
        .sort_index().sort_index(axis=1)
    )


def _draw_heatmap(ax, pivot, *, x_int=True, y_damping=True, vmin=-1.0, vmax=1.0,
                  annotate=True, aspect=None):
    """Shared RdBu_r imshow with annotations for damping×epoch / spearman grids."""
    if isinstance(pivot, pd.DataFrame) and pivot.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_axis_off()
        return None
    arr = pivot.to_numpy()
    im = ax.imshow(arr, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   **({"aspect": aspect} if aspect else {}))
    cols = pivot.columns
    idx = pivot.index
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_yticks(np.arange(arr.shape[0]))
    ax.set_xticklabels([str(int(x)) for x in cols] if x_int
                       else [f"{x:.0e}" if x != 0 else "0" for x in cols],
                       rotation=35, ha="right")
    ax.set_yticklabels([f"{y:.0e}" if y != 0 else "0" for y in idx] if y_damping
                       else [str(int(y)) for y in idx])
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.55 else "black")
    return im


def plot_lds_heatmap_for_method(df_sub, method, *, ax=None, vmin=-1.0, vmax=1.0, annotate=True):
    """LDS heatmap (damping × epoch) for one method, from a pre-sliced df_sub."""
    pivot = _lds_pivot(df_sub, method)
    fig, ax = _ax_or_new(ax, (
        max(5, 0.7 * pivot.shape[1] + 2) if not pivot.empty else 5,
        max(4, 0.7 * pivot.shape[0] + 2) if not pivot.empty else 4,
    ))
    im = _draw_heatmap(ax, pivot, x_int=True, y_damping=True, vmin=vmin, vmax=vmax,
                       annotate=annotate, aspect="auto")
    ax.set_xlabel("epoch")
    ax.set_ylabel("damping λ")
    ax.set_title(LABELS.get(method, method), fontsize=11, fontweight="bold")
    if ax.get_figure() is fig and im is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("LDS (mean)")
    return fig, pivot


def plot_lds_heatmap(df, *, model_id, method="exact", annotate=True):
    """Heatmap of LDS mean for one (model, method) over the damping × epoch grid."""
    sub = df[(df["model_id"] == model_id) & (df["approximator"] == method) & df["lds_mean"].notna()]
    if sub.empty:
        raise ValueError(f"No LDS rows for model_id={model_id!r} method={method!r}.")
    pivot = (
        sub.pivot_table(index="damping_value", columns="epoch", values="lds_mean", aggfunc="mean")
        .sort_index().sort_index(axis=1)
    )
    fig, ax = plt.subplots(
        figsize=(max(5, 0.7 * pivot.shape[1] + 2), max(4, 0.7 * pivot.shape[0] + 2)),
        constrained_layout=True,
    )
    im = _draw_heatmap(ax, pivot, x_int=True, y_damping=True, annotate=annotate)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("LDS (mean)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("damping value")
    ax.set_title(f"{model_id} — LDS (method={LABELS.get(method, method)})",
                 fontsize=11, fontweight="bold")
    return fig, pivot


# =====================================================================
# Metric bar plots
# =====================================================================
def plot_metrics_for(result_id, reference="exact", approxs=None, categories=None,
                     db_path=None):
    """One tall figure: one row per (computation_type, metric) bar plot, log-y.

    `categories` restricts (and orders) which (computation_type, metric) panels
    are drawn; None means all available.
    """
    db_path = db_path or D.DB_PATH
    with D.open_db(db_path) as con:
        df = D.load_metrics_df(con, result_id, reference=reference)
        meta = D.result_meta(con, result_id)
    run_id, model_id, epoch = meta["run_id"], meta["model_id"], meta["epoch"]

    if df.empty:
        raise ValueError(f"No metrics for reference={reference} on this result")
    df = df[df["approximator"] != reference].copy()

    if approxs is None:
        approxs = D.order_methods(df["approximator"].unique().tolist())
    else:
        approxs = [a for a in approxs if a in set(df["approximator"].unique())]

    rows = D.ordered_categories(df)
    if categories is not None:
        wanted = {tuple(c) for c in categories}
        rows = [r for r in rows if r in wanted]
        if not rows:
            raise ValueError("No matching (computation_type, metric) categories selected")

    x = np.arange(len(approxs))
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(max(10, len(approxs) * 1.1 + 2), 2.4 * len(rows) + 1.2),
        constrained_layout=True,
    )
    if len(rows) == 1:
        axes = [axes]

    for ax, (cat, metric) in zip(axes, rows):
        sub = df[(df["computation_type"] == cat) & (df["metric"] == metric)]
        lookup = dict(zip(sub["approximator"], sub["value"]))
        plot_vals = [v if (v := lookup.get(a)) is not None and v > 0 else np.nan for a in approxs]
        ax.bar(x, plot_vals, color=[COLORS.get(a, f"C{i}") for i, a in enumerate(approxs)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(a, a) for a in approxs], rotation=35, ha="right")
        if any(np.isfinite(v) for v in plot_vals):
            ax.set_yscale("log")
        ax.set_title(category_label((cat, metric)), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{model_id}  ({meta['num_parameters']} params)  Epoch {epoch}  "
        f"(vs {LABELS.get(reference, reference)})  run {run_id}",
        fontsize=12, fontweight="bold",
    )
    return fig


# =====================================================================
# Metric correlation scatter (one point per method)
# =====================================================================
def axis_label(axis) -> str:
    """Display label for a method-table axis: a (ct, metric) pair or LDS_AXIS."""
    if tuple(axis) == D.LDS_AXIS:
        return "LDS (mean)"
    return category_label(tuple(axis))


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(Pearson, Spearman) over paired samples; NaN if < 2 points or constant."""
    if len(x) < 2:
        return float("nan"), float("nan")
    with np.errstate(invalid="ignore"):
        pear = np.corrcoef(x, y)[0, 1]
        from scipy.stats import rankdata as _rk
        spear = np.corrcoef(_rk(x), _rk(y))[0, 1]
    return float(pear), float(spear)


# distinct marker per method (Fig 1b: shape = method)
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h", "8"]


def _method_markers(methods: list[str]) -> dict[str, str]:
    ordered = D.order_methods(list(dict.fromkeys(methods)))
    return {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(ordered)}


def plot_metric_correlation(table, x_axis, y_axis, *, point_methods=None,
                            point_damping=None, include=None,
                            log_x=True, log_y=True, annotate=True):
    """Scatter of two method-table axes with Pearson & Spearman over the points.

    Each row of `table` is one point. `point_methods` gives the method per row.

    Encoding:
      * single-result (point_damping=None): colour = method, one point each,
        optional per-point labels.
      * across-sweep (point_damping given): **shape = method, colour = damping λ**
        (viridis colorbar) — the Fig 1(b) encoding. A shape legend names methods.

    Pearson is on the plotted coordinates (log10 when an axis is log); Spearman
    is rank-based (scale-invariant). `include` keeps only those methods.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    x_axis, y_axis = tuple(x_axis), tuple(y_axis)
    m = np.array(list(table.index) if point_methods is None else list(point_methods))
    x = table[x_axis].to_numpy(float)
    y = table[y_axis].to_numpy(float)
    dmp = None if point_damping is None else np.asarray(point_damping, dtype=float)

    mask = ~(np.isnan(x) | np.isnan(y))
    if include is not None:
        mask &= np.isin(m, list(include))
    x, y, m = x[mask], y[mask], m[mask]
    if dmp is not None:
        dmp = dmp[mask]

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    if log_x and np.all(x > 0):
        ax.set_xscale("log")
        cx = np.log10(x)
    else:
        cx = x
    if log_y and np.all(y > 0):
        ax.set_yscale("log")
        cy = np.log10(y)
    else:
        cy = y

    present = D.order_methods(list(dict.fromkeys(m.tolist())))

    if dmp is None:
        # single-result: colour = method
        ax.scatter(x, y, s=60, c=[COLORS.get(mm, "#444444") for mm in m],
                   edgecolor="black", linewidth=0.4, zorder=3)
        if annotate:
            for xi, yi, mm in zip(x, y, m):
                ax.annotate(LABELS.get(mm, mm), (xi, yi), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
        handles = [Line2D([0], [0], marker="o", ls="", markersize=7,
                          markerfacecolor=COLORS.get(mm, "#444444"),
                          markeredgecolor="black", label=LABELS.get(mm, mm))
                   for mm in present]
        legend_title = None
    else:
        # across-sweep: shape = method, colour = damping λ
        markers = _method_markers(m.tolist())
        pos = dmp[dmp > 0]
        norm = LogNorm(vmin=float(pos.min()), vmax=float(pos.max())) if pos.size else None
        cmap = colormaps["viridis"]
        for mm in present:
            sel = m == mm
            ax.scatter(x[sel], y[sel], marker=markers[mm], s=42, c=dmp[sel],
                       cmap=cmap, norm=norm, edgecolor="black", linewidth=0.25,
                       alpha=0.9, zorder=3)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label("damping λ")
        handles = [Line2D([0], [0], marker=markers[mm], ls="", markersize=7,
                          markerfacecolor="0.4", markeredgecolor="black",
                          label=LABELS.get(mm, mm)) for mm in present]
        legend_title = "method"

    pear, spear = _corr(cx, cy)
    if handles:
        ax.legend(handles=handles, title=legend_title, fontsize=7,
                  ncol=2 if len(handles) > 7 else 1, frameon=False, loc="best")

    ax.set_xlabel(axis_label(x_axis))
    ax.set_ylabel(axis_label(y_axis))
    ax.grid(alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Pearson r = {pear:.3f}   ·   Spearman ρ = {spear:.3f}   (n = {len(x)})",
        fontsize=11, fontweight="bold",
    )
    return fig, {"pearson": pear, "spearman": spear, "n": int(len(x)),
                 "methods": present}


# =====================================================================
# Influence Spearman
# =====================================================================
def plot_influence_spearman(rho, *, methods=None, annotate=True, title=None):
    """Pairwise Spearman heatmap from a precomputed matrix `rho`
    (build with D.compute_influence_spearman_matrix)."""
    if methods is not None:
        rho = D.reorder_spearman(rho, methods)
    methods = list(rho.index)
    mat = rho.to_numpy()
    n = len(methods)
    n_query = int(rho.attrs.get("n_query", 0))
    aggregate = rho.attrs.get("aggregate", "mean")
    result_id = rho.attrs.get("result_id")

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 3), max(5, 0.7 * n + 2)),
                           constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [LABELS.get(m, m) for m in methods]
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    if annotate:
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman ρ")
    summary = (f"{n_query} query" + ("y" if n_query == 1 else "ies")
               + (" (1D cache)" if n_query == 1 else f", aggregated by {aggregate}"))
    ax.set_title(title or f"Influence Spearman pairwise — result #{result_id} ({summary})",
                 fontsize=11, fontweight="bold")
    return fig, rho


def _draw_corr_heatmap(ax, mat, *, sweep, annotate=True):
    xs = list(mat.index)
    n = len(xs)
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ([f"{x:.0e}" if x != 0 else "0" for x in xs] if sweep == "damping"
              else [str(int(x)) for x in xs])
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    if annotate:
        arr = mat.to_numpy()
        for i in range(n):
            for j in range(n):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.55 else "black")
    return im


def plot_influence_corr_across_axis(df, *, model_id, sweep, fix, method="exact",
                                    aggregate="mean", annotate=True, ax=None,
                                    with_colorbar=True, with_title=True,
                                    damping_strategy=None, pseudo_target_strategy=None):
    """Single-panel pairwise Spearman heatmap across the swept axis."""
    sweep_col = "damping_value" if sweep == "damping" else "epoch"
    fix_col = "epoch" if sweep == "damping" else "damping_value"
    mat, n_query = D.spearman_matrix_across_axis(
        df, model_id=model_id, sweep=sweep, fix=fix, method=method, aggregate=aggregate,
        damping_strategy=damping_strategy, pseudo_target_strategy=pseudo_target_strategy,
    )
    if ax is None:
        n = len(mat)
        fig, ax = plt.subplots(figsize=(max(5, 0.7 * n + 3), max(4.5, 0.7 * n + 2)),
                               constrained_layout=True)
    else:
        fig = ax.get_figure()
    im = _draw_corr_heatmap(ax, mat, sweep=sweep, annotate=annotate)
    ax.set_xlabel(sweep_col.replace("_", " "))
    ax.set_ylabel(sweep_col.replace("_", " "))
    if with_title:
        summary = (f"{n_query} query" + ("y" if n_query == 1 else "ies")
                   + (" (1D cache)" if n_query == 1 else f", aggregated by {aggregate}"))
        ax.set_title(
            f"{model_id} — influence Spearman across {sweep}\n"
            f"method={LABELS.get(method, method)}, fixed {fix_col}={fix} ({summary})",
            fontsize=11, fontweight="bold",
        )
    if with_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman ρ")
    return fig, mat


def plot_influence_corr_grid(df, *, model_id, sweep, fix_values, method="exact",
                             aggregate="mean", annotate=True, ncols=None,
                             damping_strategy=None, pseudo_target_strategy=None):
    """One figure, one panel per `fix_values` element; shared colorbar."""
    fix_col = "epoch" if sweep == "damping" else "damping_value"
    n = len(fix_values)
    ncols = ncols or min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.4),
                             constrained_layout=True, squeeze=False)
    axes_flat = axes.reshape(-1)

    last_im, skipped = None, []
    for ax, fv in zip(axes_flat, fix_values):
        try:
            mat, _ = D.spearman_matrix_across_axis(
                df, model_id=model_id, sweep=sweep, fix=fv, method=method,
                aggregate=aggregate, damping_strategy=damping_strategy,
                pseudo_target_strategy=pseudo_target_strategy,
            )
        except ValueError as e:
            skipped.append((fv, str(e).splitlines()[0]))
            ax.axis("off")
            ax.text(0.5, 0.5, f"{fix_col}={fv}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=10)
            continue
        last_im = _draw_corr_heatmap(ax, mat, sweep=sweep, annotate=annotate)
        ax.set_title(f"{fix_col}={fv}", fontsize=10)
    for ax in axes_flat[len(fix_values):]:
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02).set_label("Spearman ρ")
    fig.suptitle(
        f"{model_id} — influence Spearman across {sweep} "
        f"(method={LABELS.get(method, method)}, agg={aggregate})",
        fontsize=12, fontweight="bold",
    )
    return fig, skipped


# =====================================================================
# Factor eigenvalues
# =====================================================================
def _spectrum_panel(ax_line, ax_heat, layers, values_by_layer, *, title,
                    color_for_layer, log_x=False):
    if not values_by_layer:
        ax_line.text(0.5, 0.5, "no values", ha="center", va="center",
                     transform=ax_line.transAxes, color="gray")
        ax_line.set_axis_off()
        ax_heat.set_axis_off()
        ax_line.set_title(title, fontsize=10)
        return

    max_n = max(len(v) for v in values_by_layer.values())
    heat = np.full((len(layers), max_n), np.nan)
    for i, l in enumerate(layers):
        v = values_by_layer.get(l)
        if v is None or len(v) == 0:
            continue
        heat[i, : len(v)] = v
        # 1-based rank when log so the leading eigenvalue (rank 0) isn't at log(0)
        x = np.arange(1, len(v) + 1) if log_x else np.arange(len(v))
        ax_line.plot(x, v, label=l, color=color_for_layer(i), lw=1.4, alpha=0.9)

    ax_line.set_xlabel("rank (descending)")
    ax_line.set_ylabel("|eigenvalue|")
    ax_line.set_yscale("log")
    if log_x:
        ax_line.set_xscale("log")
    ax_line.legend(loc="upper right", fontsize=8, ncol=2 if len(values_by_layer) > 6 else 1)
    ax_line.grid(alpha=0.3, which="both")
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.set_title(title, fontsize=10)

    finite = heat[(heat > 0) & np.isfinite(heat)]
    norm = LogNorm(vmin=float(finite.min()), vmax=float(finite.max())) if finite.size else None
    im = ax_heat.imshow(heat, aspect="auto", cmap="viridis", norm=norm, interpolation="nearest")
    ax_heat.set_yticks(range(len(layers)))
    ax_heat.set_yticklabels(layers)
    ax_heat.set_xlabel("rank (descending)")
    ax_heat.set_title(f"{title} — heatmap (log)", fontsize=10)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)


def plot_factor_eigenvalues(factor_dir, layers=None, log_x=False):
    """Per-layer eigenvalue spectrum (positives + |negatives| in separate rows)."""
    factor_dir = Path(factor_dir)
    eigs_by_layer = D.load_factor_eigenvalues(factor_dir)
    if not eigs_by_layer:
        raise ValueError(f"No diagonal blocks in {factor_dir}")
    layers = list(eigs_by_layer) if layers is None else [l for l in layers if l in eigs_by_layer]

    pos_by_layer, neg_by_layer = {}, {}
    for l in layers:
        v = eigs_by_layer[l]
        pos_by_layer[l] = -np.sort(-v[v > 0])
        neg_by_layer[l] = -np.sort(-(-v[v < 0]))

    has_neg = any(arr.size for arr in neg_by_layer.values())
    n_pos = sum(arr.size for arr in pos_by_layer.values())
    n_neg = sum(arr.size for arr in neg_by_layer.values())
    n_total = n_pos + n_neg
    n_layers = len(layers)
    n_rows = 2 if has_neg else 1

    fig, axes = plt.subplots(
        n_rows, 2, figsize=(14, max(4, 0.45 * n_layers + 3) * n_rows),
        gridspec_kw={"width_ratios": [1.1, 1.0]}, constrained_layout=True, squeeze=False,
    )
    cmap = plt.get_cmap("viridis", max(n_layers, 2))
    color_for_layer = lambda i: cmap(i)

    _spectrum_panel(axes[0, 0], axes[0, 1], layers, pos_by_layer,
                    title=f"positive eigenvalues  ({n_pos}/{n_total})",
                    color_for_layer=color_for_layer, log_x=log_x)
    if has_neg:
        _spectrum_panel(axes[1, 0], axes[1, 1], layers, neg_by_layer,
                        title=f"|negative eigenvalues|  ({n_neg}/{n_total})",
                        color_for_layer=color_for_layer, log_x=log_x)

    fig.suptitle(factor_dir.name, fontsize=12, fontweight="bold")
    return fig, eigs_by_layer
