"""Data layer for the runs.db explorer.

Everything that *derives data* — loading the DB into one DataFrame, slicing it
by a config, resolving a result_id from a selection, discovering factor
directories on disk, and the influence/eigenvalue computations. No matplotlib
and no Streamlit here; this module is UI-agnostic and imported by both the
notebooks and `hessian_plots.py` / `app.py`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata as _scipy_rankdata

try:
    import bottleneck as _bn

    def _rankdata(a):
        # bottleneck.rankdata is a C impl with average-tie ranking (matches
        # scipy) and is ~4x faster on the (n_query × n_train) influence arrays.
        return _bn.rankdata(a, axis=1)

except ImportError:  # pragma: no cover - fallback when bottleneck absent

    def _rankdata(a):
        return _scipy_rankdata(a, axis=1)

DB_PATH = Path(
    "/root/Hessian-Approximation-for-Toy-Models/experiments/outputs/runs.db"
)

# Canonical method ordering (used everywhere methods are listed).
APPROX_ORDER = [
    "exact", "gnh", "fim", "mac", "emac", "block_hessian", "block_fim",
    "shampoo", "eshampoo", "ekfac", "kfac", "identity", "eidentity",
]


def order_methods(methods: list[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(APPROX_ORDER)}
    return sorted(methods, key=lambda m: (rank.get(m, len(rank)), m))


# Human-readable model names (keyed on the stored model_id hash). Displayed
# everywhere a model is shown; unknown ids fall through to the raw hash.
MODEL_LABELS = {
    "mlp_08580ee2573a": "Small MLP8",
    "resnet_mlp_0a69ab6297da": "Small ResNet MLP3",
    "resnet_mlp_adc3030f5daa": "Small ResNet MLP8",
    "resnet_mlp_fbc1db7ec868": "ResNet MLP3",
    "resnet_mlp_swiglu_d4186381c706": "ResNet SwiGLU3",
}


def model_label(model_id: str) -> str:
    """Display name for a model_id (falls back to the raw hash)."""
    return MODEL_LABELS.get(model_id, model_id)


# ── DB access ─────────────────────────────────────────────────────────
def open_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"runs.db not found at {path}")
    # read-only so the UI can never mutate the DB
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_runs_db(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return the whole runs.db as one analysis-ready DataFrame.

    Joins: results ⟕ (metrics ⟗ influence on result_id+approximator) ⟕ runs
    `influence.method` is renamed to `approximator` to match `metrics`.
    """
    DROP_FROM_RUNS = {"params_json", "code_sha"}
    DROP_FROM_RESULTS = {"params_json"}
    with open_db(db_path) as con:
        runs = pd.read_sql_query("SELECT * FROM runs", con).drop(
            columns=list(DROP_FROM_RUNS), errors="ignore"
        )
        results = pd.read_sql_query("SELECT * FROM results", con).drop(
            columns=list(DROP_FROM_RESULTS), errors="ignore"
        )
        metrics = pd.read_sql_query("SELECT * FROM metrics", con)
        influence = pd.read_sql_query(
            "SELECT *, method AS approximator FROM influence", con
        ).drop(columns=["method"])
    method_keyed = metrics.merge(
        influence, on=["result_id", "approximator"], how="outer"
    )
    return results.merge(method_keyed, on="result_id", how="left").merge(
        runs, on="run_id", how="left"
    )


# ── Result identity / selection ──────────────────────────────────────
# In this DB a single run holds a whole damping sweep, so (run, model, epoch)
# is NOT unique — it maps to one result_id per (damping_value, sampling, …).
# These columns distinguish result_ids:
RESULT_KEY_COLS = [
    "result_id", "run_id", "model_id", "epoch", "dataset_name",
    "damping_value", "damping_strategy", "pseudo_target_strategy",
    "pseudo_inverse_factor",
]


def result_options(df: pd.DataFrame) -> pd.DataFrame:
    """One row per result_id with the columns needed to pick/label it."""
    cols = [c for c in RESULT_KEY_COLS if c in df.columns]
    return (
        df[cols]
        .drop_duplicates("result_id")
        .sort_values(["model_id", "run_id", "epoch", "damping_value"])
        .reset_index(drop=True)
    )


def result_label(row) -> str:
    """Human-readable one-liner for a result_id."""
    pif = row.get("pseudo_inverse_factor")
    pif_s = "" if pif is None or (isinstance(pif, float) and pd.isna(pif)) else f" pif={pif:g}"
    return (
        f"#{int(row['result_id'])} · {model_label(row['model_id'])} · {row['dataset_name']} · "
        f"ep{int(row['epoch'])} · λ={row['damping_value']:g} · "
        f"{row['damping_strategy']}{pif_s}"
    )


def resolve_result(pool: pd.DataFrame, *, model, epoch, lam, strat, sampling):
    """Resolve a (model, epoch, λ, damping_strategy, sampling) selection to a
    single result_id. These five fields uniquely identify a result; on the rare
    re-run, the latest run is chosen. Returns (result_id, row, n_matches)."""
    p = pool[
        (pool["model_id"] == model)
        & (pool["epoch"] == epoch)
        & (pool["damping_value"] == lam)
        & (pool["damping_strategy"] == strat)
        & (pool["pseudo_target_strategy"] == sampling)
    ].sort_values("run_id")
    row = p.iloc[-1]
    return int(row["result_id"]), row, len(p)


def mean_npy_path(path: str | Path) -> Path:
    """Influence is now saved as the per-train mean over queries in a sibling
    `<name>_mean.npy` (1D) next to the legacy `<name>.npy` query×train tensor.
    The DB still stores the legacy name, so rewrite to the `_mean` file when it
    exists; fall back to the original otherwise."""
    p = Path(path)
    cand = p.with_name(p.stem + "_mean.npy")
    return cand if cand.exists() else p


def influence_paths_for_result(df: pd.DataFrame, result_id: int) -> dict[str, str]:
    """method -> influence npy path (the `_mean` file) for one result_id."""
    sub = df[(df["result_id"] == result_id) & df["npy_path"].notna()]
    return {m: str(mean_npy_path(p)) for m, p in zip(sub["approximator"], sub["npy_path"])}


# ── LDS slicing ───────────────────────────────────────────────────────
# `collector_subset_size` is a grouping key (not an averaged axis): LDS climbs
# steeply with subset size, and mcmc sweeps several sizes while all_classes only
# ever has the full set. Averaging over it would silently drag the mcmc curve
# below all_classes; instead the caller fixes a size via `subset_size`.
_LDS_GROUP = ["approximator", "epoch", "damping_value", "damping_strategy",
              "pseudo_target_strategy", "collector_subset_size"]


def slice_lds(df: pd.DataFrame, *, model: str, sampling: str,
              strategy: Optional[str] = None,
              subset_size: Optional[int] = None) -> pd.DataFrame:
    """LDS rows for one (model, sampling), optionally one damping strategy and
    one collector subset size, collapsed to one row per sweep point. Collapsing
    removes the metrics-join fan-out and averages any runs that recomputed the
    same config. `subset_size=None` keeps every size as a separate row rather
    than averaging across them (see `_LDS_GROUP`)."""
    sub = df[df["lds_mean"].notna()]
    sub = sub[(sub["model_id"] == model) & (sub["pseudo_target_strategy"] == sampling)]
    if strategy not in (None, "(all)"):
        sub = sub[sub["damping_strategy"] == strategy]
    if subset_size is not None:
        sub = sub[sub["collector_subset_size"] == subset_size]
    agg = {c: "mean" for c in ("lds_mean", "lds_ci_low", "lds_ci_high", "lds_std")
           if c in sub.columns}
    return sub.dropna(subset=_LDS_GROUP).groupby(_LDS_GROUP, as_index=False).agg(agg)


# ── Sample-size Pareto slicing ────────────────────────────────────────
# `collector_subset_size` is the raw number of training points fed to the
# collector, but the *effective* sample/compute cost (the Pareto x-axis) scales
# with two extra factors:
#   • sampling: `all_classes` draws one pseudo-target per logit, so it uses
#     `output_dim`× (the logit-space dimension) the gradients that `mcmc` does.
#   • eigenvalue correction: the E-* methods take a second pass over the
#     collector data to fit the eigenbasis diagonal, doubling their cost.
# Both are independent multipliers, so they compound.
_PARETO_GROUP = ["approximator", "collector_subset_size", "pseudo_target_strategy"]

# E-prefixed eigenvalue-corrected methods (2× the collector cost).
EIGENVALUE_CORRECTED = {"emac", "ekfac", "eshampoo", "eidentity"}
NUM_CLASSES_FALLBACK = 10  # logit dim when output_dim can't be read from the DB

# A method "family" pairs a base approximator with its eigenvalue-corrected
# E-variant (kfac ↔ ekfac, shampoo ↔ eshampoo, …). On the Pareto plot a family
# is one colour and one connected line: the E-variant is the same curve, just
# shifted right to 2× the cost. Methods with no E-variant are singleton
# families. Keyed by the base name so the family inherits the base's colour.
METHOD_FAMILY = {
    "mac": "mac", "emac": "mac",
    "kfac": "kfac", "ekfac": "kfac",
    "shampoo": "shampoo", "eshampoo": "shampoo",
    "identity": "identity", "eidentity": "identity",
}


def method_family(method: str) -> str:
    """Base-approximator family for a method (kfac & ekfac → 'kfac')."""
    return METHOD_FAMILY.get(method, method)


def model_output_dim(db_path: Path, model_id: str,
                     default: int = NUM_CLASSES_FALLBACK) -> int:
    """Logit-space dimension (`model_config.output_dim`) for a model, read from
    the stored `params_json`. Falls back to `default` if unavailable."""
    with open_db(db_path) as con:
        row = con.execute(
            "SELECT params_json FROM results "
            "WHERE model_id = ? AND params_json IS NOT NULL LIMIT 1",
            (model_id,),
        ).fetchone()
    if row and row[0]:
        try:
            return int(json.loads(row[0])["model_config"]["output_dim"])
        except (KeyError, ValueError, TypeError):
            pass
    return default


def pareto_effective_samples(subset_size, method: str, sampling: str, *,
                             num_classes: int) -> float:
    """Effective sample cost for one (method, sampling, subset_size):
    subset_size × (num_classes if all_classes) × (2 if E-* method)."""
    factor = 1
    if sampling == "all_classes":
        factor *= num_classes
    if method in EIGENVALUE_CORRECTED:
        factor *= 2
    return float(subset_size) * factor


def slice_pareto(df: pd.DataFrame, *, model: str, num_classes: int = NUM_CLASSES_FALLBACK,
                 strategy: Optional[str] = None, epoch: Optional[int] = None,
                 damping: Optional[float] = None) -> pd.DataFrame:
    """LDS rows for one (model[, strategy, epoch, damping]) collapsed to one row
    per (method, collector_subset_size, sampling) — both samplings kept so the
    Pareto plot can overlay them. Adds an `effective_samples` column (the cost
    axis; see `pareto_effective_samples`). Repeated runs and any unfixed
    epoch/damping/strategy are averaged per group."""
    sub = df[df["lds_mean"].notna()]
    sub = sub[sub["model_id"] == model]
    if strategy not in (None, "(all)"):
        sub = sub[sub["damping_strategy"] == strategy]
    if epoch is not None:
        sub = sub[sub["epoch"] == epoch]
    if damping is not None:
        sub = sub[np.isclose(sub["damping_value"], damping)]
    agg = {c: "mean" for c in ("lds_mean", "lds_ci_low", "lds_ci_high", "lds_std")
           if c in sub.columns}
    out = sub.dropna(subset=_PARETO_GROUP).groupby(_PARETO_GROUP, as_index=False).agg(agg)
    out["effective_samples"] = [
        pareto_effective_samples(sz, m, s, num_classes=num_classes)
        for sz, m, s in zip(out["collector_subset_size"], out["approximator"],
                            out["pseudo_target_strategy"])
    ]
    out["family"] = [method_family(m) for m in out["approximator"]]
    return out


# ── Metrics ───────────────────────────────────────────────────────────
def load_metrics_df(con: sqlite3.Connection, result_id: int,
                    reference: str = "exact") -> pd.DataFrame:
    """Long-format metrics for one result_id, filtered to a reference."""
    return pd.read_sql_query(
        "SELECT computation_type, reference, approximator, metric, value "
        "FROM metrics WHERE result_id = ? AND reference = ?",
        con, params=(result_id, reference),
    )


def result_meta(con: sqlite3.Connection, result_id: int) -> dict:
    row = con.execute(
        "SELECT num_parameters, val_loss, val_accuracy, model_id, epoch, run_id "
        "FROM results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    keys = ["num_parameters", "val_loss", "val_accuracy", "model_id", "epoch", "run_id"]
    return dict(zip(keys, row)) if row else {k: None for k in keys}


_CAT_PRIORITY = {"matrix": 0, "hvp": 1, "ihvp": 2, "round_trip": 3}


def ordered_categories(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Ordered, de-duplicated (computation_type, metric) pairs in a metrics df."""
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ct in sorted(df["computation_type"].unique(), key=lambda c: _CAT_PRIORITY.get(c, 99)):
        sub = df[df["computation_type"] == ct]
        for metric in sub["metric"].drop_duplicates():
            key = (ct, metric)
            if key not in seen:
                rows.append(key)
                seen.add(key)
    return rows


def metric_category_options(result_id: int, reference: str = "exact",
                            db_path: Path = DB_PATH) -> list[tuple[str, str]]:
    """Available (computation_type, metric) pairs for a result_id + reference."""
    with open_db(db_path) as con:
        df = load_metrics_df(con, result_id, reference=reference)
    df = df[df["approximator"] != reference]
    return ordered_categories(df)


# Sentinel axis for the LDS-mean column in the method×metric table.
LDS_AXIS = ("__lds__", "mean")


def method_metric_table(result_id: int, df: pd.DataFrame, reference: str = "exact",
                        db_path: Path = DB_PATH) -> pd.DataFrame:
    """Wide table for the metric-correlation scatter: one row per method (the
    reference method excluded), one column per (computation_type, metric) value,
    plus an `LDS_AXIS` column of lds_mean. Each point in the scatter is a method.
    """
    with open_db(db_path) as con:
        m = load_metrics_df(con, result_id, reference=reference)
    m = m[m["approximator"] != reference]
    wide = m.pivot_table(index="approximator",
                         columns=["computation_type", "metric"],
                         values="value", aggfunc="mean")
    # order columns canonically (matrix → hvp → ihvp → round_trip)
    wide = wide[[c for c in ordered_categories(m) if c in wide.columns]]
    lds = (df[df["result_id"] == result_id].dropna(subset=["lds_mean"])
           .groupby("approximator")["lds_mean"].mean())
    wide[LDS_AXIS] = lds
    return wide


_SWEEP_KEYS = ["epoch", "damping_value", "damping_strategy",
               "pseudo_target_strategy", "approximator"]


def method_metric_table_sweep(df: pd.DataFrame, *, model: str, reference: str = "exact",
                              sampling: Optional[str] = None,
                              damping_strategy: Optional[str] = None) -> pd.DataFrame:
    """Like `method_metric_table` but pooled across a whole sweep: one row per
    (epoch, λ, damping_strategy, sampling, approximator) config for `model`.
    Repeated runs of the same config are averaged (as in `slice_lds`), so `n`
    counts unique configs rather than raw rows. Index levels `approximator` and
    `damping_value` drive the Fig 1(b) shape/colour encoding. (The reference
    method is excluded — its error-vs-itself is 0.)"""
    sub = df[(df["model_id"] == model) & (df["reference"] == reference)
             & df["value"].notna() & (df["approximator"] != reference)]
    if sampling is not None:
        sub = sub[sub["pseudo_target_strategy"] == sampling]
    if damping_strategy is not None:
        sub = sub[sub["damping_strategy"] == damping_strategy]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index=_SWEEP_KEYS, columns=["computation_type", "metric"],
                           values="value", aggfunc="mean")  # mean over repeated runs
    wide = wide[[c for c in ordered_categories(sub) if c in wide.columns]]
    lds = (df[df["lds_mean"].notna()].groupby(_SWEEP_KEYS)["lds_mean"].mean())
    wide[LDS_AXIS] = lds
    return wide


# ── Factor directory discovery ────────────────────────────────────────
def find_factor_dirs(root: str | Path) -> list[str]:
    """Eigenvalue-bearing dirs are the per-method `*_layer_matrix/` ones,
    identified by their blocks.npz (the collector-level manifest has none)."""
    base = Path(root)
    if not base.exists():
        return []
    return sorted(str(m.parent) for m in base.rglob("blocks.npz"))


def _method_from_dir(name: str) -> str:
    """`exact_hessian_layer_matrix` -> `exact`, `kfac_layer_matrix` -> `kfac`."""
    base = name[: -len("_layer_matrix")] if name.endswith("_layer_matrix") else name
    return "exact" if base == "exact_hessian" else base


def _sampling_from_collector(name: str) -> str:
    """`collector_mcmc_r1_s42` -> `mcmc`, `collector_all_classes_r1_s42` -> `all_classes`."""
    m = re.fullmatch(r"collector_(.+)_r\d+_s\d+", name)
    return m.group(1) if m else name.replace("collector_", "")


def parse_factor_dirs(dirs, root) -> list[dict]:
    """Parse `<root>/<dataset>/<model>/epoch_<N>/<collector>/.../<method>_layer_matrix`
    factor dirs into cascade-able records. Unmatched layouts are skipped."""
    recs = []
    for d in dirs:
        try:
            parts = Path(d).relative_to(root).parts
        except ValueError:
            continue
        if len(parts) >= 4 and parts[2].startswith("epoch_"):
            try:
                epoch = int(parts[2][len("epoch_"):])
            except ValueError:
                continue
            recs.append({
                "dataset": parts[0], "model": parts[1], "epoch": epoch,
                "sampling": _sampling_from_collector(parts[3]),
                "method": _method_from_dir(parts[-1]), "path": d,
            })
    return recs


# ── Influence Spearman (vectorised, bottleneck-accelerated) ───────────
def _as_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected influence shape {arr.shape}")


def _row_centered_unit(ranks: np.ndarray) -> np.ndarray:
    centred = ranks - ranks.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    return centred / norm


def _per_query_spearman_vec(ua: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.einsum("qt,qt->q", ua, ub)


def _aggregate(values: np.ndarray, how: str) -> float:
    if how == "mean":
        return float(np.nanmean(values))
    if how == "median":
        return float(np.nanmedian(values))
    raise ValueError(f"Unknown aggregate {how!r} (use mean or median)")


def _drop_degenerate_rowcols(mat: pd.DataFrame) -> pd.DataFrame:
    """Drop rows/columns that carry no rank signal.

    A constant influence vector (e.g. an all-zero pseudo-inverse row) centres to
    the zero vector, so its per-query Spearman with *every* other entry is
    exactly 0 (or NaN). In the heatmap that shows up as a blank 0.0 stripe down a
    whole row and column — remove those entries entirely so only informative
    ones remain. The diagonal (self-correlation, 1.0) is ignored when deciding."""
    arr = mat.to_numpy()
    n = len(arr)
    if n == 0:
        return mat
    dead = np.isnan(arr) | (arr == 0.0)
    np.fill_diagonal(dead, False)  # the diagonal is always 1.0 — don't count it
    keep = dead.sum(axis=1) != (n - 1)  # keep unless the whole off-diagonal is dead
    if keep.all():
        return mat
    labels = mat.index[keep]
    out = mat.loc[labels, labels]
    out.attrs.update(mat.attrs)
    return out


def compute_influence_spearman_matrix(
    result_id: int,
    paths: dict[str, str],
    methods: Optional[list[str]] = None,
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Pairwise per-query Spearman matrix across methods for one result_id.

    `paths` maps method -> influence .npy path (use influence_paths_for_result).
    """
    if not paths:
        raise ValueError("No influence rows for this result_id")

    methods = order_methods(list(paths)) if methods is None else [m for m in methods if m in paths]

    prepared: dict[str, np.ndarray] = {}
    canonical_shape: Optional[tuple[int, ...]] = None
    for m in methods:
        arr = _as_2d(np.load(paths[m]).astype(np.float64))
        if canonical_shape is None:
            canonical_shape = arr.shape
        if arr.shape != canonical_shape:
            print(f"  skip {m}: shape {arr.shape} != {canonical_shape}")
            continue
        prepared[m] = _row_centered_unit(_rankdata(arr))

    methods = [m for m in methods if m in prepared]
    n = len(methods)
    mat = np.full((n, n), np.nan)
    for i, mi in enumerate(methods):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            rhos = _per_query_spearman_vec(prepared[mi], prepared[methods[j]])
            mat[i, j] = mat[j, i] = _aggregate(rhos, aggregate)

    out = pd.DataFrame(
        mat,
        index=pd.Index(methods, name="method"),
        columns=pd.Index(methods, name="method"),
    )
    out.attrs.update(
        n_query=canonical_shape[0] if canonical_shape else 0,
        n_train=canonical_shape[1] if canonical_shape else 0,
        aggregate=aggregate, result_id=result_id,
    )
    return _drop_degenerate_rowcols(out)


def reorder_spearman(rho: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    available = list(rho.index)
    kept = [m for m in methods if m in available]
    order = kept + [m for m in available if m not in kept]
    out = rho.loc[order, order]
    out.attrs.update(rho.attrs)
    return out


def spearman_matrix_across_axis(
    df: pd.DataFrame,
    *,
    model_id: str,
    sweep: str,
    fix,
    method: str = "exact",
    aggregate: str = "mean",
    damping_strategy: Optional[str] = None,
    pseudo_target_strategy: Optional[str] = None,
) -> tuple[pd.DataFrame, int]:
    """Pairwise per-query Spearman of `method`'s influence vectors across the
    swept axis. Reads influence paths straight from the joined `df` (which
    carries result_id + npy_path), so it is damping- and sampling-aware.
    Returns (matrix indexed by sweep value, n_query)."""
    if sweep not in {"damping", "epoch"}:
        raise ValueError("sweep must be 'damping' or 'epoch'")
    sweep_col = "damping_value" if sweep == "damping" else "epoch"
    fix_col = "epoch" if sweep == "damping" else "damping_value"

    sub = df[
        (df["model_id"] == model_id)
        & (df["approximator"] == method)
        & df["npy_path"].notna()
    ]
    if damping_strategy is not None:
        sub = sub[sub["damping_strategy"] == damping_strategy]
    if pseudo_target_strategy is not None:
        sub = sub[sub["pseudo_target_strategy"] == pseudo_target_strategy]
    rows = sub[sub[fix_col] == fix].drop_duplicates(sweep_col).sort_values(sweep_col)
    if rows.empty:
        avail_fix = sorted(sub[fix_col].dropna().unique().tolist())
        raise ValueError(
            f"No rows for model_id={model_id} method={method} with {fix_col}={fix}.\n"
            f"  available {fix_col}: {avail_fix}"
        )

    prepared: dict = {}
    canonical_shape = None
    for _, r in rows.iterrows():
        x = r[sweep_col]
        npy_path = mean_npy_path(r["npy_path"])
        if not npy_path.exists():
            print(f"  skip {sweep}={x}: influence file missing on disk ({npy_path})")
            continue
        arr = _as_2d(np.load(npy_path).astype(np.float64))
        if canonical_shape is None:
            canonical_shape = arr.shape
        if arr.shape != canonical_shape:
            print(f"  skip {sweep}={x}: shape {arr.shape} != {canonical_shape}")
            continue
        prepared[x] = _row_centered_unit(_rankdata(arr))

    if len(prepared) < 2:
        avail_sweep = sorted(sub[sweep_col].dropna().unique().tolist())
        raise ValueError(
            f"Need ≥2 sweep points with usable {method!r} influence; got {len(prepared)}.\n"
            f"  filter: model_id={model_id}, {fix_col}={fix}\n"
            f"  loaded {sweep} values: {sorted(prepared)}\n"
            f"  all {sweep_col} for this model: {avail_sweep}"
        )

    xs = sorted(prepared)
    n = len(xs)
    mat = np.full((n, n), np.nan)
    for i, xi in enumerate(xs):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            rhos = _per_query_spearman_vec(prepared[xi], prepared[xs[j]])
            mat[i, j] = mat[j, i] = _aggregate(rhos, aggregate)

    n_query = canonical_shape[0] if canonical_shape else 0
    return _drop_degenerate_rowcols(pd.DataFrame(mat, index=xs, columns=xs)), n_query


def _prepared_influence(df: pd.DataFrame, *, model, epoch, damping, strategy,
                        method, sampling) -> Optional[np.ndarray]:
    """Rank-prepared influence vector for one fully-specified config, or None."""
    r = df[(df["model_id"] == model) & (df["epoch"] == epoch)
           & (df["damping_value"] == damping) & (df["damping_strategy"] == strategy)
           & (df["approximator"] == method) & (df["pseudo_target_strategy"] == sampling)
           & df["npy_path"].notna()]
    if r.empty:
        return None
    p = mean_npy_path(r.iloc[0]["npy_path"])
    if not p.exists():
        return None
    return _row_centered_unit(_rankdata(_as_2d(np.load(p).astype(np.float64))))


def sampling_spearman(df: pd.DataFrame, *, model: str, epoch: int, damping: float,
                      strategy: str, sampling_a: str, sampling_b: str,
                      methods: Optional[list[str]] = None,
                      aggregate: str = "mean") -> pd.Series:
    """Per-method Spearman between the influence vectors under two samplings
    (e.g. mcmc vs all_classes) at one fixed (model, epoch, λ, strategy). ρ≈1 ⇒
    the sampling choice barely changes that method's influence ranking."""
    base = df[(df["model_id"] == model) & (df["epoch"] == epoch)
              & (df["damping_value"] == damping) & (df["damping_strategy"] == strategy)
              & df["npy_path"].notna()]
    if methods is None:
        methods = order_methods(base["approximator"].dropna().unique().tolist())
    out: dict[str, float] = {}
    for m in methods:
        va = _prepared_influence(df, model=model, epoch=epoch, damping=damping,
                                 strategy=strategy, method=m, sampling=sampling_a)
        vb = _prepared_influence(df, model=model, epoch=epoch, damping=damping,
                                 strategy=strategy, method=m, sampling=sampling_b)
        if va is None or vb is None or va.shape != vb.shape:
            continue
        out[m] = _aggregate(_per_query_spearman_vec(va, vb), aggregate)
    return pd.Series(out, name="spearman")


# ── Factor eigenvalues ────────────────────────────────────────────────
def _assemble_full_dense_eigs(manifest: dict, z) -> np.ndarray:
    """Eigenvalues of the full assembled matrix from all dense blocks.

    Used when the factor stores off-diagonal (cross-layer) blocks — i.e. it is
    a genuine full matrix, not block-diagonal. Blocks are laid out by
    `layer_names`; the matrix is symmetrised before `eigvalsh`.
    """
    diag_dims = {
        b["key"].split("::")[0]: np.asarray(z[f"{b['key']}//matrix"]).shape[0]
        for b in manifest["blocks"]
        if b["block_type"] == "dense"
        and b["key"].split("::")[0] == b["key"].split("::")[1]
    }
    order = [ln for ln in manifest.get("layer_names", []) if ln in diag_dims]
    order += [ln for ln in diag_dims if ln not in order]  # any not listed

    offset, pos = {}, 0
    for ln in order:
        offset[ln] = pos
        pos += diag_dims[ln]

    M = np.zeros((pos, pos))
    for blk in manifest["blocks"]:
        a, b = blk["key"].split("::")
        if a not in offset or b not in offset or blk["block_type"] != "dense":
            continue
        mat = np.asarray(z[f"{blk['key']}//matrix"])
        M[offset[a]:offset[a] + diag_dims[a], offset[b]:offset[b] + diag_dims[b]] = mat

    return np.linalg.eigvalsh(0.5 * (M + M.T))


def load_factor_eigenvalues(factor_dir: Path | str) -> dict[str, np.ndarray]:
    """Eigenvalues of a saved factor directory (manifest.json + blocks.npz).

    If the factor stores off-diagonal (cross-layer) blocks it is a full matrix,
    so we diagonalise the whole assembled matrix and return a single spectrum
    keyed ``"full matrix"``. Otherwise it is block-diagonal and we return the
    per-layer diagonal-block spectra (one entry per layer).
    """
    factor_dir = Path(factor_dir)
    manifest = json.loads((factor_dir / "manifest.json").read_text())
    z = np.load(factor_dir / "blocks.npz", allow_pickle=True)

    has_offdiag = any(
        blk["key"].split("::")[0] != blk["key"].split("::")[1]
        for blk in manifest["blocks"]
    )
    if has_offdiag:
        eigs = _assemble_full_dense_eigs(manifest, z)
        return {"full matrix": np.sort(eigs)[::-1]}

    out: dict[str, np.ndarray] = {}
    for blk in manifest["blocks"]:
        key = blk["key"]
        a, b = key.split("::")
        if a != b:
            continue
        bt = blk["block_type"]
        if bt == "kronecker":
            fields = set(blk["fields"])
            if "Lambda" in fields:
                eigs = np.asarray(z[f"{key}//Lambda"]).ravel()
            else:
                lA = np.asarray(z[f"{key}//lambda_A"])
                lG = np.asarray(z[f"{key}//lambda_G"])
                eigs = np.outer(lA, lG).ravel()
        elif bt == "dense":
            mat = np.asarray(z[f"{key}//matrix"])
            eigs = np.linalg.eigvalsh(0.5 * (mat + mat.T))
        else:
            print(f"  skip {key}: unknown block_type {bt!r}")
            continue
        out[a] = np.sort(eigs)[::-1]
    return out
