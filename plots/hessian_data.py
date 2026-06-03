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
    pts = row.get("pseudo_target_strategy")
    pts_s = f" · {pts}" if pts else ""
    return (
        f"#{int(row['result_id'])} · {row['model_id']} · {row['dataset_name']} · "
        f"ep{int(row['epoch'])} · λ={row['damping_value']:g} · "
        f"{row['damping_strategy']}{pif_s}{pts_s} · {row['run_id']}"
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


def influence_paths_for_result(df: pd.DataFrame, result_id: int) -> dict[str, str]:
    """method -> npy_path for one result_id, read straight from the joined df."""
    sub = df[(df["result_id"] == result_id) & df["npy_path"].notna()]
    return dict(zip(sub["approximator"], sub["npy_path"]))


# ── LDS slicing ───────────────────────────────────────────────────────
_LDS_GROUP = ["approximator", "epoch", "damping_value", "damping_strategy",
              "pseudo_target_strategy"]


def slice_lds(df: pd.DataFrame, *, model: str, sampling: str,
              strategy: Optional[str] = None) -> pd.DataFrame:
    """LDS rows for one (model, sampling), optionally one damping strategy,
    collapsed to one row per sweep point. Collapsing removes the metrics-join
    fan-out and averages any runs that recomputed the same config."""
    sub = df[df["lds_mean"].notna()]
    sub = sub[(sub["model_id"] == model) & (sub["pseudo_target_strategy"] == sampling)]
    if strategy not in (None, "(all)"):
        sub = sub[sub["damping_strategy"] == strategy]
    agg = {c: "mean" for c in ("lds_mean", "lds_ci_low", "lds_ci_high", "lds_std")
           if c in sub.columns}
    return sub.dropna(subset=_LDS_GROUP).groupby(_LDS_GROUP, as_index=False).agg(agg)


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


def method_metric_table_sweep(df: pd.DataFrame, *, model: str, reference: str = "exact",
                              sampling: Optional[str] = None,
                              damping_strategy: Optional[str] = None) -> pd.DataFrame:
    """Like `method_metric_table` but pooled across a whole sweep: one row per
    (result_id, approximator) over all epochs/dampings for `model`. Built from
    the joined `df` directly. Use the `approximator` index level to colour the
    scatter. (The reference method is excluded — its error-vs-itself is 0.)"""
    sub = df[(df["model_id"] == model) & (df["reference"] == reference)
             & df["value"].notna() & (df["approximator"] != reference)]
    if sampling is not None:
        sub = sub[sub["pseudo_target_strategy"] == sampling]
    if damping_strategy is not None:
        sub = sub[sub["damping_strategy"] == damping_strategy]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index=["result_id", "approximator"],
                           columns=["computation_type", "metric"],
                           values="value", aggfunc="mean")
    wide = wide[[c for c in ordered_categories(sub) if c in wide.columns]]
    lds = (df[df["lds_mean"].notna()]
           .groupby(["result_id", "approximator"])["lds_mean"].mean())
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
    return out


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
        npy_path = Path(r["npy_path"])
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
    return pd.DataFrame(mat, index=xs, columns=xs), n_query


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
