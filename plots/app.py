"""Interactive explorer for runs.db — reproduce the notebook plots with any
hyperparameters you like.

Thin Streamlit UI: widgets gather a config, then `hessian_data` (D) slices/
resolves it and `hessian_plots` (P) draws it.

Run with:
    uv run streamlit run experiments/app.py
    # or point at a different DB:
    uv run streamlit run experiments/app.py -- --db /path/to/runs.db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hessian_data as D  # noqa: E402
import hessian_plots as P  # noqa: E402

st.set_page_config(page_title="runs.db explorer", layout="wide")

# ── Config hash registry ──────────────────────────────────────────────
# Every config-defining widget records its value into CFG via track(); at the
# end we hash CFG to an 8-char id and persist {id: config} so a pasted id
# restores the exact selections. Widget keys are all prefixed "cfg_".
_REGISTRY = Path(__file__).resolve().parent / ".app_configs.json"
CFG: dict = {}


def track(key: str, value):
    CFG[key] = value
    return value


def _jsonsafe(v):
    if isinstance(v, (list, tuple)):
        return [_jsonsafe(x) for x in v]
    if isinstance(v, (str, bytes, bool, int, float)) or v is None:
        return v
    if hasattr(v, "item"):  # numpy / pandas scalar (int64, float64, …)
        return v.item()
    return v


def _coerce(v):
    # invert _jsonsafe just enough: categories are list-of-[ct,metric] -> tuples
    if isinstance(v, list):
        return [tuple(x) if isinstance(x, list) else x for x in v]
    return v


def _load_registry() -> dict:
    try:
        return json.loads(_REGISTRY.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def config_hash() -> str:
    canon = json.dumps({k: _jsonsafe(v) for k, v in CFG.items()}, sort_keys=True)
    h = hashlib.sha256(canon.encode()).hexdigest()[:8]
    reg = _load_registry()
    if reg.get(h) != json.loads(canon):
        reg[h] = json.loads(canon)
        _REGISTRY.write_text(json.dumps(reg, indent=0))
    return h


# Apply a pending config BEFORE any widget is created (Streamlit forbids
# writing a widget-backed session_state key after the widget exists).
if "_pending_cfg" in st.session_state:
    for k, v in st.session_state.pop("_pending_cfg").items():
        st.session_state[k] = _coerce(v)


# ── CLI default DB ────────────────────────────────────────────────────
def _default_db() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--db", default=str(D.DB_PATH))
    args, _ = p.parse_known_args()
    return args.db


# ── Cached loaders (keyed on path + mtime so edits invalidate) ────────
@st.cache_data(show_spinner="Loading runs.db …")
def get_df(db_path: str, mtime: float):
    return D.load_runs_db(Path(db_path))


@st.cache_data(show_spinner=False)
def get_result_ids_in(db_path: str, mtime: float, table: str) -> set[int]:
    with D.open_db(Path(db_path)) as con:
        return {int(r[0]) for r in
                con.execute(f"SELECT DISTINCT result_id FROM {table}").fetchall()}


@st.cache_data(show_spinner=False)
def get_factor_dirs(root: str) -> list[str]:
    return D.find_factor_dirs(root)


def show(fig):
    """Render a matplotlib figure and free it."""
    st.pyplot(fig, width="content")
    plt.close(fig)


def fmt(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def pick_result_id(pool, *, key: str):
    """Cascading Model → Epoch → λ → strategy → sampling selectors that resolve
    to one result_id from `pool`. Returns (result_id, row)."""
    model = track(f"cfg_{key}_model", st.sidebar.selectbox(
        "Model", sorted(pool["model_id"].unique()), key=f"cfg_{key}_model"))
    p = pool[pool["model_id"] == model]
    epoch = track(f"cfg_{key}_epoch", st.sidebar.selectbox(
        "Epoch", sorted(p["epoch"].unique()), key=f"cfg_{key}_epoch"))
    p = p[p["epoch"] == epoch]
    lam = track(f"cfg_{key}_lam", st.sidebar.selectbox(
        "Damping λ", sorted(p["damping_value"].unique()), format_func=fmt, key=f"cfg_{key}_lam"))
    p = p[p["damping_value"] == lam]
    strat = track(f"cfg_{key}_strat", st.sidebar.selectbox(
        "Damping strategy", sorted(p["damping_strategy"].unique()), key=f"cfg_{key}_strat"))
    p = p[p["damping_strategy"] == strat]
    sampling = track(f"cfg_{key}_pts", st.sidebar.selectbox(
        "Sampling (pseudo-target)", sorted(p["pseudo_target_strategy"].unique()),
        key=f"cfg_{key}_pts"))
    rid, row, n = D.resolve_result(
        pool, model=model, epoch=epoch, lam=lam, strat=strat, sampling=sampling)
    if n > 1:
        st.sidebar.caption(f"{n} runs match — using latest ({row['run_id']})")
    return rid, row


# ── Sidebar: DB + family ──────────────────────────────────────────────
st.sidebar.title("runs.db explorer")
db_path = track("cfg_db", st.sidebar.text_input("Database path", value=_default_db(), key="cfg_db"))
if not Path(db_path).exists():
    st.error(f"DB not found: {db_path}")
    st.stop()

mtime = Path(db_path).stat().st_mtime
df = get_df(db_path, mtime)
metric_rids = get_result_ids_in(db_path, mtime, "metrics")
infl_rids = get_result_ids_in(db_path, mtime, "influence")
opts = D.result_options(df)

st.sidebar.caption(
    f"{df['run_id'].nunique()} runs · {len(opts)} results · "
    f"{df['model_id'].nunique()} models"
)

family = track("cfg_family", st.sidebar.radio(
    "Plot family",
    ["LDS sweeps", "Metric bars", "Metric correlation",
     "Influence Spearman", "Factor eigenvalues"],
    key="cfg_family",
))
st.sidebar.divider()

ALL_METHODS = D.order_methods(df["approximator"].dropna().unique().tolist())


# ──────────────────────────────────────────────────────────────────────
# LDS sweeps
# ──────────────────────────────────────────────────────────────────────
if family == "LDS sweeps":
    st.header("LDS sweeps")
    lds = df[df["lds_mean"].notna()]
    models = sorted(lds["model_id"].dropna().unique())
    if not models:
        st.warning("No LDS rows in this DB.")
        st.stop()

    model = track("cfg_lds_model", st.sidebar.selectbox("Model", models, key="cfg_lds_model"))
    mlds = lds[lds["model_id"] == model]
    sampling = track("cfg_lds_pts", st.sidebar.selectbox(
        "Sampling (pseudo-target)",
        sorted(mlds["pseudo_target_strategy"].dropna().unique()), key="cfg_lds_pts"))
    strategies = sorted(mlds[mlds["pseudo_target_strategy"] == sampling]
                        ["damping_strategy"].dropna().unique())
    strat = track("cfg_lds_strat", st.sidebar.selectbox(
        "Damping strategy", ["(all)"] + strategies, key="cfg_lds_strat"))

    df_sub = D.slice_lds(df, model=model, sampling=sampling,
                         strategy=None if strat == "(all)" else strat)

    variant = track("cfg_lds_variant", st.sidebar.radio(
        "Variant",
        ["Fix method (epoch × damping)", "Fix damping (epoch × method)",
         "Fix epoch (damping × method)", "Heatmap per method"],
        key="cfg_lds_variant",
    ))
    is_heatmap = variant == "Heatmap per method"
    if not is_heatmap:
        kind = "bar" if track("cfg_lds_style", st.sidebar.radio(
            "Style", ["Lines", "Bars"], key="cfg_lds_style")) == "Bars" else "line"
    else:
        kind = "line"
    show_band = track("cfg_lds_band", st.sidebar.checkbox(
        "Show CI" + ("" if is_heatmap else " (error bars)" if kind == "bar" else " band"),
        value=True, key="cfg_lds_band",
    ))
    annotate = track("cfg_lds_annot", st.sidebar.checkbox(
        "Annotate heatmap", value=True, key="cfg_lds_annot"))

    methods_here = D.order_methods(df_sub["approximator"].dropna().unique().tolist())
    epochs = sorted(int(e) for e in df_sub["epoch"].dropna().unique())
    dampings = sorted(df_sub["damping_value"].dropna().unique())

    st.caption(f"{len(df_sub)} LDS rows · model `{model}` · sampling `{sampling}` · strategy `{strat}`")

    if variant == "Fix method (epoch × damping)":
        sel = track("cfg_lds_mf_methods", st.sidebar.multiselect(
            "Method(s)", methods_here, default=methods_here[:1], key="cfg_lds_mf_methods"))
        for m in sel:
            fig = P.plot_lds_methodfix(df_sub, m, show_band=show_band, kind=kind)
            fig.suptitle(model, fontsize=10)
            show(fig)

    elif variant == "Fix damping (epoch × method)":
        d = track("cfg_lds_df_lam", st.sidebar.selectbox(
            "Damping λ", dampings, format_func=fmt, key="cfg_lds_df_lam"))
        msel = track("cfg_lds_df_methods", st.sidebar.multiselect(
            "Methods", methods_here, default=methods_here, key="cfg_lds_df_methods"))
        fig = P.plot_lds_dampingfix(
            df_sub, d, methods=msel or None, show_band=show_band, kind=kind,
            strategy=None if strat == "(all)" else strat,
        )
        fig.suptitle(model, fontsize=10)
        show(fig)

    elif variant == "Fix epoch (damping × method)":
        e = track("cfg_lds_ef_epoch", st.sidebar.selectbox(
            "Epoch", epochs, key="cfg_lds_ef_epoch"))
        msel = track("cfg_lds_ef_methods", st.sidebar.multiselect(
            "Methods", methods_here, default=methods_here, key="cfg_lds_ef_methods"))
        fig = P.plot_lds_epochfix(
            df_sub, e, methods=msel or None, show_band=show_band, kind=kind,
            strategy=None if strat == "(all)" else strat,
        )
        fig.suptitle(model, fontsize=10)
        show(fig)

    else:  # Heatmap per method
        sel = track("cfg_lds_hm_methods", st.sidebar.multiselect(
            "Method(s)", methods_here, default=methods_here[:1], key="cfg_lds_hm_methods"))
        for m in sel:
            fig, _ = P.plot_lds_heatmap_for_method(df_sub, m, annotate=annotate)
            fig.suptitle(model, fontsize=10)
            show(fig)


# ──────────────────────────────────────────────────────────────────────
# Metric bars
# ──────────────────────────────────────────────────────────────────────
elif family == "Metric bars":
    st.header("Metric bar plots")
    mopts = opts[opts["result_id"].isin(metric_rids)].copy()
    if mopts.empty:
        st.warning("No results with metrics in this DB.")
        st.stop()

    rid, _row = pick_result_id(mopts, key="mb")
    reference = track("cfg_mb_ref", st.sidebar.selectbox(
        "Reference (vs)", ["exact", "gnh"], key="cfg_mb_ref"))
    approxs = track("cfg_mb_approxs", st.sidebar.multiselect(
        "Approximators (empty = all)", ALL_METHODS, default=[], key="cfg_mb_approxs"))

    cat_opts = D.metric_category_options(int(rid), reference=reference, db_path=Path(db_path))
    if not cat_opts:
        st.warning(f"No metrics for reference={reference} on this result.")
        st.stop()
    cats = track("cfg_mb_cats", st.sidebar.multiselect(
        "Categories (empty = all)", cat_opts, default=[],
        format_func=P.category_label, key="cfg_mb_cats"))
    try:
        fig = P.plot_metrics_for(
            int(rid), reference=reference, approxs=approxs or None,
            categories=cats or None, db_path=Path(db_path),
        )
        show(fig)
    except ValueError as e:
        st.warning(str(e))


# ──────────────────────────────────────────────────────────────────────
# Metric correlation (one point per method)
# ──────────────────────────────────────────────────────────────────────
elif family == "Metric correlation":
    st.header("Metric correlation")
    scope = track("cfg_mc_scope", st.sidebar.radio(
        "Scope", ["Single result", "Across sweep"], key="cfg_mc_scope"))

    if scope == "Single result":
        mopts = opts[opts["result_id"].isin(metric_rids)].copy()
        if mopts.empty:
            st.warning("No results with metrics in this DB.")
            st.stop()
        rid, _row = pick_result_id(mopts, key="mc")
        reference = track("cfg_mc_ref", st.sidebar.selectbox(
            "Reference (vs)", ["exact", "gnh"], key="cfg_mc_ref"))
        table = D.method_metric_table(int(rid), df, reference=reference, db_path=Path(db_path))
        point_methods = None  # index is the method
        sweep_damping = None  # single config → colour by method, not λ
        annotate_default = True
    else:  # Across sweep — one point per (config, method)
        mopts = opts[opts["result_id"].isin(metric_rids)]
        model = track("cfg_mc_model", st.sidebar.selectbox(
            "Model", sorted(mopts["model_id"].unique()), key="cfg_mc_model"))
        msub = mopts[mopts["model_id"] == model]
        sampling = track("cfg_mc_pts", st.sidebar.selectbox(
            "Sampling (pseudo-target)",
            sorted(msub["pseudo_target_strategy"].dropna().unique()), key="cfg_mc_pts"))
        strats = sorted(msub[msub["pseudo_target_strategy"] == sampling]
                        ["damping_strategy"].dropna().unique())
        strat = track("cfg_mc_strat", st.sidebar.selectbox(
            "Damping strategy", ["(all)"] + strats, key="cfg_mc_strat"))
        reference = track("cfg_mc_ref", st.sidebar.selectbox(
            "Reference (vs)", ["exact", "gnh"], key="cfg_mc_ref"))
        table = D.method_metric_table_sweep(
            df, model=model, reference=reference, sampling=sampling,
            damping_strategy=None if strat == "(all)" else strat)
        if not table.empty:
            point_methods = table.index.get_level_values("approximator")
            rid_to_damp = df.drop_duplicates("result_id").set_index("result_id")["damping_value"]
            sweep_damping = rid_to_damp.reindex(
                table.index.get_level_values("result_id")).to_numpy()
        else:
            point_methods, sweep_damping = [], None
        annotate_default = False

    if table.empty or len(table.columns) < 2:
        st.warning("Need at least two metric axes for a correlation.")
        st.stop()

    axes = list(table.columns)
    methods_present = D.order_methods(
        list(table.index if point_methods is None else dict.fromkeys(point_methods)))

    def _axis_select(label, key, default_axis):
        idx = axes.index(default_axis) if default_axis in axes else 0
        return track(key, st.sidebar.selectbox(
            label, axes, index=idx, format_func=P.axis_label, key=key))

    x_axis = _axis_select("X axis", "cfg_mc_x", axes[0])
    y_axis = _axis_select("Y axis", "cfg_mc_y", D.LDS_AXIS)
    log_x = track("cfg_mc_logx", st.sidebar.checkbox("Log X", value=True, key="cfg_mc_logx"))
    log_y = track("cfg_mc_logy", st.sidebar.checkbox("Log Y", value=True, key="cfg_mc_logy"))
    # Per-point labels only make sense for the single-result scope (one point per
    # method); across a sweep there are thousands of points, so colour+legend
    # carry the method identity instead (Fig 1b style).
    if scope == "Single result":
        annotate = track("cfg_mc_annot", st.sidebar.checkbox(
            "Label points", value=annotate_default, key="cfg_mc_annot"))
    else:
        annotate = False
    points = track("cfg_mc_points", st.sidebar.multiselect(
        "Methods to include", methods_present, default=methods_present,
        format_func=lambda m: P.LABELS.get(m, m), key="cfg_mc_points"))
    if len(points) < 1:
        st.info("Include at least one method.")
        st.stop()

    fig, stats = P.plot_metric_correlation(
        table, x_axis, y_axis, point_methods=point_methods, point_damping=sweep_damping,
        include=points, log_x=log_x, log_y=log_y, annotate=annotate)
    show(fig)
    st.caption(
        f"Pearson r = {stats['pearson']:.3f} · Spearman ρ = {stats['spearman']:.3f} "
        f"· n = {stats['n']} points"
    )


# ──────────────────────────────────────────────────────────────────────
# Influence Spearman
# ──────────────────────────────────────────────────────────────────────
elif family == "Influence Spearman":
    st.header("Influence Spearman")
    mode = track("cfg_sp_mode", st.sidebar.radio(
        "Mode", ["Single result (methods)", "Across swept axis"], key="cfg_sp_mode"))
    aggregate = track("cfg_sp_agg", st.sidebar.selectbox(
        "Aggregate over queries", ["mean", "median"], key="cfg_sp_agg"))
    annotate = track("cfg_sp_annot", st.sidebar.checkbox(
        "Annotate cells", value=True, key="cfg_sp_annot"))

    if mode == "Single result (methods)":
        iopts = opts[opts["result_id"].isin(infl_rids)].copy()
        if iopts.empty:
            st.warning("No results with influence vectors.")
            st.stop()
        rid, row = pick_result_id(iopts, key="sp")
        paths = D.influence_paths_for_result(df, rid)
        avail = D.order_methods(list(paths))
        msel = track("cfg_sp_methods", st.sidebar.multiselect(
            "Methods", avail, default=avail, key="cfg_sp_methods"))
        if len(msel) < 2:
            st.info("Pick at least two methods.")
            st.stop()
        with st.spinner("Computing pairwise Spearman …"):
            rho = D.compute_influence_spearman_matrix(
                rid, paths, methods=msel, aggregate=aggregate)
            fig, _ = P.plot_influence_spearman(
                rho, annotate=annotate, title=D.result_label(row))
        show(fig)
        st.dataframe(rho.round(3))

    else:  # Across swept axis
        infl = df[df["npy_path"].notna()]
        model = track("cfg_sx_model", st.sidebar.selectbox(
            "Model", sorted(infl["model_id"].unique()), key="cfg_sx_model"))
        minfl = infl[infl["model_id"] == model]
        sampling = track("cfg_sx_pts", st.sidebar.selectbox(
            "Sampling (pseudo-target)",
            sorted(minfl["pseudo_target_strategy"].dropna().unique()), key="cfg_sx_pts"))
        minfl = minfl[minfl["pseudo_target_strategy"] == sampling]
        method = track("cfg_sx_method", st.sidebar.selectbox(
            "Influence method to track",
            D.order_methods(minfl["approximator"].dropna().unique().tolist()),
            key="cfg_sx_method",
        ))
        strategies = sorted(minfl["damping_strategy"].dropna().unique())
        strat = track("cfg_sx_strat", st.sidebar.selectbox(
            "Damping strategy", strategies, key="cfg_sx_strat"))
        sweep = track("cfg_sx_sweep", st.sidebar.radio(
            "Sweep axis", ["damping", "epoch"], key="cfg_sx_sweep"))
        fix_col = "epoch" if sweep == "damping" else "damping_value"
        fix_vals = sorted(
            minfl[minfl["damping_strategy"] == strat][fix_col].dropna().unique())
        grid = track("cfg_sx_grid", st.sidebar.checkbox(
            "Grid (one panel per fixed value)", value=False, key="cfg_sx_grid"))

        try:
            if grid:
                chosen = track("cfg_sx_fixvals", st.sidebar.multiselect(
                    f"Fixed {fix_col} values", fix_vals, default=fix_vals,
                    format_func=fmt, key="cfg_sx_fixvals"))
                if not chosen:
                    st.info("Pick at least one fixed value.")
                    st.stop()
                with st.spinner("Computing Spearman grid …"):
                    fig, skipped = P.plot_influence_corr_grid(
                        df, model_id=model, sweep=sweep, fix_values=chosen,
                        method=method, aggregate=aggregate, annotate=annotate,
                        damping_strategy=strat, pseudo_target_strategy=sampling,
                    )
                show(fig)
                if skipped:
                    st.caption("Skipped panels: " + ", ".join(f"{fix_col}={fv}" for fv, _ in skipped))
            else:
                fv = track("cfg_sx_fix", st.sidebar.selectbox(
                    f"Fixed {fix_col}", fix_vals, format_func=fmt, key="cfg_sx_fix"))
                with st.spinner("Computing Spearman …"):
                    fig, mat = P.plot_influence_corr_across_axis(
                        df, model_id=model, sweep=sweep, fix=fv,
                        method=method, aggregate=aggregate, annotate=annotate,
                        damping_strategy=strat, pseudo_target_strategy=sampling,
                    )
                show(fig)
                st.dataframe(mat.round(3))
        except ValueError as e:
            st.warning(str(e))


# ──────────────────────────────────────────────────────────────────────
# Factor eigenvalues
# ──────────────────────────────────────────────────────────────────────
elif family == "Factor eigenvalues":
    st.header("Factor eigenvalues")
    default_root = str(Path(db_path).parent / "models")
    root = track("cfg_fe_root", st.sidebar.text_input(
        "Factor search root", value=default_root, key="cfg_fe_root"))
    dirs = get_factor_dirs(root)
    if not dirs:
        st.warning(f"No factor dirs (blocks.npz) under {root}")
        st.stop()

    st.sidebar.caption(f"{len(dirs)} factor dirs found")
    rootp = Path(root)
    recs = D.parse_factor_dirs(dirs, rootp)

    if recs:  # cascade dataset → model → epoch → sampling → method
        r = recs
        ds = track("cfg_fe_dataset", st.sidebar.selectbox(
            "Dataset", sorted({x["dataset"] for x in r}), key="cfg_fe_dataset"))
        r = [x for x in r if x["dataset"] == ds]
        model = track("cfg_fe_model", st.sidebar.selectbox(
            "Model", sorted({x["model"] for x in r}), key="cfg_fe_model"))
        r = [x for x in r if x["model"] == model]
        epoch = track("cfg_fe_epoch", st.sidebar.selectbox(
            "Epoch", sorted({x["epoch"] for x in r}), key="cfg_fe_epoch"))
        r = [x for x in r if x["epoch"] == epoch]
        sampling = track("cfg_fe_pts", st.sidebar.selectbox(
            "Sampling (pseudo-target)", sorted({x["sampling"] for x in r}), key="cfg_fe_pts"))
        r = [x for x in r if x["sampling"] == sampling]
        methods = D.order_methods(sorted({x["method"] for x in r}))
        meth = track("cfg_fe_method", st.sidebar.selectbox(
            "Method", methods, format_func=lambda m: P.LABELS.get(m, m), key="cfg_fe_method"))
        factor_dir = next(x["path"] for x in r if x["method"] == meth)
    else:  # unparseable layout — fall back to raw path picker
        factor_dir = track("cfg_fe_dir", st.sidebar.selectbox(
            "Factor directory", dirs,
            format_func=lambda d: str(Path(d).relative_to(rootp))
            if str(d).startswith(str(rootp)) else d,
            key="cfg_fe_dir",
        ))
    try:
        eigs = D.load_factor_eigenvalues(factor_dir)
        layers = list(eigs)
        lsel = track("cfg_fe_layers", st.sidebar.multiselect(
            "Layers", layers, default=layers, key="cfg_fe_layers"))
        log_x = track("cfg_fe_logx", st.sidebar.checkbox(
            "Log x-axis (rank)", value=False, key="cfg_fe_logx"))
        if not lsel:
            st.info("Pick at least one layer.")
            st.stop()
        with st.spinner("Computing eigenvalue spectrum …"):
            fig, _ = P.plot_factor_eigenvalues(factor_dir, layers=lsel, log_x=log_x)
        show(fig)
    except Exception as e:  # noqa: BLE001 - surface any load/parse error to the UI
        st.error(f"{type(e).__name__}: {e}")


# ── Config hash: copy to save, paste to jump to a config ──────────────
st.sidebar.divider()
with st.sidebar.expander("🔗 Config hash", expanded=True):
    st.caption("Copy to save this exact config; paste one below to jump to it.")
    st.code(config_hash(), language=None)
    load = st.text_input("Load a hash", key="_load_hash").strip()
    if st.button("Go", key="_go_hash") and load:
        reg = _load_registry()
        if load in reg:
            st.session_state["_pending_cfg"] = reg[load]
            st.rerun()
        else:
            st.error("Unknown hash (not saved on this machine).")
