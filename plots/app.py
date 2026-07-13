"""Interactive explorer for runs.db — reproduce the notebook plots with any
hyperparameters you like.

Thin Streamlit UI: widgets gather a config, then `hessian_data` (D) slices/
resolves it and `hessian_plots` (P) draws it.

Run with:
    streamlit run plots/app.py
    # or point at a different DB:
    streamlit run plots/app.py -- --db /path/to/runs.db
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_sortables import sort_items

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_code as C  # noqa: E402
import hessian_data as D  # noqa: E402
import hessian_plots as P  # noqa: E402

st.set_page_config(page_title="runs.db explorer", layout="wide")

# widget option lists shared with the codec (see config_code.SCHEMA)
V_MF, V_DF, V_EF, V_HM = C.LDS_VARIANTS
SCOPE_SINGLE, SCOPE_SWEEP = C.MC_SCOPES
MODE_SINGLE, MODE_AXIS, MODE_COMPARE = C.SP_MODES

# ── Config codes ──────────────────────────────────────────────────────
# Every config-defining widget records its value into CFG via track(); at the
# end `config_code.encode(CFG)` renders a compact reversible code (family
# symbol + per-widget fields), so a pasted code restores the exact selections
# on any machine — no registry. Widget keys are all prefixed "cfg_".
CFG: dict = {}


def track(key: str, value):
    CFG[key] = value
    return value


def _coerce(v):
    # defensive: category pairs must be tuples (list-of-[ct,metric] -> tuples)
    if isinstance(v, list):
        return [tuple(x) if isinstance(x, list) else x for x in v]
    return v


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
        return {
            int(r[0])
            for r in con.execute(f"SELECT DISTINCT result_id FROM {table}").fetchall()
        }


@st.cache_data(show_spinner=False)
def get_factor_dirs(root: str) -> list[str]:
    return D.find_factor_dirs(root)


# running counter so each download_button gets a unique key across a rerun
_dl_counter = [0]


def _plot_basename() -> str:
    """Filename stem for plot downloads: the config code (the "hash") for the
    current selections, sanitised to filesystem-safe chars. Falls back to the
    plot family when the config can't be encoded yet."""
    try:
        code = C.encode(CFG, db_default=_default_db())
    except Exception:  # noqa: BLE001 - any encode failure → family fallback
        code = str(CFG.get("cfg_family", "plot"))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", code).strip("_")
    return safe or "plot"


def show(fig, *, suffix: str = "", download: bool = True):
    """Render a matplotlib figure, offer PNG/PDF downloads (named by config
    code), then free it."""
    st.pyplot(fig, width="content")
    if download:
        base = _plot_basename()
        name = f"{base}_{suffix}" if suffix else base
        c1, c2, _ = st.columns([1, 1, 6])
        for col, ext, mime in ((c1, "png", "image/png"), (c2, "pdf", "application/pdf")):
            buf = io.BytesIO()
            fig.savefig(buf, format=ext, dpi=200, bbox_inches="tight")
            _dl_counter[0] += 1
            col.download_button(
                f"⬇ {ext.upper()}", data=buf.getvalue(),
                file_name=f"{name}.{ext}", mime=mime,
                key=f"dl_{_dl_counter[0]}", use_container_width=True,
            )
    plt.close(fig)


def fmt(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


# compact chips for the reorder strip (small font / tight padding)
_CHIP_STYLE = """
.sortable-item {
  font-size: 0.74rem; padding: 1px 7px; margin: 2px; border-radius: 10px;
}
"""


def methods_order(label, all_methods, *, state_key):
    """Sidebar `st.multiselect` for membership; resolves the current order (prior
    drag order for still-selected methods, newly added appended). The draggable
    strip itself is rendered separately under the plot via `reorder_strip`.
    Returns the ordered method list used for the plot."""
    disp = lambda m: P.LABELS.get(m, m)
    ms_key = state_key + "_sel"
    kw = {} if ms_key in st.session_state else {"default": all_methods}
    sel = track(
        ms_key,
        st.sidebar.multiselect(label, all_methods, format_func=disp, key=ms_key, **kw),
    )
    prev = [m for m in st.session_state.get(state_key, []) if m in sel]
    order = prev + [m for m in sel if m not in prev]
    st.session_state[state_key] = order
    return track(state_key, order)


def reorder_strip(all_methods, *, state_key, container=st):
    """Compact draggable chip strip to reorder the selected methods. Renders into
    `container` (a placeholder positioned under the plot) and **returns** the
    resolved order, also writing it to st.session_state[state_key].

    Call this BEFORE drawing the plot and feed its return into the plot: the
    drag value is only known once `sort_items` runs, so reading it after the plot
    leaves the plot one rerun stale (the "drag twice" bug)."""
    order = [m for m in st.session_state.get(state_key, []) if m in all_methods]
    if len(order) <= 1:
        return order
    disp = lambda m: P.LABELS.get(m, m)
    inv = {disp(m): m for m in all_methods}
    target = st.container() if container is st else container
    target.caption("drag to reorder methods")
    # `sort_items` with key=None re-mounts (losing its drag state, falling back to
    # `default=items`) whenever `items` changes — i.e. on every reorder, which
    # eats the very drag that triggered the rerun. So we give it a stable key and
    # only force a re-mount (via a bumped generation counter) when `order` was
    # changed by something *other* than this widget — a config-code load, or an
    # add/remove. That keeps drag state across reorders while still refreshing the
    # chips when the order is set externally (config codes encode method order).
    gen_key, last_key = state_key + "_sortgen", state_key + "_sortlast"
    gen = st.session_state.get(gen_key, 0)
    last = st.session_state.get(last_key)
    if last is not None and last != order:  # external change since our last render
        gen += 1
        st.session_state[gen_key] = gen
    sort_key = f"{state_key}::gen{gen}"
    with target:
        res = sort_items(
            [disp(m) for m in order], direction="horizontal",
            custom_style=_CHIP_STYLE, key=sort_key,
        )
    try:
        new_order = [inv[d] for d in res]
    except (TypeError, KeyError):  # component returned nothing (e.g. AppTest)
        new_order = order
    st.session_state[state_key] = new_order
    st.session_state[last_key] = new_order
    return new_order


def pick_result_id(pool, *, key: str):
    """Cascading Model → Epoch → λ → strategy → sampling selectors that resolve
    to one result_id from `pool`. Returns (result_id, row)."""
    model = track(
        f"cfg_{key}_model",
        st.sidebar.selectbox(
            "Model", sorted(pool["model_id"].unique()),
            format_func=P.model_label, key=f"cfg_{key}_model"
        ),
    )
    p = pool[pool["model_id"] == model]
    epoch = track(
        f"cfg_{key}_epoch",
        st.sidebar.selectbox(
            "Epoch", sorted(p["epoch"].unique()), key=f"cfg_{key}_epoch"
        ),
    )
    p = p[p["epoch"] == epoch]
    lam = track(
        f"cfg_{key}_lam",
        st.sidebar.selectbox(
            "Damping λ",
            sorted(p["damping_value"].unique()),
            format_func=fmt,
            key=f"cfg_{key}_lam",
        ),
    )
    p = p[p["damping_value"] == lam]
    strat = track(
        f"cfg_{key}_strat",
        st.sidebar.selectbox(
            "Damping strategy",
            sorted(p["damping_strategy"].unique()),
            key=f"cfg_{key}_strat",
        ),
    )
    p = p[p["damping_strategy"] == strat]
    sampling = track(
        f"cfg_{key}_pts",
        st.sidebar.selectbox(
            "Sampling (pseudo-target)",
            sorted(p["pseudo_target_strategy"].unique()),
            key=f"cfg_{key}_pts",
        ),
    )
    rid, row, n = D.resolve_result(
        pool, model=model, epoch=epoch, lam=lam, strat=strat, sampling=sampling
    )
    if n > 1:
        st.sidebar.caption(f"{n} runs match — using latest ({row['run_id']})")
    return rid, row


# ── Sidebar: DB + family ──────────────────────────────────────────────
st.sidebar.title("runs.db explorer")
db_path = track(
    "cfg_db", st.sidebar.text_input("Database path", value=_default_db(), key="cfg_db")
)
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

family = track(
    "cfg_family",
    st.sidebar.radio(
        "Plot family",
        C.FAMILIES,
        key="cfg_family",
    ),
)
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

    model = track(
        "cfg_lds_model",
        st.sidebar.selectbox("Model", models, format_func=P.model_label, key="cfg_lds_model"),
    )
    mlds = lds[lds["model_id"] == model]
    sampling = track(
        "cfg_lds_pts",
        st.sidebar.selectbox(
            "Sampling (pseudo-target)",
            sorted(mlds["pseudo_target_strategy"].dropna().unique()),
            key="cfg_lds_pts",
        ),
    )
    strategies = sorted(
        mlds[mlds["pseudo_target_strategy"] == sampling]["damping_strategy"]
        .dropna()
        .unique()
    )
    strat = track(
        "cfg_lds_strat",
        st.sidebar.selectbox(
            "Damping strategy", strategies, key="cfg_lds_strat"
        ),
    )
    # Collector subset size is a real hyperparameter, not something to average
    # over: mcmc sweeps several sizes while all_classes only has the full set,
    # so averaging makes mcmc look worse than all_classes for the same method.
    sub_pool = mlds[mlds["pseudo_target_strategy"] == sampling]
    if strat != "(all)":
        sub_pool = sub_pool[sub_pool["damping_strategy"] == strat]
    subsets = sorted(sub_pool["collector_subset_size"].dropna().unique())
    subset = track(
        "cfg_lds_subset",
        st.sidebar.selectbox(
            "Collector subset size",
            subsets,
            index=len(subsets) - 1,  # default to the full set (largest)
            format_func=lambda s: f"{int(s):,}",
            key="cfg_lds_subset",
        ),
    )

    df_sub = D.slice_lds(
        df, model=model, sampling=sampling,
        strategy=None if strat == "(all)" else strat,
        subset_size=subset,
    )

    variant = track(
        "cfg_lds_variant",
        st.sidebar.radio(
            "Variant",
            C.LDS_VARIANTS,
            key="cfg_lds_variant",
        ),
    )
    is_heatmap = variant == V_HM
    if not is_heatmap:
        kind = (
            "bar"
            if track(
                "cfg_lds_style",
                st.sidebar.radio("Style", C.LDS_STYLES, key="cfg_lds_style"),
            )
            == C.LDS_STYLES[1]
            else "line"
        )
    else:
        kind = "line"
    show_band = track(
        "cfg_lds_band",
        st.sidebar.checkbox(
            "Show CI"
            + ("" if is_heatmap else " (error bars)" if kind == "bar" else " band"),
            value=True,
            key="cfg_lds_band",
        ),
    )
    annotate = track(
        "cfg_lds_annot",
        st.sidebar.checkbox("Annotate heatmap", value=True, key="cfg_lds_annot"),
    )

    # General axis focus — optionally restrict any free axis to a subset of its
    # values (empty = keep all). Applied to df_sub before the variant selectors,
    # so it scopes every variant uniformly: e.g. pin the "Fix epoch" view to a
    # single λ, or zoom "Fix method" to a few epochs. Method already has a
    # subset control in every variant, so only epoch/λ need this here.
    pool_epochs = sorted(int(e) for e in df_sub["epoch"].dropna().unique())
    pool_damps = sorted(df_sub["damping_value"].dropna().unique())
    keep_epochs = track(
        "cfg_lds_keep_epochs",
        st.sidebar.multiselect(
            "Focus epochs (empty = all)", pool_epochs, key="cfg_lds_keep_epochs"
        ),
    )
    keep_damps = track(
        "cfg_lds_keep_damps",
        st.sidebar.multiselect(
            "Focus λ (empty = all)", pool_damps, format_func=fmt,
            key="cfg_lds_keep_damps",
        ),
    )
    if keep_epochs:
        df_sub = df_sub[df_sub["epoch"].isin(keep_epochs)]
    if keep_damps:
        df_sub = df_sub[df_sub["damping_value"].isin(keep_damps)]

    methods_here = D.order_methods(df_sub["approximator"].dropna().unique().tolist())
    epochs = sorted(int(e) for e in df_sub["epoch"].dropna().unique())
    dampings = sorted(df_sub["damping_value"].dropna().unique())

    st.caption(
        f"{len(df_sub)} LDS rows · model `{model}` · sampling `{sampling}` · "
        f"strategy `{strat}` · subset `{int(subset):,}`"
    )

    if variant == V_MF:
        sel = track(
            "cfg_lds_mf_methods",
            st.sidebar.multiselect(
                "Method(s)",
                methods_here,
                default=methods_here[:1],
                key="cfg_lds_mf_methods",
            ),
        )
        for m in sel:
            fig = P.plot_lds_methodfix(df_sub, m, show_band=show_band, kind=kind)
            fig.suptitle(P.model_label(model), fontsize=10)
            show(fig, suffix=m)

    elif variant == V_DF:
        d = track(
            "cfg_lds_df_lam",
            st.sidebar.selectbox(
                "Damping λ", dampings, format_func=fmt, key="cfg_lds_df_lam"
            ),
        )
        msel = track(
            "cfg_lds_df_methods",
            st.sidebar.multiselect(
                "Methods", methods_here, default=methods_here, key="cfg_lds_df_methods"
            ),
        )
        fig = P.plot_lds_dampingfix(
            df_sub,
            d,
            methods=msel or None,
            show_band=show_band,
            kind=kind,
            strategy=None if strat == "(all)" else strat,
        )
        fig.suptitle(P.model_label(model), fontsize=10)
        show(fig)

    elif variant == V_EF:
        e = track(
            "cfg_lds_ef_epoch",
            st.sidebar.selectbox("Epoch", epochs, key="cfg_lds_ef_epoch"),
        )
        msel = track(
            "cfg_lds_ef_methods",
            st.sidebar.multiselect(
                "Methods", methods_here, default=methods_here, key="cfg_lds_ef_methods"
            ),
        )
        fig = P.plot_lds_epochfix(
            df_sub,
            e,
            methods=msel or None,
            show_band=show_band,
            kind=kind,
            strategy=None if strat == "(all)" else strat,
        )
        fig.suptitle(P.model_label(model), fontsize=10)
        show(fig)

    else:  # Heatmap per method
        sel = track(
            "cfg_lds_hm_methods",
            st.sidebar.multiselect(
                "Method(s)",
                methods_here,
                default=methods_here[:1],
                key="cfg_lds_hm_methods",
            ),
        )
        for m in sel:
            fig, _ = P.plot_lds_heatmap_for_method(df_sub, m, annotate=annotate)
            fig.suptitle(P.model_label(model), fontsize=10)
            show(fig, suffix=m)


# ──────────────────────────────────────────────────────────────────────
# Sample-size Pareto (num samples × LDS)
# ──────────────────────────────────────────────────────────────────────
elif family == "Sample-size Pareto":
    st.header("Sample-size Pareto: num samples × LDS")
    lds = df[df["lds_mean"].notna()]
    models = sorted(lds["model_id"].dropna().unique())
    if not models:
        st.warning("No LDS rows in this DB.")
        st.stop()

    model = track(
        "cfg_pareto_model",
        st.sidebar.selectbox("Model", models, format_func=P.model_label,
                             key="cfg_pareto_model"),
    )
    mlds = lds[lds["model_id"] == model]
    strategies = sorted(mlds["damping_strategy"].dropna().unique())
    strat = track(
        "cfg_pareto_strat",
        st.sidebar.selectbox(
            "Damping strategy", strategies, key="cfg_pareto_strat"
        ),
    )
    tsub = mlds if strat == "(all)" else mlds[mlds["damping_strategy"] == strat]
    epochs = sorted(int(e) for e in tsub["epoch"].dropna().unique())
    epoch = track(
        "cfg_pareto_epoch",
        st.sidebar.selectbox("Epoch", epochs, key="cfg_pareto_epoch"),
    )
    esub = tsub[tsub["epoch"] == epoch]
    dampings = sorted(esub["damping_value"].dropna().unique())
    lam = track(
        "cfg_pareto_lam",
        st.sidebar.selectbox(
            "Damping λ", dampings, format_func=fmt, key="cfg_pareto_lam"
        ),
    )

    # logit-space dimension → all_classes x-axis multiplier (per model)
    num_classes = D.model_output_dim(Path(db_path), model)
    df_sub = D.slice_pareto(
        df,
        model=model,
        num_classes=num_classes,
        strategy=None if strat == "(all)" else strat,
        epoch=epoch,
        damping=lam,
    )
    methods_here = D.order_methods(df_sub["approximator"].dropna().unique().tolist())
    msel = track(
        "cfg_pareto_methods",
        st.sidebar.multiselect(
            "Methods",
            methods_here,
            default=methods_here,
            format_func=lambda m: P.LABELS.get(m, m),
            key="cfg_pareto_methods",
        ),
    )
    show_band = track(
        "cfg_pareto_band",
        st.sidebar.checkbox("Show CI (error bars)", value=True, key="cfg_pareto_band"),
    )
    log_x = track(
        "cfg_pareto_logx",
        st.sidebar.checkbox("Log x (effective samples)", value=True, key="cfg_pareto_logx"),
    )

    sizes = sorted(int(s) for s in df_sub["collector_subset_size"].dropna().unique())
    samps = sorted(df_sub["pseudo_target_strategy"].dropna().unique())
    st.caption(
        f"{len(df_sub)} points · {len(sizes)} sample size(s): "
        f"{', '.join(str(s) for s in sizes) or '—'} · samplings: {', '.join(samps) or '—'} "
        f"· logit dim (all_classes ×): {num_classes} · E-* methods ×2"
    )
    if len(sizes) < 2:
        st.info(
            "Only one `collector_subset_size` present for this config — the Pareto "
            "axis needs ≥2 sample sizes. Run the sample-size sweep to populate it."
        )
    fig = P.plot_lds_pareto(
        df_sub, methods=msel or None, show_band=show_band, log_x=log_x
    )
    fig.suptitle(f"{P.model_label(model)} · ep{epoch} · λ={lam:g} · {strat}", fontsize=10)
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
    reference = track(
        "cfg_mb_ref",
        st.sidebar.selectbox("Reference (vs)", C.REFERENCES, key="cfg_mb_ref"),
    )
    approxs = methods_order(
        "Approximators (bar order)", ALL_METHODS, state_key="cfg_mb_approxs"
    )

    cat_opts = D.metric_category_options(
        int(rid), reference=reference, db_path=Path(db_path)
    )
    if not cat_opts:
        st.warning(f"No metrics for reference={reference} on this result.")
        st.stop()
    cats = track(
        "cfg_mb_cats",
        st.sidebar.multiselect(
            "Categories (empty = all)",
            cat_opts,
            default=[],
            format_func=P.category_label,
            key="cfg_mb_cats",
        ),
    )
    # Plot above, drag strip below — but resolve the drag order (reorder_strip)
    # BEFORE drawing so a drag takes effect on the same rerun.
    plot_box, strip_box = st.container(), st.container()
    approxs = reorder_strip(ALL_METHODS, state_key="cfg_mb_approxs", container=strip_box)
    try:
        with plot_box:
            fig = P.plot_metrics_for(
                int(rid),
                reference=reference,
                approxs=approxs or None,
                categories=cats or None,
                db_path=Path(db_path),
            )
            show(fig)
    except ValueError as e:
        plot_box.warning(str(e))


# ──────────────────────────────────────────────────────────────────────
# Metric correlation (one point per method)
# ──────────────────────────────────────────────────────────────────────
elif family == "Metric correlation":
    st.header("Metric correlation")
    scope = track(
        "cfg_mc_scope", st.sidebar.radio("Scope", C.MC_SCOPES, key="cfg_mc_scope")
    )

    if scope == SCOPE_SINGLE:
        mopts = opts[opts["result_id"].isin(metric_rids)].copy()
        if mopts.empty:
            st.warning("No results with metrics in this DB.")
            st.stop()
        rid, _row = pick_result_id(mopts, key="mc")
        reference = track(
            "cfg_mc_ref",
            st.sidebar.selectbox("Reference (vs)", C.REFERENCES, key="cfg_mc_ref"),
        )
        table = D.method_metric_table(
            int(rid), df, reference=reference, db_path=Path(db_path)
        )
        point_methods = None  # index is the method
        sweep_damping = None  # single config → colour by method, not λ
        annotate_default = True
    else:  # Across sweep — one point per (config, method)
        mopts = opts[opts["result_id"].isin(metric_rids)]
        model = track(
            "cfg_mc_model",
            st.sidebar.selectbox(
                "Model", sorted(mopts["model_id"].unique()),
                format_func=P.model_label, key="cfg_mc_model"
            ),
        )
        msub = mopts[mopts["model_id"] == model]
        sampling = track(
            "cfg_mc_pts",
            st.sidebar.selectbox(
                "Sampling (pseudo-target)",
                sorted(msub["pseudo_target_strategy"].dropna().unique()),
                key="cfg_mc_pts",
            ),
        )
        strats = sorted(
            msub[msub["pseudo_target_strategy"] == sampling]["damping_strategy"]
            .dropna()
            .unique()
        )
        strat = track(
            "cfg_mc_strat",
            st.sidebar.selectbox(
                "Damping strategy", strats, key="cfg_mc_strat"
            ),
        )
        reference = track(
            "cfg_mc_ref",
            st.sidebar.selectbox("Reference (vs)", C.REFERENCES, key="cfg_mc_ref"),
        )
        table = D.method_metric_table_sweep(
            df,
            model=model,
            reference=reference,
            sampling=sampling,
            damping_strategy=None if strat == "(all)" else strat,
        )
        # Optionally fix / focus the sweep to a subset of damping values (empty =
        # all λ). Picking a single λ isolates the epoch × method variation; the
        # colour axis (λ) then collapses to one shade while shape still marks the
        # method. Options come from the table so only realised λ appear.
        if not table.empty:
            damp_opts = sorted(table.index.get_level_values("damping_value").unique())
            keep_damps = track(
                "cfg_mc_keep_damps",
                st.sidebar.multiselect(
                    "Focus λ (empty = all)", damp_opts, format_func=fmt,
                    key="cfg_mc_keep_damps",
                ),
            )
            if keep_damps:
                table = table[
                    table.index.get_level_values("damping_value").isin(keep_damps)
                ]
        if not table.empty:
            point_methods = table.index.get_level_values("approximator")
            sweep_damping = table.index.get_level_values("damping_value").to_numpy()
        else:
            point_methods, sweep_damping = [], None
        annotate_default = False

    if table.empty or len(table.columns) < 2:
        st.warning("Need at least two metric axes for a correlation.")
        st.stop()

    axes = list(table.columns)
    methods_present = D.order_methods(
        list(table.index if point_methods is None else dict.fromkeys(point_methods))
    )

    def _axis_select(label, key, default_axis):
        idx = axes.index(default_axis) if default_axis in axes else 0
        return track(
            key,
            st.sidebar.selectbox(
                label, axes, index=idx, format_func=P.axis_label, key=key
            ),
        )

    x_axis = _axis_select("X axis", "cfg_mc_x", axes[0])
    y_axis = _axis_select("Y axis", "cfg_mc_y", D.LDS_AXIS)
    log_x = track(
        "cfg_mc_logx", st.sidebar.checkbox("Log X", value=True, key="cfg_mc_logx")
    )
    log_y = track(
        "cfg_mc_logy", st.sidebar.checkbox("Log Y", value=True, key="cfg_mc_logy")
    )
    # Per-point labels only make sense for the single-result scope (one point per
    # method); across a sweep there are thousands of points, so colour+legend
    # carry the method identity instead (Fig 1b style).
    if scope == SCOPE_SINGLE:
        annotate = track(
            "cfg_mc_annot",
            st.sidebar.checkbox(
                "Label points", value=annotate_default, key="cfg_mc_annot"
            ),
        )
    else:
        annotate = False
    points = track(
        "cfg_mc_points",
        st.sidebar.multiselect(
            "Methods to include",
            methods_present,
            default=methods_present,
            format_func=lambda m: P.LABELS.get(m, m),
            key="cfg_mc_points",
        ),
    )
    if len(points) < 1:
        st.info("Include at least one method.")
        st.stop()

    fig, stats = P.plot_metric_correlation(
        table,
        x_axis,
        y_axis,
        point_methods=point_methods,
        point_damping=sweep_damping,
        include=points,
        log_x=log_x,
        log_y=log_y,
        annotate=annotate,
    )
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
    mode = track("cfg_sp_mode", st.sidebar.radio("Mode", C.SP_MODES, key="cfg_sp_mode"))
    aggregate = track(
        "cfg_sp_agg",
        st.sidebar.selectbox("Aggregate over queries", C.SP_AGGS, key="cfg_sp_agg"),
    )
    annotate = track(
        "cfg_sp_annot",
        st.sidebar.checkbox("Annotate cells", value=True, key="cfg_sp_annot"),
    )

    if mode == MODE_SINGLE:
        iopts = opts[opts["result_id"].isin(infl_rids)].copy()
        if iopts.empty:
            st.warning("No results with influence vectors.")
            st.stop()
        rid, row = pick_result_id(iopts, key="sp")
        paths = D.influence_paths_for_result(df, rid)
        avail = D.order_methods(list(paths))
        methods_order("Methods", avail, state_key="cfg_sp_methods")  # membership (sidebar)
        # Result above, drag strip below — resolve drag order before computing so
        # a reorder takes effect on the same rerun (not one drag late).
        result_box, strip_box = st.container(), st.container()
        msel = reorder_strip(avail, state_key="cfg_sp_methods", container=strip_box)
        if len(msel) < 2:
            result_box.info("Pick at least two methods.")
            st.stop()
        with result_box:
            with st.spinner("Computing pairwise Spearman …"):
                rho = D.compute_influence_spearman_matrix(
                    rid, paths, methods=msel, aggregate=aggregate
                )
                fig, _ = P.plot_influence_spearman(
                    rho, annotate=annotate, title=D.result_label(row)
                )
            show(fig)
            st.dataframe(rho.round(3))

    elif mode == MODE_AXIS:
        infl = df[df["npy_path"].notna()]
        model = track(
            "cfg_sx_model",
            st.sidebar.selectbox(
                "Model", sorted(infl["model_id"].unique()),
                format_func=P.model_label, key="cfg_sx_model"
            ),
        )
        minfl = infl[infl["model_id"] == model]
        sampling = track(
            "cfg_sx_pts",
            st.sidebar.selectbox(
                "Sampling (pseudo-target)",
                sorted(minfl["pseudo_target_strategy"].dropna().unique()),
                key="cfg_sx_pts",
            ),
        )
        minfl = minfl[minfl["pseudo_target_strategy"] == sampling]
        method = track(
            "cfg_sx_method",
            st.sidebar.selectbox(
                "Influence method to track",
                D.order_methods(minfl["approximator"].dropna().unique().tolist()),
                key="cfg_sx_method",
            ),
        )
        strategies = sorted(minfl["damping_strategy"].dropna().unique())
        strat = track(
            "cfg_sx_strat",
            st.sidebar.selectbox("Damping strategy", strategies, key="cfg_sx_strat"),
        )
        sweep = track(
            "cfg_sx_sweep",
            st.sidebar.radio("Sweep axis", C.SX_SWEEPS, key="cfg_sx_sweep"),
        )
        fix_col = "epoch" if sweep == C.SX_SWEEPS[0] else "damping_value"
        fix_vals = sorted(
            minfl[minfl["damping_strategy"] == strat][fix_col].dropna().unique()
        )
        grid = track(
            "cfg_sx_grid",
            st.sidebar.checkbox(
                "Grid (one panel per fixed value)", value=False, key="cfg_sx_grid"
            ),
        )

        try:
            if grid:
                chosen = track(
                    "cfg_sx_fixvals",
                    st.sidebar.multiselect(
                        f"Fixed {fix_col} values",
                        fix_vals,
                        default=fix_vals,
                        format_func=fmt,
                        key="cfg_sx_fixvals",
                    ),
                )
                if not chosen:
                    st.info("Pick at least one fixed value.")
                    st.stop()
                with st.spinner("Computing Spearman grid …"):
                    fig, skipped = P.plot_influence_corr_grid(
                        df,
                        model_id=model,
                        sweep=sweep,
                        fix_values=chosen,
                        method=method,
                        aggregate=aggregate,
                        annotate=annotate,
                        damping_strategy=strat,
                        pseudo_target_strategy=sampling,
                    )
                show(fig)
                if skipped:
                    st.caption(
                        "Skipped panels: "
                        + ", ".join(f"{fix_col}={fv}" for fv, _ in skipped)
                    )
            else:
                fv = track(
                    "cfg_sx_fix",
                    st.sidebar.selectbox(
                        f"Fixed {fix_col}", fix_vals, format_func=fmt, key="cfg_sx_fix"
                    ),
                )
                with st.spinner("Computing Spearman …"):
                    fig, mat = P.plot_influence_corr_across_axis(
                        df,
                        model_id=model,
                        sweep=sweep,
                        fix=fv,
                        method=method,
                        aggregate=aggregate,
                        annotate=annotate,
                        damping_strategy=strat,
                        pseudo_target_strategy=sampling,
                    )
                show(fig)
                st.dataframe(mat.round(3))
        except ValueError as e:
            st.warning(str(e))

    else:  # Compare samplings (e.g. mcmc vs all_classes) at a fixed config
        infl = df[df["npy_path"].notna()]
        multi = [
            m
            for m in sorted(infl["model_id"].unique())
            if infl[infl["model_id"] == m]["pseudo_target_strategy"].nunique() >= 2
        ]
        if not multi:
            st.info("No model has ≥2 samplings with influence to compare.")
            st.stop()
        model = track(
            "cfg_cs_model",
            st.sidebar.selectbox("Model", multi, format_func=P.model_label, key="cfg_cs_model"),
        )
        msub = infl[infl["model_id"] == model]
        samps = sorted(msub["pseudo_target_strategy"].dropna().unique())
        a = track(
            "cfg_cs_a",
            st.sidebar.selectbox("Sampling A", samps, index=0, key="cfg_cs_a"),
        )
        b = track(
            "cfg_cs_b",
            st.sidebar.selectbox(
                "Sampling B", samps, index=min(1, len(samps) - 1), key="cfg_cs_b"
            ),
        )
        if a == b:
            st.info("Pick two different samplings.")
            st.stop()
        # restrict the config cascade to (epoch, λ, strategy) present under BOTH
        both = msub.groupby(["epoch", "damping_value", "damping_strategy"]).filter(
            lambda g: {a, b} <= set(g["pseudo_target_strategy"].dropna())
        )
        if both.empty:
            st.warning(f"No config has both {a} and {b} influence for this model.")
            st.stop()
        epoch = track(
            "cfg_cs_epoch",
            st.sidebar.selectbox(
                "Epoch", sorted(both["epoch"].dropna().unique()), key="cfg_cs_epoch"
            ),
        )
        esub = both[both["epoch"] == epoch]
        lam = track(
            "cfg_cs_lam",
            st.sidebar.selectbox(
                "Damping λ",
                sorted(esub["damping_value"].dropna().unique()),
                format_func=fmt,
                key="cfg_cs_lam",
            ),
        )
        lsub = esub[esub["damping_value"] == lam]
        strat = track(
            "cfg_cs_strat",
            st.sidebar.selectbox(
                "Damping strategy",
                sorted(lsub["damping_strategy"].dropna().unique()),
                key="cfg_cs_strat",
            ),
        )
        with st.spinner("Computing per-method sampling agreement …"):
            series = D.sampling_spearman(
                df,
                model=model,
                epoch=epoch,
                damping=lam,
                strategy=strat,
                sampling_a=a,
                sampling_b=b,
                aggregate=aggregate,
            )
        if series.empty:
            st.warning("No methods have both samplings at this config.")
            st.stop()
        fig = P.plot_sampling_comparison(
            series,
            sampling_a=a,
            sampling_b=b,
            title=f"{P.model_label(model)} · ep{epoch} · λ={lam:g} · {strat}\ninfluence agreement: {a} vs {b}",
        )
        show(fig)
        st.dataframe(series.round(3).rename("Spearman ρ"))


# ──────────────────────────────────────────────────────────────────────
# Factor eigenvalues
# ──────────────────────────────────────────────────────────────────────
elif family == "Factor eigenvalues":
    st.header("Factor eigenvalues")
    default_root = str(Path(db_path).parent / "models")
    root = track(
        "cfg_fe_root",
        st.sidebar.text_input(
            "Factor search root", value=default_root, key="cfg_fe_root"
        ),
    )
    dirs = get_factor_dirs(root)
    if not dirs:
        st.warning(f"No factor dirs (blocks.npz) under {root}")
        st.stop()

    st.sidebar.caption(f"{len(dirs)} factor dirs found")
    rootp = Path(root)
    recs = D.parse_factor_dirs(dirs, rootp)

    if recs:  # cascade dataset → model → epoch → sampling → method
        r = recs
        ds = track(
            "cfg_fe_dataset",
            st.sidebar.selectbox(
                "Dataset", sorted({x["dataset"] for x in r}), key="cfg_fe_dataset"
            ),
        )
        r = [x for x in r if x["dataset"] == ds]
        model = track(
            "cfg_fe_model",
            st.sidebar.selectbox(
                "Model", sorted({x["model"] for x in r}),
                format_func=P.model_label, key="cfg_fe_model"
            ),
        )
        r = [x for x in r if x["model"] == model]
        epoch = track(
            "cfg_fe_epoch",
            st.sidebar.selectbox(
                "Epoch", sorted({x["epoch"] for x in r}), key="cfg_fe_epoch"
            ),
        )
        r = [x for x in r if x["epoch"] == epoch]
        sampling = track(
            "cfg_fe_pts",
            st.sidebar.selectbox(
                "Sampling (pseudo-target)",
                sorted({x["sampling"] for x in r}),
                key="cfg_fe_pts",
            ),
        )
        r = [x for x in r if x["sampling"] == sampling]
        methods = D.order_methods(sorted({x["method"] for x in r}))
        meth = track(
            "cfg_fe_method",
            st.sidebar.selectbox(
                "Method",
                methods,
                format_func=lambda m: P.LABELS.get(m, m),
                key="cfg_fe_method",
            ),
        )
        factor_dir = next(x["path"] for x in r if x["method"] == meth)
    else:  # unparseable layout — fall back to raw path picker
        factor_dir = track(
            "cfg_fe_dir",
            st.sidebar.selectbox(
                "Factor directory",
                dirs,
                format_func=lambda d: (
                    str(Path(d).relative_to(rootp))
                    if str(d).startswith(str(rootp))
                    else d
                ),
                key="cfg_fe_dir",
            ),
        )
    try:
        eigs = D.load_factor_eigenvalues(factor_dir)
        layers = list(eigs)
        lsel = track(
            "cfg_fe_layers",
            st.sidebar.multiselect(
                "Layers", layers, default=layers, key="cfg_fe_layers"
            ),
        )
        log_x = track(
            "cfg_fe_logx",
            st.sidebar.checkbox("Log x-axis (rank)", value=False, key="cfg_fe_logx"),
        )
        if not lsel:
            st.info("Pick at least one layer.")
            st.stop()
        with st.spinner("Computing eigenvalue spectrum …"):
            fig, _ = P.plot_factor_eigenvalues(factor_dir, layers=lsel, log_x=log_x)
        show(fig)
    except Exception as e:  # noqa: BLE001 - surface any load/parse error to the UI
        st.error(f"{type(e).__name__}: {e}")


# ── Plot phrase: copy to save, paste to jump to a config ──────────────
st.sidebar.divider()
with st.sidebar.expander("🔗 Plot phrase", expanded=True):
    st.caption("Copy these words to save this exact plot; paste a phrase below to jump to it.")
    try:
        st.code(C.encode(CFG, db_default=_default_db()), language=None)
    except ValueError as e:
        st.error(f"Cannot encode this config: {e}")
    load = st.text_input("Load a phrase", key="_load_code").strip()
    if st.button("Go", key="_go_code") and load:
        try:
            st.session_state["_pending_cfg"] = C.decode(load)
            st.rerun()
        except ValueError as e:
            st.error(f"Invalid plot phrase: {e}")
