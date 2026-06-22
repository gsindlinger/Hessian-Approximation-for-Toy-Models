"""Round-trip and malformed-input tests for plots/config_code.py."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))
import config_code as cc  # noqa: E402
from hessian_data import APPROX_ORDER, LDS_AXIS  # noqa: E402

DB = "/dbs/runs.db"  # explicit db_default so tests are machine-independent
ALL13 = list(APPROX_ORDER)


def enc(cfg):
    return cc.encode(cfg, db_default=DB)


# ── Fixture configs (lifted from the old .app_configs.json registry,
#    plus hand-written ones for branches the registry never saved) ─────
LDS_MF = {
    "cfg_db": DB, "cfg_family": "LDS sweeps",
    "cfg_lds_model": "mlp_08580ee2573a", "cfg_lds_pts": "mcmc",
    "cfg_lds_strat": "auto_mean", "cfg_lds_subset": 3823,
    "cfg_lds_variant": "Fix method (epoch × damping)",
    "cfg_lds_style": "Lines", "cfg_lds_band": True, "cfg_lds_annot": True,
    "cfg_lds_mf_methods": ["exact"],
}
LDS_MF_ALLSTRAT = {**LDS_MF, "cfg_lds_strat": "(all)"}
LDS_DF = {
    "cfg_db": DB, "cfg_family": "LDS sweeps",
    "cfg_lds_model": "mlp_08580ee2573a", "cfg_lds_pts": "mcmc",
    "cfg_lds_strat": "auto_mean", "cfg_lds_subset": 3823,
    "cfg_lds_variant": "Fix damping (epoch × method)",
    "cfg_lds_style": "Lines", "cfg_lds_band": True, "cfg_lds_annot": True,
    "cfg_lds_df_lam": 1e-09, "cfg_lds_df_methods": ALL13,
}
LDS_EF = {
    "cfg_db": DB, "cfg_family": "LDS sweeps",
    "cfg_lds_model": "resnet_mlp_swiglu_d4186381c706", "cfg_lds_pts": "all_classes",
    "cfg_lds_strat": "pseudo_inverse", "cfg_lds_subset": 60000,
    "cfg_lds_variant": "Fix epoch (damping × method)",
    "cfg_lds_style": "Bars", "cfg_lds_band": False, "cfg_lds_annot": True,
    "cfg_lds_ef_epoch": 10, "cfg_lds_ef_methods": ["exact", "gnh"],
}
LDS_HM = {  # heatmap variant: cfg_lds_style is never tracked
    "cfg_db": DB, "cfg_family": "LDS sweeps",
    "cfg_lds_model": "mlp_08580ee2573a", "cfg_lds_pts": "mcmc",
    "cfg_lds_strat": "auto_mean", "cfg_lds_subset": 3823,
    "cfg_lds_variant": "Heatmap per method",
    "cfg_lds_band": True, "cfg_lds_annot": False,
    "cfg_lds_hm_methods": ["exact"],
}
MB_CANON = {
    "cfg_db": DB, "cfg_family": "Metric bars",
    "cfg_mb_model": "mlp_08580ee2573a", "cfg_mb_epoch": 10, "cfg_mb_lam": 1e-09,
    "cfg_mb_strat": "auto_mean", "cfg_mb_pts": "mcmc", "cfg_mb_ref": "exact",
    "cfg_mb_approxs_sel": ALL13, "cfg_mb_approxs": ALL13, "cfg_mb_cats": [],
}
MB_DRAGGED = {
    **MB_CANON,
    "cfg_mb_approxs_sel": ["exact", "gnh", "fim"],
    "cfg_mb_approxs": ["gnh", "exact", "fim"],
    "cfg_mb_cats": [("matrix", "frobenius"), ("hvp", "RELATIVE_ERROR")],
}
MC_SINGLE = {
    "cfg_db": DB, "cfg_family": "Metric correlation",
    "cfg_mc_scope": "Single result",
    "cfg_mc_model": "mlp_08580ee2573a", "cfg_mc_epoch": 10, "cfg_mc_lam": 1e-09,
    "cfg_mc_strat": "auto_mean", "cfg_mc_pts": "mcmc", "cfg_mc_ref": "exact",
    "cfg_mc_x": ("matrix", "frobenius"), "cfg_mc_y": LDS_AXIS,
    "cfg_mc_logx": True, "cfg_mc_logy": True, "cfg_mc_annot": True,
    "cfg_mc_points": ALL13[1:],
}
MC_SWEEP = {  # across-sweep scope: no epoch/lam and no cfg_mc_annot
    "cfg_db": DB, "cfg_family": "Metric correlation",
    "cfg_mc_scope": "Across sweep",
    "cfg_mc_model": "mlp_08580ee2573a", "cfg_mc_pts": "mcmc",
    "cfg_mc_strat": "auto_mean", "cfg_mc_ref": "exact",
    "cfg_mc_x": ("matrix", "frobenius"), "cfg_mc_y": LDS_AXIS,
    "cfg_mc_logx": True, "cfg_mc_logy": True, "cfg_mc_points": ALL13[1:],
}
SP_SINGLE = {
    "cfg_db": DB, "cfg_family": "Influence Spearman",
    "cfg_sp_mode": "Single result (methods)", "cfg_sp_agg": "mean",
    "cfg_sp_annot": True,
    "cfg_sp_model": "mlp_08580ee2573a", "cfg_sp_epoch": 10, "cfg_sp_lam": 1e-09,
    "cfg_sp_strat": "auto_mean", "cfg_sp_pts": "mcmc",
    "cfg_sp_methods_sel": ALL13, "cfg_sp_methods": ALL13,
}
SX_FIX = {
    "cfg_db": DB, "cfg_family": "Influence Spearman",
    "cfg_sp_mode": "Across swept axis", "cfg_sp_agg": "mean", "cfg_sp_annot": True,
    "cfg_sx_model": "mlp_08580ee2573a", "cfg_sx_pts": "mcmc",
    "cfg_sx_method": "exact", "cfg_sx_strat": "auto_mean",
    "cfg_sx_sweep": "damping", "cfg_sx_grid": False, "cfg_sx_fix": 10,
}
SX_GRID = {
    "cfg_db": DB, "cfg_family": "Influence Spearman",
    "cfg_sp_mode": "Across swept axis", "cfg_sp_agg": "median", "cfg_sp_annot": False,
    "cfg_sx_model": "resnet_mlp_0a69ab6297da", "cfg_sx_pts": "all_classes",
    "cfg_sx_method": "kfac", "cfg_sx_strat": "pseudo_inverse",
    "cfg_sx_sweep": "epoch", "cfg_sx_grid": True,
    "cfg_sx_fixvals": [1e-09, 0.001, 0.1],
}
CS = {
    "cfg_db": DB, "cfg_family": "Influence Spearman",
    "cfg_sp_mode": "Compare samplings", "cfg_sp_agg": "mean", "cfg_sp_annot": True,
    "cfg_cs_model": "resnet_mlp_fbc1db7ec868", "cfg_cs_a": "all_classes",
    "cfg_cs_b": "mcmc", "cfg_cs_epoch": 4, "cfg_cs_lam": 1e-09,
    "cfg_cs_strat": "auto_mean",
}
PARETO = {
    "cfg_db": DB, "cfg_family": "Sample-size Pareto",
    "cfg_pareto_model": "mlp_08580ee2573a",
    "cfg_pareto_strat": "auto_mean", "cfg_pareto_epoch": 10, "cfg_pareto_lam": 1e-09,
    "cfg_pareto_methods": ALL13, "cfg_pareto_band": True, "cfg_pareto_logx": True,
}
PARETO_ALLSTRAT = {**PARETO, "cfg_pareto_strat": "(all)",
                   "cfg_pareto_methods": ["exact", "kfac"], "cfg_pareto_logx": False}
FE_RECS = {
    "cfg_db": DB, "cfg_family": "Factor eigenvalues",
    "cfg_fe_root": "/dbs/models",  # == default for this cfg_db -> drops out
    "cfg_fe_dataset": "digits", "cfg_fe_model": "mlp_08580ee2573a",
    "cfg_fe_epoch": 10, "cfg_fe_pts": "mcmc", "cfg_fe_method": "exact",
    "cfg_fe_layers": ["full matrix"], "cfg_fe_logx": False,
}
FE_DIR = {  # unparseable-layout fallback, with delimiter chars in the paths
    "cfg_db": DB, "cfg_family": "Factor eigenvalues",
    "cfg_fe_root": "/elsewhere/factor root",
    "cfg_fe_dir": "/elsewhere/factor root/runs & sweeps~v2,final/exact_hessian_layer_matrix",
    "cfg_fe_layers": ["full matrix", "0_weight"], "cfg_fe_logx": True,
}

# (name, cfg, keys canonicalized away by encoding: default paths, empty lists)
CASES = [
    ("lds_mf", LDS_MF, {"cfg_db"}),
    ("lds_mf_allstrat", LDS_MF_ALLSTRAT, {"cfg_db"}),
    ("lds_df", LDS_DF, {"cfg_db"}),
    ("lds_ef", LDS_EF, {"cfg_db"}),
    ("lds_hm", LDS_HM, {"cfg_db"}),
    ("mb_canon", MB_CANON, {"cfg_db", "cfg_mb_cats"}),
    ("mb_dragged", MB_DRAGGED, {"cfg_db"}),
    ("mc_single", MC_SINGLE, {"cfg_db"}),
    ("mc_sweep", MC_SWEEP, {"cfg_db"}),
    ("sp_single", SP_SINGLE, {"cfg_db"}),
    ("sx_fix", SX_FIX, {"cfg_db"}),
    ("sx_grid", SX_GRID, {"cfg_db"}),
    ("cs", CS, {"cfg_db"}),
    ("fe_recs", FE_RECS, {"cfg_db", "cfg_fe_root"}),
    ("fe_dir", FE_DIR, {"cfg_db"}),
    ("pareto", PARETO, {"cfg_db"}),
    ("pareto_allstrat", PARETO_ALLSTRAT, {"cfg_db"}),
]


# ── Bitmap ────────────────────────────────────────────────────────────
def test_bitmap_examples():
    assert cc.methods_to_bitmap(["exact"]) == "0001"
    assert cc.methods_to_bitmap(ALL13) == "1fff"
    assert cc.methods_to_bitmap([]) == "0000"
    assert cc.methods_to_bitmap(["exact", "gnh", "fim"]) == "0007"
    assert cc.methods_to_bitmap(["eidentity"]) == "1000"


@pytest.mark.parametrize("subset", [
    [], ["exact"], ["exact", "kfac"], ALL13, ["eidentity"], list(reversed(ALL13)),
])
def test_bitmap_roundtrip(subset):
    out = cc.bitmap_to_methods(cc.methods_to_bitmap(subset))
    assert out == [m for m in APPROX_ORDER if m in subset]  # canonical order


@pytest.mark.parametrize("bad", ["001", "00001", "A001", "xyzw", "2000", "ffff"])
def test_bitmap_malformed(bad):
    with pytest.raises(ValueError):
        cc.bitmap_to_methods(bad)


def test_bitmap_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        cc.methods_to_bitmap(["exact", "not_a_method"])


# ── Lehmer code ───────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [0, 1, 2, 5, 13])
def test_lehmer_roundtrip(n):
    assert cc.lehmer_rank(list(range(n))) == 0  # identity is rank 0
    for rank in {0, 1, math.factorial(n) - 1} & set(range(math.factorial(n))):
        assert cc.lehmer_rank(cc.lehmer_unrank(rank, n)) == rank
    with pytest.raises(ValueError):
        cc.lehmer_unrank(math.factorial(n), n)
    with pytest.raises(ValueError):
        cc.lehmer_unrank(-1, n)


# ── Worked examples (exact code strings) ──────────────────────────────
def test_exact_codes():
    assert enc(LDS_MF) == "1&1&1&1&3823&0&0&1&1&0001"
    assert enc(LDS_MF_ALLSTRAT) == "1&1&1&0&3823&0&0&1&1&0001"
    assert enc(MB_CANON) == "2&1&10&1e-09&1&1&0&1fff"  # perm/cats/db stripped
    assert enc(MB_DRAGGED) == "2&1&10&1e-09&1&1&0&0007&2&f5"


def test_perm_canonical_is_empty_slot():
    d = cc.decode(enc(MB_CANON))
    assert d["cfg_mb_approxs_sel"] == ALL13
    assert d["cfg_mb_approxs"] == ALL13
    d = cc.decode(enc(MB_DRAGGED))
    assert d["cfg_mb_approxs_sel"] == ["exact", "gnh", "fim"]  # canonical membership
    assert d["cfg_mb_approxs"] == ["gnh", "exact", "fim"]      # dragged order


# ── Per-family round trips ────────────────────────────────────────────
@pytest.mark.parametrize("name,cfg,dropped", CASES, ids=[c[0] for c in CASES])
def test_family_roundtrip(name, cfg, dropped):
    expected = {k: v for k, v in cfg.items() if k not in dropped}
    assert cc.decode(enc(cfg)) == expected


@pytest.mark.parametrize("name,cfg,dropped", CASES, ids=[c[0] for c in CASES])
def test_idempotence(name, cfg, dropped):
    code = enc(cfg)
    assert enc(cc.decode(code)) == code


# ── Value handling ────────────────────────────────────────────────────
def test_numpy_scalars():
    cfg = {**MB_CANON, "cfg_mb_epoch": np.int64(10), "cfg_mb_lam": np.float64(1e-09)}
    assert enc(cfg) == enc(MB_CANON)
    d = cc.decode(enc(cfg))
    assert type(d["cfg_mb_epoch"]) is int
    assert type(d["cfg_mb_lam"]) is float


@pytest.mark.parametrize("lam", [1e-09, 1e-06, 0.001, 0.1, 1.0])
def test_float_repr_roundtrip(lam):
    cfg = {**MB_CANON, "cfg_mb_lam": lam}
    d = cc.decode(enc(cfg))
    assert d["cfg_mb_lam"] == lam and type(d["cfg_mb_lam"]) is float
    assert type(d["cfg_mb_epoch"]) is int


def test_unknown_model_literal():
    cfg = {**LDS_MF, "cfg_lds_model": "brandnew_model_1"}
    code = enc(cfg)
    assert "~brandnew_model_1" in code
    assert cc.decode(code)["cfg_lds_model"] == "brandnew_model_1"


def test_pair_literal_fallback():
    cfg = {**MC_SINGLE, "cfg_mc_x": ("new_ct", "NEW METRIC&odd,name")}
    d = cc.decode(enc(cfg))
    assert d["cfg_mc_x"] == ("new_ct", "NEW METRIC&odd,name")
    cfg = {**MB_DRAGGED,
           "cfg_mb_cats": [("matrix", "frobenius"), ("future_ct", "weird~name")]}
    d = cc.decode(enc(cfg))
    assert d["cfg_mb_cats"] == [("matrix", "frobenius"), ("future_ct", "weird~name")]


def test_path_literals():
    cfg = {**LDS_MF, "cfg_db": "/other dir/runs.db"}
    code = enc(cfg)
    assert code.endswith("&~/other%20dir/runs.db")
    assert cc.decode(code)["cfg_db"] == "/other dir/runs.db"
    # default paths vanish from both the code and the decoded config
    d = cc.decode(enc(LDS_MF))
    assert "cfg_db" not in d
    d = cc.decode(enc(FE_RECS))
    assert "cfg_fe_root" not in d
    # delimiter chars in literals survive percent-quoting
    d = cc.decode(enc(FE_DIR))
    assert d["cfg_fe_dir"] == FE_DIR["cfg_fe_dir"]
    assert d["cfg_fe_layers"] == FE_DIR["cfg_fe_layers"]


# ── Malformed codes ───────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [
    "",                                   # empty
    "   ",                                # whitespace only
    "9",                                  # unknown family symbol
    "1&6",                                # unknown model symbol
    "1&1&1&1&7",                          # variant index out of range
    "3&5",                                # scope index out of range
    "1",                                  # missing branch discriminator
    "1&1&1&1&0&0&1&1&zzzz",               # bad bitmap
    "2&1&abc",                            # bad number
    "2&1&10&1e-09&1&1&0&0007&6",          # perm rank 6 >= 3!
    "2&1&10&1e-09&1&1&0&0007&Z",          # perm rank not base36
    "2&1&10&1e-09&1&1&0&&2",              # order without membership
    "2&1&10&1e-09&1&1&0&0007&&~x",        # unterminated literal pair
    "1&1&1&1&0&0&1&1&0001&&extra",        # too many fields
    "1&1&1&1&0&0&1&1&0001&/abs/path",     # path field without '~'
])
def test_malformed_codes(bad):
    with pytest.raises(ValueError):
        cc.decode(bad)


# ── Drift guards & fixed tables ───────────────────────────────────────
def test_schema_drift_guard():
    with pytest.raises(ValueError, match="cfg_lds_new_widget"):
        enc({**LDS_MF, "cfg_lds_new_widget": 1})


def test_symbol_tables():
    for table in (cc.FAMILY_SYMS, cc.MODEL_SYMS, cc.SAMPLING_SYMS,
                  cc.STRATEGY_SYMS, cc.DATASET_SYMS, cc.PAIR_SYMS):
        syms = list(table.values())
        assert len(set(syms)) == len(syms)
        assert all(len(s) == 1 for s in syms)
    assert cc.PAIR_SYMS[LDS_AXIS] == "l"
    assert set(cc.SCHEMA) == set(cc.FAMILIES) == set(cc.FAMILY_SYMS)
    assert len(APPROX_ORDER) == 13  # the 4-hex bitmap width assumes <= 16
