"""Round-trip and malformed-input tests for the word-phrase config codes in
plots/config_code.py.

Reversibility is against the committed plots/code_vocab.json, so the fixture
values below (models, epochs, dampings, subset sizes, layers) are all drawn
from that vocab.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))
import config_code as cc  # noqa: E402
from config_code import _vkey  # noqa: E402
from hessian_data import APPROX_ORDER, LDS_AXIS  # noqa: E402

ALL13 = list(APPROX_ORDER)


def enc(cfg):
    return cc.encode(cfg)


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _vkey(v)
    return v


def _same(a, b):
    """Value equality tolerant of int/float/numpy and set ordering."""
    if isinstance(a, list) and isinstance(b, list):
        na, nb = [_norm(x) for x in a], [_norm(x) for x in b]
        return na == nb or sorted(map(str, na)) == sorted(map(str, nb))
    return _norm(a) == _norm(b)


# ── Fixture configs (one per family / branch) ─────────────────────────
LDS_MF = {
    "cfg_family": "LDS sweeps",
    "cfg_lds_model": "mlp_08580ee2573a", "cfg_lds_pts": "mcmc",
    "cfg_lds_strat": "auto_mean", "cfg_lds_subset": 3823,
    "cfg_lds_variant": "Fix method (epoch × damping)",
    "cfg_lds_style": "Lines", "cfg_lds_band": True, "cfg_lds_annot": True,
    "cfg_lds_mf_methods": ["exact"],
}
LDS_DF = {
    "cfg_family": "LDS sweeps",
    "cfg_lds_model": "mlp_08580ee2573a", "cfg_lds_pts": "mcmc",
    "cfg_lds_strat": "auto_mean", "cfg_lds_subset": 3823,
    "cfg_lds_variant": "Fix damping (epoch × method)",
    "cfg_lds_style": "Lines", "cfg_lds_band": True, "cfg_lds_annot": True,
    "cfg_lds_df_lam": 1e-09, "cfg_lds_df_methods": ALL13,
}
LDS_EF = {
    "cfg_family": "LDS sweeps",
    "cfg_lds_model": "resnet_mlp_swiglu_d4186381c706", "cfg_lds_pts": "all_classes",
    "cfg_lds_strat": "pseudo_inverse", "cfg_lds_subset": 60000,
    "cfg_lds_variant": "Fix epoch (damping × method)",
    "cfg_lds_style": "Bars", "cfg_lds_band": False, "cfg_lds_annot": True,
    "cfg_lds_ef_epoch": 10, "cfg_lds_ef_methods": ["exact", "gnh"],
}
LDS_HM = {  # heatmap variant: cfg_lds_style is never tracked (absent)
    "cfg_family": "LDS sweeps",
    "cfg_lds_model": "mlp_08580ee2573a", "cfg_lds_pts": "mcmc",
    "cfg_lds_strat": "auto_mean", "cfg_lds_subset": 3823,
    "cfg_lds_variant": "Heatmap per method",
    "cfg_lds_band": True, "cfg_lds_annot": False,
    "cfg_lds_hm_methods": ["exact"],
}
MB_CANON = {
    "cfg_family": "Metric bars",
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
    "cfg_family": "Metric correlation", "cfg_mc_scope": "Single result",
    "cfg_mc_model": "mlp_08580ee2573a", "cfg_mc_epoch": 10, "cfg_mc_lam": 1e-09,
    "cfg_mc_strat": "auto_mean", "cfg_mc_pts": "mcmc", "cfg_mc_ref": "exact",
    "cfg_mc_x": ("matrix", "frobenius"), "cfg_mc_y": LDS_AXIS,
    "cfg_mc_logx": True, "cfg_mc_logy": True, "cfg_mc_annot": True,
    "cfg_mc_points": ALL13[1:],
}
MC_SWEEP = {  # across-sweep scope: no epoch/lam and no cfg_mc_annot
    "cfg_family": "Metric correlation", "cfg_mc_scope": "Across sweep",
    "cfg_mc_model": "mlp_08580ee2573a", "cfg_mc_pts": "mcmc",
    "cfg_mc_strat": "auto_mean", "cfg_mc_ref": "exact",
    "cfg_mc_x": ("matrix", "frobenius"), "cfg_mc_y": LDS_AXIS,
    "cfg_mc_logx": True, "cfg_mc_logy": True, "cfg_mc_points": ALL13[1:],
}
SP_SINGLE = {
    "cfg_family": "Influence Spearman",
    "cfg_sp_mode": "Single result (methods)", "cfg_sp_agg": "mean",
    "cfg_sp_annot": True,
    "cfg_sp_model": "mlp_08580ee2573a", "cfg_sp_epoch": 10, "cfg_sp_lam": 1e-09,
    "cfg_sp_strat": "auto_mean", "cfg_sp_pts": "mcmc",
    "cfg_sp_methods_sel": ALL13, "cfg_sp_methods": ALL13,
}
SX_FIX = {
    "cfg_family": "Influence Spearman",
    "cfg_sp_mode": "Across swept axis", "cfg_sp_agg": "mean", "cfg_sp_annot": True,
    "cfg_sx_model": "mlp_08580ee2573a", "cfg_sx_pts": "mcmc",
    "cfg_sx_method": "exact", "cfg_sx_strat": "auto_mean",
    "cfg_sx_sweep": "damping", "cfg_sx_grid": False, "cfg_sx_fix": 10,
}
SX_GRID = {
    "cfg_family": "Influence Spearman",
    "cfg_sp_mode": "Across swept axis", "cfg_sp_agg": "median", "cfg_sp_annot": False,
    "cfg_sx_model": "resnet_mlp_0a69ab6297da", "cfg_sx_pts": "all_classes",
    "cfg_sx_method": "kfac", "cfg_sx_strat": "pseudo_inverse",
    "cfg_sx_sweep": "epoch", "cfg_sx_grid": True,
    "cfg_sx_fixvals": [1e-09, 0.001, 0.1],
}
CS = {
    "cfg_family": "Influence Spearman",
    "cfg_sp_mode": "Compare samplings", "cfg_sp_agg": "mean", "cfg_sp_annot": True,
    "cfg_cs_model": "resnet_mlp_fbc1db7ec868", "cfg_cs_a": "all_classes",
    "cfg_cs_b": "mcmc", "cfg_cs_epoch": 4, "cfg_cs_lam": 1e-09,
    "cfg_cs_strat": "auto_mean",
}
PARETO = {
    "cfg_family": "Sample-size Pareto",
    "cfg_pareto_model": "mlp_08580ee2573a",
    "cfg_pareto_strat": "auto_mean", "cfg_pareto_epoch": 10, "cfg_pareto_lam": 1e-09,
    "cfg_pareto_methods": ALL13, "cfg_pareto_band": True, "cfg_pareto_logx": True,
}
FE = {
    "cfg_family": "Factor eigenvalues",
    "cfg_fe_dataset": "digits", "cfg_fe_model": "mlp_08580ee2573a",
    "cfg_fe_epoch": 10, "cfg_fe_pts": "mcmc", "cfg_fe_method": "exact",
    "cfg_fe_layers": ["full matrix"], "cfg_fe_logx": False,
}

CASES = [
    ("lds_mf", LDS_MF), ("lds_df", LDS_DF), ("lds_ef", LDS_EF), ("lds_hm", LDS_HM),
    ("mb_canon", MB_CANON), ("mb_dragged", MB_DRAGGED),
    ("mc_single", MC_SINGLE), ("mc_sweep", MC_SWEEP),
    ("sp_single", SP_SINGLE), ("sx_fix", SX_FIX), ("sx_grid", SX_GRID), ("cs", CS),
    ("pareto", PARETO), ("fe", FE),
]


# ── Wordlist / integer serialisation ──────────────────────────────────
@pytest.mark.parametrize("n", [0, 1, 5, 2047, 2048, 2049, 10**6, 10**30])
def test_int_phrase_roundtrip(n):
    assert cc.phrase_to_int(cc.int_to_phrase(n)) == n


def test_phrase_shape():
    assert cc.int_to_phrase(0) == cc.WORDLIST[0]
    assert "-" in cc.int_to_phrase(10**9)  # multi-word
    # phrases are hyphen-joined lowercase words
    assert all(w in cc.WORDLIST for w in cc.int_to_phrase(12345).split("-"))


# ── Methods bitmap ────────────────────────────────────────────────────
@pytest.mark.parametrize("subset", [
    [], ["exact"], ["exact", "kfac"], ALL13, ["eidentity"], list(reversed(ALL13)),
])
def test_bitmap_roundtrip(subset):
    out = cc.bits_to_methods(cc.methods_to_bits(subset))
    assert out == [m for m in APPROX_ORDER if m in subset]  # canonical order


def test_bitmap_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        cc.methods_to_bits(["exact", "not_a_method"])


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


# ── Per-family round trips ────────────────────────────────────────────
@pytest.mark.parametrize("name,cfg", CASES, ids=[c[0] for c in CASES])
def test_idempotence(name, cfg):
    # Re-encoding the decoded config is stable — the true reversibility invariant
    # (defaults such as a canonical order / empty set are legitimately omitted).
    code = enc(cfg)
    assert enc(cc.decode(code)) == code


@pytest.mark.parametrize("name,cfg", CASES, ids=[c[0] for c in CASES])
def test_fidelity(name, cfg):
    # Every key present in both must agree (int/float/set-order tolerant).
    back = cc.decode(enc(cfg))
    for k in set(cfg) & set(back):
        assert _same(cfg[k], back[k]), f"{k}: {cfg[k]!r} != {back[k]!r}"


@pytest.mark.parametrize("name,cfg", CASES, ids=[c[0] for c in CASES])
def test_phrase_is_short(name, cfg):
    assert len(enc(cfg).split("-")) <= 8  # every real config is a short phrase


def test_perm_canonical_omitted():
    # canonical order + empty categories drop out; a drag order survives.
    d = cc.decode(enc(MB_CANON))
    assert d["cfg_mb_approxs_sel"] == ALL13
    assert "cfg_mb_approxs" not in d and "cfg_mb_cats" not in d
    d = cc.decode(enc(MB_DRAGGED))
    assert d["cfg_mb_approxs_sel"] == ["exact", "gnh", "fim"]
    assert d["cfg_mb_approxs"] == ["gnh", "exact", "fim"]
    assert d["cfg_mb_cats"] == [("matrix", "frobenius"), ("hvp", "RELATIVE_ERROR")]


def test_paths_not_carried():
    # DB / factor-root are excluded from the phrase entirely.
    d = cc.decode(enc({**LDS_MF, "cfg_db": "/somewhere/else.db"}))
    assert "cfg_db" not in d
    assert enc({**LDS_MF, "cfg_db": "/a"}) == enc({**LDS_MF, "cfg_db": "/b"})


# ── Value handling ────────────────────────────────────────────────────
def test_numpy_scalars():
    cfg = {**MB_CANON, "cfg_mb_epoch": np.int64(10), "cfg_mb_lam": np.float64(1e-09),
           "cfg_mb_model": "mlp_08580ee2573a"}
    assert enc(cfg) == enc(MB_CANON)
    d = cc.decode(enc(cfg))
    assert d["cfg_mb_epoch"] == 10 and d["cfg_mb_lam"] == 1e-09


@pytest.mark.parametrize("lam", [1e-09, 1e-06, 0.001, 0.1, 1.0])
def test_float_roundtrip(lam):
    d = cc.decode(enc({**MB_CANON, "cfg_mb_lam": lam}))
    assert d["cfg_mb_lam"] == lam


def test_integer_valued_float_matches_int():
    # subset stored as int in vocab; a float 3823.0 must still resolve.
    a = enc({**LDS_MF, "cfg_lds_subset": 3823})
    b = enc({**LDS_MF, "cfg_lds_subset": 3823.0})
    assert a == b


def test_unknown_value_rejected():
    with pytest.raises(ValueError, match="vocab"):
        enc({**LDS_MF, "cfg_lds_model": "brandnew_model_1"})


# ── Malformed phrases ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [
    "",                       # empty
    "   ",                    # whitespace only
    "notarealword",          # unknown word
    "abandon-notaword",      # one unknown word
    "abandon",               # decodes to family index 0 → no family
    "-".join(["zoo"] * 12),  # huge N → trailing data for whatever family
])
def test_malformed_phrases(bad):
    with pytest.raises(ValueError):
        cc.decode(bad)


# ── Drift guards & fixed tables ───────────────────────────────────────
def test_schema_drift_guard():
    with pytest.raises(ValueError, match="cfg_lds_new_widget"):
        enc({**LDS_MF, "cfg_lds_new_widget": 1})


def test_vocab_and_schema():
    assert set(cc.SCHEMA) == set(cc.FAMILIES)
    assert len(cc.WORDLIST) == 2048 == len(set(cc.WORDLIST))
    for domain in ("model", "dataset", "sampling", "strategy", "epoch",
                   "damping", "subset_size", "layer"):
        assert cc.VOCAB.get(domain), f"missing vocab domain {domain}"
    assert len(APPROX_ORDER) == 13
    assert LDS_AXIS in cc._PAIRS
