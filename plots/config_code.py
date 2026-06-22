"""Reversible config codes for the runs.db explorer.

Replaces the old SHA-256 hash + on-disk registry: a config code is a compact,
deterministic, *decodable* string, so pasting one restores the exact widget
selections on any machine — the code itself is the config.

Format: ``<family>&<slot>&<slot>&…`` — the first field is the plot-family
symbol (FAMILY_SYMS), the rest follow the per-family slot schema in SCHEMA in
widget (track) order. A slot whose cfg key was not tracked encodes as '';
trailing empty fields are stripped. Branch-discriminating widgets (LDS
variant, correlation scope, Spearman mode) are encoded before the slots that
depend on them, so the decoder always knows which slots follow. The cfg_db
slot is last in every family and empty for the default DB, so it usually
strips away.

Field encodings:
* fixed-option widgets       -> base36 index char                   (Choice)
* data-valued strings        -> 1-char symbol from a fixed table (Sym); values
                                not in the table fall back to a '~'-prefixed
                                percent-quoted literal
* method subsets             -> bitmap over APPROX_ORDER (bit i = method i,
                                LSB first) as exactly 4 hex chars:
                                {exact} -> 0001, all 13 -> 1fff      (Bitmap)
* method order (drag strips) -> Lehmer rank of the permutation relative to
                                canonical order, base36, '' if canonical (Perm)
* numbers                    -> str(int) / repr(float)               (Num)
* (computation_type, metric) -> 1-char symbol from PAIR_SYMS         (Pair)
* paths                      -> '' when equal to the runtime default, else
                                '~' + quoted literal                 (PathCodec)

Example: ``1&1&1&1&3823&0&0&1&1&0001`` = LDS sweeps, model 1, mcmc, auto_mean,
collector subset 3823, fix-method variant, Lines, CI band on, annotate on,
methods {exact}.

Symbol tables are append-only: never reorder or reuse a symbol, or existing
codes silently change meaning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote, unquote

from hessian_data import APPROX_ORDER, DB_PATH, LDS_AXIS, order_methods

# ── Widget option lists (shared with app.py so they cannot drift) ─────
FAMILIES = ("LDS sweeps", "Metric bars", "Metric correlation",
            "Influence Spearman", "Factor eigenvalues", "Sample-size Pareto")
LDS_VARIANTS = ("Fix method (epoch × damping)", "Fix damping (epoch × method)",
                "Fix epoch (damping × method)", "Heatmap per method")
LDS_STYLES = ("Lines", "Bars")
REFERENCES = ("exact", "gnh")
MC_SCOPES = ("Single result", "Across sweep")
SP_MODES = ("Single result (methods)", "Across swept axis", "Compare samplings")
SP_AGGS = ("mean", "median")
SX_SWEEPS = ("damping", "epoch")

# ── Fixed symbol tables (append-only — never reorder or reuse) ────────
FAMILY_SYMS = {
    "LDS sweeps": "1", "Metric bars": "2", "Metric correlation": "3",
    "Influence Spearman": "4", "Factor eigenvalues": "5", "Sample-size Pareto": "6",
}
MODEL_SYMS = {
    "mlp_08580ee2573a": "1",
    "resnet_mlp_0a69ab6297da": "2",
    "resnet_mlp_adc3030f5daa": "3",
    "resnet_mlp_fbc1db7ec868": "4",
    "resnet_mlp_swiglu_d4186381c706": "5",
}
SAMPLING_SYMS = {"all_classes": "0", "mcmc": "1"}
STRATEGY_SYMS = {"(all)": "0", "auto_mean": "1", "pseudo_inverse": "2",
                 "uniform": "3"}  # "(all)" is the UI-only sentinel
DATASET_SYMS = {"digits": "0", "fashion_mnist": "1"}

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"

# (computation_type, metric) pairs, sorted, plus the LDS sentinel axis.
_PAIRS = (
    ("hvp", "ABSOLUTE_L2_DIFF"), ("hvp", "COSINE_SIMILARITY"),
    ("hvp", "INNER_PRODUCT_DIFF"), ("hvp", "INNER_PRODUCT_RATIO"),
    ("hvp", "RELATIVE_ENERGY_DIFF"), ("hvp", "RELATIVE_ERROR"),
    ("hvp", "RELATIVE_INNER_PRODUCT_DIFF"),
    ("ihvp", "ABSOLUTE_L2_DIFF"), ("ihvp", "COSINE_SIMILARITY"),
    ("ihvp", "INNER_PRODUCT_DIFF"), ("ihvp", "INNER_PRODUCT_RATIO"),
    ("ihvp", "RELATIVE_ENERGY_DIFF"), ("ihvp", "RELATIVE_ERROR"),
    ("ihvp", "RELATIVE_INNER_PRODUCT_DIFF"),
    ("matrix", "cosine_similarity"), ("matrix", "frobenius"),
    ("matrix", "relative_frobenius"), ("matrix", "spectral"),
    ("matrix", "spectral_relative"), ("matrix", "trace_distance"),
    ("round_trip", "relative_error"),
    LDS_AXIS,
)
PAIR_SYMS = {p: _B36[i] for i, p in enumerate(_PAIRS)}
_PAIR_REV = {s: p for p, s in PAIR_SYMS.items()}
_FAMILY_REV = {s: f for f, s in FAMILY_SYMS.items()}


# ── Quoting & small number codecs ─────────────────────────────────────
def qlit(s: str) -> str:
    """Percent-quote a literal so it can never contain '&', ',' or '~'.
    ('~' is RFC-3986 unreserved, so quote() alone would leave our literal
    marker/terminator ambiguous — escape it by hand.)"""
    return quote(str(s), safe="/").replace("~", "%7E")


def unqlit(s: str) -> str:
    return unquote(s)


def _b36enc(n: int) -> str:
    if n == 0:
        return "0"
    digits = []
    while n:
        n, r = divmod(n, 36)
        digits.append(_B36[r])
    return "".join(reversed(digits))


def _b36dec(s: str) -> int:
    if not re.fullmatch(r"[0-9a-z]+", s or ""):
        raise ValueError(f"bad base36 number {s!r}")
    return int(s, 36)


# ── Methods bitmap (4 hex chars over APPROX_ORDER) ────────────────────
def methods_to_bitmap(methods: Iterable[str]) -> str:
    bits = 0
    for m in methods:
        try:
            bits |= 1 << APPROX_ORDER.index(m)
        except ValueError:
            raise ValueError(f"unknown method {m!r}") from None
    return format(bits, "04x")


def bitmap_to_methods(s: str) -> list[str]:
    if not re.fullmatch(r"[0-9a-f]{4}", s or ""):
        raise ValueError(f"bad methods bitmap {s!r}")
    bits = int(s, 16)
    if bits >= 1 << len(APPROX_ORDER):
        raise ValueError(f"methods bitmap {s!r} has unknown bits set")
    return [m for i, m in enumerate(APPROX_ORDER) if bits >> i & 1]


# ── Lehmer code (lexicographic permutation rank) ──────────────────────
def lehmer_rank(perm: Sequence[int]) -> int:
    n = len(perm)
    return sum(sum(1 for q in perm[i + 1:] if q < p) * math.factorial(n - 1 - i)
               for i, p in enumerate(perm))


def lehmer_unrank(rank: int, n: int) -> list[int]:
    if not 0 <= rank < math.factorial(n):
        raise ValueError(f"permutation rank {rank} out of range for {n} items")
    avail, out = list(range(n)), []
    for i in range(n, 0, -1):
        idx, rank = divmod(rank, math.factorial(i - 1))
        out.append(avail.pop(idx))
    return out


# ── Field codecs ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Ctx:
    cfg: dict
    db_default: str


class Codec:
    """enc/dec turn one present value into/from one non-empty field; the
    *_field hooks add the "absent key <-> empty field" rule and key context."""

    def enc(self, value, ctx: _Ctx) -> str:
        raise NotImplementedError

    def dec(self, text: str):
        raise NotImplementedError

    def enc_field(self, key: str, ctx: _Ctx) -> str:
        if key not in ctx.cfg:
            return ""
        try:
            return self.enc(ctx.cfg[key], ctx)
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from None

    def dec_field(self, text: str, key: str, out: dict) -> None:
        if text:
            try:
                out[key] = self.dec(text)
            except ValueError as e:
                raise ValueError(f"{key}: {e}") from None


class Choice(Codec):
    def __init__(self, options: Sequence):
        assert len(options) <= len(_B36)
        self.options = tuple(options)

    def enc(self, value, ctx):
        if value not in self.options:
            raise ValueError(f"{value!r} not one of {self.options}")
        return _B36[self.options.index(value)]

    def dec(self, text):
        if len(text) != 1 or text not in _B36 or _B36.index(text) >= len(self.options):
            raise ValueError(f"bad choice {text!r}")
        return self.options[_B36.index(text)]


BOOL = Choice((False, True))


class Sym(Codec):
    def __init__(self, table: dict):
        self.table = table
        self.rev = {s: v for v, s in table.items()}

    def enc(self, value, ctx):
        sym = self.table.get(value)
        return sym if sym is not None else "~" + qlit(value)

    def dec(self, text):
        if text.startswith("~"):
            return unqlit(text[1:])
        if text not in self.rev:
            raise ValueError(f"unknown symbol {text!r}")
        return self.rev[text]


class Bitmap(Codec):
    def enc(self, value, ctx):
        return methods_to_bitmap(value)

    def dec(self, text):
        return bitmap_to_methods(text)


class Perm(Codec):
    """Order of a drag-reorderable method list, as the Lehmer rank of its
    permutation relative to canonical order. The membership itself lives in
    the companion `sel_key` Bitmap slot, which must precede this one."""

    def __init__(self, sel_key: str):
        self.sel_key = sel_key

    def enc_field(self, key, ctx):
        cfg = ctx.cfg
        if key not in cfg or self.sel_key not in cfg:
            return ""
        members = set(cfg[self.sel_key])
        canon = order_methods(list(members))
        order = [m for m in cfg[key] if m in members]
        order += [m for m in canon if m not in order]
        rank = lehmer_rank([canon.index(m) for m in order])
        return "" if rank == 0 else _b36enc(rank)

    def dec_field(self, text, key, out):
        sel = out.get(self.sel_key)
        if sel is None:
            if text:
                raise ValueError(f"{key}: order given without a method selection")
            return
        try:
            rank = _b36dec(text) if text else 0
            out[key] = [sel[i] for i in lehmer_unrank(rank, len(sel))]
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from None


class Num(Codec):
    def enc(self, value, ctx):
        v = value.item() if hasattr(value, "item") else value  # numpy scalar
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"not a number: {value!r}")
        return repr(v) if isinstance(v, float) else str(v)

    def dec(self, text):
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"bad number {text!r}") from None


_NUM = Num()


class NumList(Codec):
    def enc(self, value, ctx):
        return ",".join(_NUM.enc(v, ctx) for v in value)

    def dec(self, text):
        return [_NUM.dec(t) for t in text.split(",")]


class MethodIdx(Codec):
    def enc(self, value, ctx):
        if value in APPROX_ORDER:
            return _B36[APPROX_ORDER.index(value)]
        return "~" + qlit(value)

    def dec(self, text):
        if text.startswith("~"):
            return unqlit(text[1:])
        if len(text) != 1 or text not in _B36 or _B36.index(text) >= len(APPROX_ORDER):
            raise ValueError(f"bad method index {text!r}")
        return APPROX_ORDER[_B36.index(text)]


class Pair(Codec):
    def enc(self, value, ctx):
        p = tuple(value)
        sym = PAIR_SYMS.get(p)
        return sym if sym is not None else f"~{qlit(p[0])},{qlit(p[1])}"

    def dec(self, text):
        if text.startswith("~"):
            ct, sep, metric = text[1:].partition(",")
            if not sep:
                raise ValueError(f"bad literal pair {text!r}")
            return (unqlit(ct), unqlit(metric))
        if text not in _PAIR_REV:
            raise ValueError(f"unknown pair symbol {text!r}")
        return _PAIR_REV[text]


class PairList(Codec):
    """Concatenated 1-char pair symbols; literal pairs are embedded as
    '~ct,metric~' (the trailing '~' terminates the literal)."""

    def enc(self, value, ctx):
        parts = []
        for p in value:
            p = tuple(p)
            sym = PAIR_SYMS.get(p)
            parts.append(sym if sym is not None else f"~{qlit(p[0])},{qlit(p[1])}~")
        return "".join(parts)

    def dec(self, text):
        out, i = [], 0
        while i < len(text):
            if text[i] == "~":
                j = text.find("~", i + 1)
                if j < 0:
                    raise ValueError(f"unterminated literal pair in {text!r}")
                ct, sep, metric = text[i + 1:j].partition(",")
                if not sep:
                    raise ValueError(f"bad literal pair in {text!r}")
                out.append((unqlit(ct), unqlit(metric)))
                i = j + 1
            elif text[i] in _PAIR_REV:
                out.append(_PAIR_REV[text[i]])
                i += 1
            else:
                raise ValueError(f"unknown pair symbol {text[i]!r}")
        return out


class PathCodec(Codec):
    """Empty when the value equals the runtime default, else a literal."""

    def __init__(self, default_fn):
        self.default_fn = default_fn

    def enc(self, value, ctx):
        d = self.default_fn(ctx)
        if d is not None and str(value) == str(d):
            return ""
        return "~" + qlit(value)

    def dec(self, text):
        if not text.startswith("~"):
            raise ValueError(f"bad path field {text!r}")
        return unqlit(text[1:])


class StrList(Codec):
    def enc(self, value, ctx):
        return ",".join(qlit(v) for v in value)

    def dec(self, text):
        return [unqlit(t) for t in text.split(",")]


# ── Per-family slot schemas (slot order = track() order in app.py) ────
@dataclass(frozen=True)
class Slot:
    key: str
    codec: Codec


@dataclass(frozen=True)
class Branch:
    on: str     # cfg key of an earlier Slot whose value picks the arm
    arms: dict  # decoded value -> tuple of Slot/Branch items


def _fe_root_default(ctx: _Ctx) -> str:
    return str(Path(ctx.cfg.get("cfg_db") or ctx.db_default).parent / "models")


DB_SLOT = Slot("cfg_db", PathCodec(lambda ctx: ctx.db_default))

SCHEMA: dict[str, tuple] = {
    "LDS sweeps": (
        Slot("cfg_lds_model", Sym(MODEL_SYMS)),
        Slot("cfg_lds_pts", Sym(SAMPLING_SYMS)),
        Slot("cfg_lds_strat", Sym(STRATEGY_SYMS)),
        Slot("cfg_lds_subset", Num()),
        Slot("cfg_lds_variant", Choice(LDS_VARIANTS)),
        Slot("cfg_lds_style", Choice(LDS_STYLES)),  # absent for heatmap
        Slot("cfg_lds_band", BOOL),
        Slot("cfg_lds_annot", BOOL),
        Branch("cfg_lds_variant", {
            LDS_VARIANTS[0]: (Slot("cfg_lds_mf_methods", Bitmap()),),
            LDS_VARIANTS[1]: (Slot("cfg_lds_df_lam", Num()),
                              Slot("cfg_lds_df_methods", Bitmap())),
            LDS_VARIANTS[2]: (Slot("cfg_lds_ef_epoch", Num()),
                              Slot("cfg_lds_ef_methods", Bitmap())),
            LDS_VARIANTS[3]: (Slot("cfg_lds_hm_methods", Bitmap()),),
        }),
        DB_SLOT,
    ),
    "Metric bars": (
        Slot("cfg_mb_model", Sym(MODEL_SYMS)),
        Slot("cfg_mb_epoch", Num()),
        Slot("cfg_mb_lam", Num()),
        Slot("cfg_mb_strat", Sym(STRATEGY_SYMS)),
        Slot("cfg_mb_pts", Sym(SAMPLING_SYMS)),
        Slot("cfg_mb_ref", Choice(REFERENCES)),
        Slot("cfg_mb_approxs_sel", Bitmap()),
        Slot("cfg_mb_approxs", Perm("cfg_mb_approxs_sel")),
        Slot("cfg_mb_cats", PairList()),
        DB_SLOT,
    ),
    "Metric correlation": (
        Slot("cfg_mc_scope", Choice(MC_SCOPES)),
        Branch("cfg_mc_scope", {
            MC_SCOPES[0]: (Slot("cfg_mc_model", Sym(MODEL_SYMS)),
                           Slot("cfg_mc_epoch", Num()),
                           Slot("cfg_mc_lam", Num()),
                           Slot("cfg_mc_strat", Sym(STRATEGY_SYMS)),
                           Slot("cfg_mc_pts", Sym(SAMPLING_SYMS))),
            MC_SCOPES[1]: (Slot("cfg_mc_model", Sym(MODEL_SYMS)),
                           Slot("cfg_mc_pts", Sym(SAMPLING_SYMS)),
                           Slot("cfg_mc_strat", Sym(STRATEGY_SYMS))),
        }),
        Slot("cfg_mc_ref", Choice(REFERENCES)),
        Slot("cfg_mc_x", Pair()),
        Slot("cfg_mc_y", Pair()),
        Slot("cfg_mc_logx", BOOL),
        Slot("cfg_mc_logy", BOOL),
        Slot("cfg_mc_annot", BOOL),  # absent in "Across sweep" scope
        Slot("cfg_mc_points", Bitmap()),
        DB_SLOT,
    ),
    "Influence Spearman": (
        Slot("cfg_sp_mode", Choice(SP_MODES)),
        Slot("cfg_sp_agg", Choice(SP_AGGS)),
        Slot("cfg_sp_annot", BOOL),
        Branch("cfg_sp_mode", {
            SP_MODES[0]: (Slot("cfg_sp_model", Sym(MODEL_SYMS)),
                          Slot("cfg_sp_epoch", Num()),
                          Slot("cfg_sp_lam", Num()),
                          Slot("cfg_sp_strat", Sym(STRATEGY_SYMS)),
                          Slot("cfg_sp_pts", Sym(SAMPLING_SYMS)),
                          Slot("cfg_sp_methods_sel", Bitmap()),
                          Slot("cfg_sp_methods", Perm("cfg_sp_methods_sel"))),
            SP_MODES[1]: (Slot("cfg_sx_model", Sym(MODEL_SYMS)),
                          Slot("cfg_sx_pts", Sym(SAMPLING_SYMS)),
                          Slot("cfg_sx_method", MethodIdx()),
                          Slot("cfg_sx_strat", Sym(STRATEGY_SYMS)),
                          Slot("cfg_sx_sweep", Choice(SX_SWEEPS)),
                          Slot("cfg_sx_grid", BOOL),
                          Slot("cfg_sx_fix", Num()),        # grid off
                          Slot("cfg_sx_fixvals", NumList())),  # grid on
            SP_MODES[2]: (Slot("cfg_cs_model", Sym(MODEL_SYMS)),
                          Slot("cfg_cs_a", Sym(SAMPLING_SYMS)),
                          Slot("cfg_cs_b", Sym(SAMPLING_SYMS)),
                          Slot("cfg_cs_epoch", Num()),
                          Slot("cfg_cs_lam", Num()),
                          Slot("cfg_cs_strat", Sym(STRATEGY_SYMS))),
        }),
        DB_SLOT,
    ),
    "Sample-size Pareto": (
        Slot("cfg_pareto_model", Sym(MODEL_SYMS)),
        Slot("cfg_pareto_strat", Sym(STRATEGY_SYMS)),
        Slot("cfg_pareto_epoch", Num()),
        Slot("cfg_pareto_lam", Num()),
        Slot("cfg_pareto_methods", Bitmap()),
        Slot("cfg_pareto_band", BOOL),
        Slot("cfg_pareto_logx", BOOL),
        DB_SLOT,
    ),
    "Factor eigenvalues": (
        Slot("cfg_fe_root", PathCodec(_fe_root_default)),
        # parsed-layout cascade … or the raw-dir fallback; the unused
        # branch's keys are simply absent and encode as empty slots.
        Slot("cfg_fe_dataset", Sym(DATASET_SYMS)),
        Slot("cfg_fe_model", Sym(MODEL_SYMS)),
        Slot("cfg_fe_epoch", Num()),
        Slot("cfg_fe_pts", Sym(SAMPLING_SYMS)),
        Slot("cfg_fe_method", MethodIdx()),
        Slot("cfg_fe_dir", PathCodec(lambda ctx: None)),
        Slot("cfg_fe_layers", StrList()),
        Slot("cfg_fe_logx", BOOL),
        DB_SLOT,
    ),
}


# ── encode / decode ───────────────────────────────────────────────────
def _enc_walk(items, ctx, fields, visited):
    for it in items:
        if isinstance(it, Slot):
            visited.add(it.key)
            fields.append(it.codec.enc_field(it.key, ctx))
        else:
            val = ctx.cfg.get(it.on)
            if val not in it.arms:
                raise ValueError(f"cannot encode: {it.on} is {val!r}")
            _enc_walk(it.arms[val], ctx, fields, visited)


def encode(cfg: dict, *, db_default: str | None = None) -> str:
    """Encode a tracked-widget CFG dict into a config code."""
    fam = cfg.get("cfg_family")
    if fam not in SCHEMA:
        raise ValueError(f"unknown plot family {fam!r}")
    ctx = _Ctx(cfg=cfg, db_default=str(db_default) if db_default else str(DB_PATH))
    fields, visited = [FAMILY_SYMS[fam]], {"cfg_family"}
    _enc_walk(SCHEMA[fam], ctx, fields, visited)
    extra = set(cfg) - visited
    if extra:  # drift guard: a new track() key needs a schema slot
        raise ValueError(f"cfg keys missing from the schema: {sorted(extra)}")
    while fields and fields[-1] == "":
        fields.pop()
    return "&".join(fields)


_END = object()


def _dec_walk(items, it, out):
    for item in items:
        if isinstance(item, Slot):
            item.codec.dec_field(next(it, ""), item.key, out)
        else:
            if item.on not in out:
                raise ValueError(f"missing {item.on} before its dependent fields")
            val = out[item.on]
            if val not in item.arms:
                raise ValueError(f"no schema branch for {item.on} = {val!r}")
            _dec_walk(item.arms[val], it, out)


def decode(code: str) -> dict:
    """Invert encode(): a {cfg_key: value} dict ready for _pending_cfg."""
    code = (code or "").strip()
    if not code:
        raise ValueError("empty config code")
    fields = code.split("&")
    fam = _FAMILY_REV.get(fields[0])
    if fam is None:
        raise ValueError(f"unknown plot-family symbol {fields[0]!r}")
    out, it = {"cfg_family": fam}, iter(fields[1:])
    _dec_walk(SCHEMA[fam], it, out)
    if next(it, _END) is not _END:
        raise ValueError("too many fields for this plot family")
    return out


# ── Import-time sanity checks on the fixed tables ─────────────────────
def _check_tables():
    for name, table in [("FAMILY_SYMS", FAMILY_SYMS), ("MODEL_SYMS", MODEL_SYMS),
                        ("SAMPLING_SYMS", SAMPLING_SYMS), ("STRATEGY_SYMS", STRATEGY_SYMS),
                        ("DATASET_SYMS", DATASET_SYMS), ("PAIR_SYMS", PAIR_SYMS)]:
        syms = list(table.values())
        assert len(set(syms)) == len(syms) and all(
            len(s) == 1 and s in _B36 for s in syms), f"bad symbols in {name}"
    assert len(APPROX_ORDER) <= 16, "methods bitmap no longer fits 4 hex chars"
    assert LDS_AXIS in PAIR_SYMS
    assert set(SCHEMA) == set(FAMILIES) == set(FAMILY_SYMS)


_check_tables()
