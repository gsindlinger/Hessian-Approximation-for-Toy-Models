"""Human-readable word-phrase config codes for the runs.db explorer.

A config code is a short phrase like ``otter-cobalt-maple`` that fully encodes
the current plot's widget selections — paste it back and the exact plot is
restored. It replaces the older ``&``-delimited symbol string.

How it stays both short and reversible:

* **Every field is an index, not a literal.** Enum widgets index a fixed option
  list; the data-valued fields (model, epoch, damping, subset size, layers, …)
  index the canonical ordering in ``code_vocab.json`` (built by
  ``build_code_vocab.py``). So ``damping = 1e-3`` codes as "the 5th damping",
  not the literal ``0.001`` — a few bits instead of a dozen characters.
* **All indices pack into one integer** in a mixed radix (each field contributes
  ``log2(its option count)`` bits). The walk order is the per-family ``SCHEMA``;
  the plot-family goes in the least-significant position so the decoder can read
  it first and know which schema (and therefore which radices) follow.
* **That integer renders to words** via ``wordlist.txt`` (2048 words = 11 bits
  each, base-2048): integer → digits → words, joined by ``-``.

Reversibility is against ``code_vocab.json``: a phrase decodes correctly on any
machine sharing that (committed, append-only) file. New data values must be
appended — never reordered — or existing phrases change meaning.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

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

# (computation_type, metric) pairs, sorted, plus the LDS sentinel axis. Order is
# fixed (append-only) — a pair's position is its code.
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

_HERE = Path(__file__).resolve().parent


# ── Wordlist (integer ↔ phrase) ───────────────────────────────────────
def _load_wordlist() -> list[str]:
    words = [w.strip() for w in (_HERE / "wordlist.txt").read_text().splitlines() if w.strip()]
    if len(words) != 2048 or len(set(words)) != 2048:
        raise ValueError(f"wordlist.txt must hold 2048 unique words, got {len(words)}")
    return words


WORDLIST = _load_wordlist()
_WORD_POS = {w: i for i, w in enumerate(WORDLIST)}
BASE = len(WORDLIST)  # 2048


def int_to_phrase(n: int) -> str:
    """Non-negative integer → hyphen-joined word phrase (base-2048, MSW first)."""
    if n < 0:
        raise ValueError("phrase integer must be non-negative")
    digits = []
    while True:
        n, r = divmod(n, BASE)
        digits.append(WORDLIST[r])
        if n == 0:
            break
    return "-".join(reversed(digits))


def phrase_to_int(phrase: str) -> int:
    words = [w for w in (phrase or "").strip().lower().replace("_", "-").split("-") if w]
    if not words:
        raise ValueError("empty phrase")
    n = 0
    for w in words:
        if w not in _WORD_POS:
            raise ValueError(f"unknown word {w!r}")
        n = n * BASE + _WORD_POS[w]
    return n


# ── Vocab (data-valued domains ↔ index) ───────────────────────────────
def _vkey(v) -> str:
    """Canonical string key for a value, so int/float/numpy compare equal.
    Integer-valued floats collapse to the integer form (3823.0 == 3823)."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _load_vocab() -> dict[str, list]:
    vocab = json.loads((_HERE / "code_vocab.json").read_text())
    for domain, values in vocab.items():
        keys = [_vkey(v) for v in values]
        if len(set(keys)) != len(keys):
            raise ValueError(f"code_vocab.json domain {domain!r} has duplicate values")
    return vocab


VOCAB = _load_vocab()
_VOCAB_POS = {d: {_vkey(v): i for i, v in enumerate(vals)} for d, vals in VOCAB.items()}


def _vocab_pos(domain: str, value) -> int:
    pos = _VOCAB_POS.get(domain, {}).get(_vkey(value))
    if pos is None:
        raise ValueError(
            f"{value!r} not in vocab domain {domain!r} — run build_code_vocab.py")
    return pos


def _vocab_val(domain: str, pos: int):
    vals = VOCAB.get(domain, [])
    if not 0 <= pos < len(vals):
        raise ValueError(f"index {pos} out of range for vocab domain {domain!r}")
    return vals[pos]


# ── Methods bitmap helpers (bit i = APPROX_ORDER[i]) ──────────────────
def methods_to_bits(methods) -> int:
    bits = 0
    for m in methods:
        try:
            bits |= 1 << APPROX_ORDER.index(m)
        except ValueError:
            raise ValueError(f"unknown method {m!r}") from None
    return bits


def bits_to_methods(bits: int) -> list[str]:
    if bits >= 1 << len(APPROX_ORDER):
        raise ValueError(f"methods bits {bits} has unknown bits set")
    return [m for i, m in enumerate(APPROX_ORDER) if bits >> i & 1]


# ── Lehmer code (lexicographic permutation rank) ──────────────────────
def lehmer_rank(perm) -> int:
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


# ── Codecs ────────────────────────────────────────────────────────────
# Each codec maps one field to an index in [0, radix): index 0 means "the field
# is absent from this config" (an optional slot / non-taken branch leaf), and
# 1..cardinality are the real values. `reader` is the sibling-value source —
# `cfg` when encoding, the partially-decoded `out` when decoding — so a codec
# whose size depends on another field (Perm on its bitmap, the sweep-conditional
# axis) can look it up.
class Codec:
    def cardinality(self, key: str, reader) -> int:
        raise NotImplementedError

    def value_index(self, value, key: str, reader) -> int:
        raise NotImplementedError

    def value_at(self, pos: int, key: str, reader):
        raise NotImplementedError

    def radix(self, key: str, reader) -> int:
        return self.cardinality(key, reader) + 1

    def enc(self, key: str, reader) -> int:
        value = reader.get(key)
        if value is None:
            return 0
        pos = self.value_index(value, key, reader)
        if not 0 <= pos < self.cardinality(key, reader):
            raise ValueError(f"{key}: {value!r} out of domain")
        return pos + 1

    def dec(self, idx: int, key: str, reader) -> None:
        if idx:  # 0 == absent
            reader[key] = self.value_at(idx - 1, key, reader)


class _DefaultAbsent:
    """Mixin: a value equal to the widget default encodes as absent (index 0),
    so a canonical method order / an empty category set / an all-layers pick
    costs zero bits instead of inflating every field packed after it. Decode
    leaves the key unset and the app supplies the same default."""

    def is_default(self, value, key, reader) -> bool:  # noqa: D401
        raise NotImplementedError

    def enc(self, key: str, reader) -> int:
        value = reader.get(key)
        if value is None or self.is_default(value, key, reader):
            return 0
        pos = self.value_index(value, key, reader)
        if not 0 <= pos < self.cardinality(key, reader):
            raise ValueError(f"{key}: {value!r} out of domain")
        return pos + 1


class Choice(Codec):
    """Index into a fixed option tuple."""

    def __init__(self, options):
        self.options = tuple(options)

    def cardinality(self, key, reader):
        return len(self.options)

    def value_index(self, value, key, reader):
        if value not in self.options:
            raise ValueError(f"{value!r} not one of {self.options}")
        return self.options.index(value)

    def value_at(self, pos, key, reader):
        return self.options[pos]


BOOL = Choice((False, True))


class Dom(Codec):
    """Index into a `code_vocab.json` domain. `domain` may be a name or a
    callable(reader)->name for a sweep-conditional axis."""

    def __init__(self, domain):
        self.domain = domain

    def _name(self, reader) -> str:
        return self.domain(reader) if callable(self.domain) else self.domain

    def cardinality(self, key, reader):
        return len(VOCAB.get(self._name(reader), []))

    def value_index(self, value, key, reader):
        return _vocab_pos(self._name(reader), value)

    def value_at(self, pos, key, reader):
        return _vocab_val(self._name(reader), pos)


class MethodIdx(Codec):
    """A single approximator, indexed into APPROX_ORDER."""

    def cardinality(self, key, reader):
        return len(APPROX_ORDER)

    def value_index(self, value, key, reader):
        if value not in APPROX_ORDER:
            raise ValueError(f"unknown method {value!r}")
        return APPROX_ORDER.index(value)

    def value_at(self, pos, key, reader):
        return APPROX_ORDER[pos]


class Pair(Codec):
    """A single (computation_type, metric) pair, indexed into _PAIRS."""

    def cardinality(self, key, reader):
        return len(_PAIRS)

    def value_index(self, value, key, reader):
        p = tuple(value)
        if p not in _PAIRS:
            raise ValueError(f"unknown pair {p!r}")
        return _PAIRS.index(p)

    def value_at(self, pos, key, reader):
        return _PAIRS[pos]


class Bitmap(Codec):
    """A *set* of methods as a bitmap over APPROX_ORDER (order-insensitive)."""

    def cardinality(self, key, reader):
        return 1 << len(APPROX_ORDER)

    def value_index(self, value, key, reader):
        return methods_to_bits(value)

    def value_at(self, pos, key, reader):
        return bits_to_methods(pos)


class PairSet(_DefaultAbsent, Codec):
    """A *set* of pairs as a bitmap over _PAIRS (e.g. metric-bar categories).
    Empty (= "all categories", the widget default) encodes as absent."""

    def is_default(self, value, key, reader):
        return not value

    def cardinality(self, key, reader):
        return 1 << len(_PAIRS)

    def _pos(self, p):
        p = tuple(p)
        if p not in _PAIRS:
            raise ValueError(f"unknown pair {p!r}")
        return _PAIRS.index(p)

    def value_index(self, value, key, reader):
        bits = 0
        for p in value:
            bits |= 1 << self._pos(p)
        return bits

    def value_at(self, pos, key, reader):
        return [_PAIRS[i] for i in range(len(_PAIRS)) if pos >> i & 1]


class DomSet(_DefaultAbsent, Codec):
    """A *set* of values from a vocab domain, as a bitmap over that domain
    (order-insensitive). `domain` may be a name or callable(reader)->name.
    Empty encodes as absent."""

    def __init__(self, domain):
        self.domain = domain

    def is_default(self, value, key, reader):
        return not value

    def _name(self, reader) -> str:
        return self.domain(reader) if callable(self.domain) else self.domain

    def cardinality(self, key, reader):
        return 1 << len(VOCAB.get(self._name(reader), []))

    def value_index(self, value, key, reader):
        name = self._name(reader)
        bits = 0
        for v in value:
            bits |= 1 << _vocab_pos(name, v)
        return bits

    def value_at(self, pos, key, reader):
        name = self._name(reader)
        vals = VOCAB.get(name, [])
        return [vals[i] for i in range(len(vals)) if pos >> i & 1]


class Perm(_DefaultAbsent, Codec):
    """Order of a drag-reorderable method list, as the Lehmer rank relative to
    canonical order. The membership lives in the companion `sel_key` Bitmap slot
    (which precedes this one), so the rank's radix is `len(sel)!`. Canonical
    order (rank 0, the default) encodes as absent."""

    def __init__(self, sel_key: str):
        self.sel_key = sel_key

    def is_default(self, value, key, reader):
        sel = self._sel(reader)
        return sel is None or self.value_index(value, key, reader) == 0

    def _sel(self, reader):
        return reader.get(self.sel_key)

    def cardinality(self, key, reader):
        sel = self._sel(reader)
        return math.factorial(len(sel)) if sel is not None else 0

    def value_index(self, value, key, reader):
        sel = self._sel(reader)
        members = set(sel)
        canon = order_methods(list(members))
        order = [m for m in value if m in members]
        order += [m for m in canon if m not in order]
        return lehmer_rank([canon.index(m) for m in order])

    def value_at(self, pos, key, reader):
        canon = order_methods(list(self._sel(reader)))
        return [canon[i] for i in lehmer_unrank(pos, len(canon))]


class Drop(Codec):
    """A field intentionally excluded from the phrase (paths / DB location).
    Always absent; decode leaves the widget at its runtime default."""

    def cardinality(self, key, reader):
        return 0

    def enc(self, key, reader):
        return 0

    def dec(self, idx, key, reader):
        return None


# ── Per-family slot schemas ───────────────────────────────────────────
@dataclass(frozen=True)
class Slot:
    key: str
    codec: Codec


@dataclass(frozen=True)
class Branch:
    on: str     # cfg key of an earlier Slot whose value picks the arm
    arms: dict  # decoded value -> tuple of Slot/Branch items


# sweep-conditional axis: cfg_sx_fix / _fixvals hold the *fixed* axis, which is
# epoch when sweeping damping and damping when sweeping epoch.
def _sx_fixed_domain(reader) -> str:
    return "epoch" if reader.get("cfg_sx_sweep") == SX_SWEEPS[0] else "damping"


DB_SLOT = Slot("cfg_db", Drop())

SCHEMA: dict[str, tuple] = {
    "LDS sweeps": (
        Slot("cfg_lds_model", Dom("model")),
        Slot("cfg_lds_pts", Dom("sampling")),
        Slot("cfg_lds_strat", Dom("strategy")),
        Slot("cfg_lds_subset", Dom("subset_size")),
        Slot("cfg_lds_variant", Choice(LDS_VARIANTS)),
        Slot("cfg_lds_style", Choice(LDS_STYLES)),  # absent for heatmap
        Slot("cfg_lds_band", BOOL),
        Slot("cfg_lds_annot", BOOL),
        Slot("cfg_lds_keep_epochs", DomSet("epoch")),
        Slot("cfg_lds_keep_damps", DomSet("damping")),
        Branch("cfg_lds_variant", {
            LDS_VARIANTS[0]: (Slot("cfg_lds_mf_methods", Bitmap()),),
            LDS_VARIANTS[1]: (Slot("cfg_lds_df_lam", Dom("damping")),
                              Slot("cfg_lds_df_methods", Bitmap())),
            LDS_VARIANTS[2]: (Slot("cfg_lds_ef_epoch", Dom("epoch")),
                              Slot("cfg_lds_ef_methods", Bitmap())),
            LDS_VARIANTS[3]: (Slot("cfg_lds_hm_methods", Bitmap()),),
        }),
        DB_SLOT,
    ),
    "Metric bars": (
        Slot("cfg_mb_model", Dom("model")),
        Slot("cfg_mb_epoch", Dom("epoch")),
        Slot("cfg_mb_lam", Dom("damping")),
        Slot("cfg_mb_strat", Dom("strategy")),
        Slot("cfg_mb_pts", Dom("sampling")),
        Slot("cfg_mb_ref", Choice(REFERENCES)),
        Slot("cfg_mb_approxs_sel", Bitmap()),
        Slot("cfg_mb_approxs", Perm("cfg_mb_approxs_sel")),
        Slot("cfg_mb_cats", PairSet()),
        DB_SLOT,
    ),
    "Metric correlation": (
        Slot("cfg_mc_scope", Choice(MC_SCOPES)),
        Branch("cfg_mc_scope", {
            MC_SCOPES[0]: (Slot("cfg_mc_model", Dom("model")),
                           Slot("cfg_mc_epoch", Dom("epoch")),
                           Slot("cfg_mc_lam", Dom("damping")),
                           Slot("cfg_mc_strat", Dom("strategy")),
                           Slot("cfg_mc_pts", Dom("sampling"))),
            MC_SCOPES[1]: (Slot("cfg_mc_model", Dom("model")),
                           Slot("cfg_mc_pts", Dom("sampling")),
                           Slot("cfg_mc_strat", Dom("strategy")),
                           Slot("cfg_mc_keep_damps", DomSet("damping"))),
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
            SP_MODES[0]: (Slot("cfg_sp_model", Dom("model")),
                          Slot("cfg_sp_epoch", Dom("epoch")),
                          Slot("cfg_sp_lam", Dom("damping")),
                          Slot("cfg_sp_strat", Dom("strategy")),
                          Slot("cfg_sp_pts", Dom("sampling")),
                          Slot("cfg_sp_methods_sel", Bitmap()),
                          Slot("cfg_sp_methods", Perm("cfg_sp_methods_sel"))),
            SP_MODES[1]: (Slot("cfg_sx_model", Dom("model")),
                          Slot("cfg_sx_pts", Dom("sampling")),
                          Slot("cfg_sx_method", MethodIdx()),
                          Slot("cfg_sx_strat", Dom("strategy")),
                          Slot("cfg_sx_sweep", Choice(SX_SWEEPS)),
                          Slot("cfg_sx_grid", BOOL),
                          Slot("cfg_sx_fix", Dom(_sx_fixed_domain)),         # grid off
                          Slot("cfg_sx_fixvals", DomSet(_sx_fixed_domain))),  # grid on
            SP_MODES[2]: (Slot("cfg_cs_model", Dom("model")),
                          Slot("cfg_cs_a", Dom("sampling")),
                          Slot("cfg_cs_b", Dom("sampling")),
                          Slot("cfg_cs_epoch", Dom("epoch")),
                          Slot("cfg_cs_lam", Dom("damping")),
                          Slot("cfg_cs_strat", Dom("strategy"))),
        }),
        DB_SLOT,
    ),
    "Sample-size Pareto": (
        Slot("cfg_pareto_model", Dom("model")),
        Slot("cfg_pareto_strat", Dom("strategy")),
        Slot("cfg_pareto_epoch", Dom("epoch")),
        Slot("cfg_pareto_lam", Dom("damping")),
        Slot("cfg_pareto_methods", Bitmap()),
        Slot("cfg_pareto_band", BOOL),
        Slot("cfg_pareto_logx", BOOL),
        DB_SLOT,
    ),
    "Factor eigenvalues": (
        # raw-dir fallback path is dropped; the parsed cascade is what encodes.
        Slot("cfg_fe_root", Drop()),
        Slot("cfg_fe_dir", Drop()),
        Slot("cfg_fe_dataset", Dom("dataset")),
        Slot("cfg_fe_model", Dom("model")),
        Slot("cfg_fe_epoch", Dom("epoch")),
        Slot("cfg_fe_pts", Dom("sampling")),
        Slot("cfg_fe_method", MethodIdx()),
        Slot("cfg_fe_logx", BOOL),
        Slot("cfg_fe_layers", DomSet("layer")),
        DB_SLOT,
    ),
}


# ── encode / decode ───────────────────────────────────────────────────
def _walk_slots(items, reader):
    """Yield (codec, key) in schema order, descending the branch arm selected by
    `reader`. Lazy so decode can rely on the discriminator being decoded into
    `reader` (=out) by the time its Branch is reached."""
    for it in items:
        if isinstance(it, Slot):
            yield it.codec, it.key
        else:
            val = reader.get(it.on)
            if val not in it.arms:
                raise ValueError(f"cannot resolve branch on {it.on} = {val!r}")
            yield from _walk_slots(it.arms[val], reader)


def encode(cfg: dict, *, db_default=None) -> str:
    """Encode a tracked-widget CFG dict into a word-phrase code.

    `db_default` is accepted for signature compatibility; DB/path fields are not
    carried in the phrase.
    """
    fam = cfg.get("cfg_family")
    if fam not in SCHEMA:
        raise ValueError(f"unknown plot family {fam!r}")
    # family occupies the least-significant position (decoded first).
    n, place = FAMILIES.index(fam) + 1, len(FAMILIES) + 1
    visited = {"cfg_family"}
    for codec, key in _walk_slots(SCHEMA[fam], cfg):
        visited.add(key)
        radix = codec.radix(key, cfg)
        idx = codec.enc(key, cfg)
        if not 0 <= idx < radix:
            raise ValueError(f"{key}: index {idx} out of radix {radix}")
        n += idx * place
        place *= radix
    extra = set(cfg) - visited
    if extra:  # a new track() key needs a schema slot
        raise ValueError(f"cfg keys missing from the schema: {sorted(extra)}")
    return int_to_phrase(n)


def decode(code: str) -> dict:
    """Invert encode(): a {cfg_key: value} dict ready for _pending_cfg."""
    n = phrase_to_int(code)
    fam_radix = len(FAMILIES) + 1
    n, fam_idx = n // fam_radix, n % fam_radix
    if fam_idx == 0 or fam_idx > len(FAMILIES):
        raise ValueError("phrase does not start with a valid plot family")
    fam = FAMILIES[fam_idx - 1]
    out: dict = {"cfg_family": fam}
    for codec, key in _walk_slots(SCHEMA[fam], out):
        radix = codec.radix(key, out)
        n, idx = n // radix, n % radix
        codec.dec(idx, key, out)
    if n != 0:
        raise ValueError("phrase has trailing data for this plot family")
    return out


# ── Import-time sanity checks ─────────────────────────────────────────
def _check_tables():
    assert set(SCHEMA) == set(FAMILIES), "SCHEMA / FAMILIES mismatch"
    assert len(_PAIRS) == len(set(_PAIRS)), "duplicate pair in _PAIRS"
    assert LDS_AXIS in _PAIRS
    for domain in ("model", "dataset", "sampling", "strategy", "epoch",
                   "damping", "subset_size", "layer"):
        assert VOCAB.get(domain), f"code_vocab.json missing/empty domain {domain!r}"


_check_tables()
