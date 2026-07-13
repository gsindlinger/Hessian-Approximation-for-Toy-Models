"""Build / refresh `code_vocab.json` — the single canonical ordering that the
word-phrase config codes index into (see config_code.py).

Every data-derived value domain (models, epochs, dampings, subset sizes, …) is
stored here as an **ordered list**; a value's code is its position in the list.
The file is **append-only**: this script never reorders or drops an existing
entry (that would silently change the meaning of already-shared phrases), it
only appends values it hasn't seen. Run it after new experiments introduce new
models / epochs / dampings / etc.:

    python plots/build_code_vocab.py
    # or point at another DB:
    python plots/build_code_vocab.py --db /path/to/runs.db
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hessian_data as D

VOCAB_PATH = Path(__file__).resolve().parent / "code_vocab.json"


def _vkey(v) -> str:
    """Stable string key for de-duping a value across int/float/str/numpy."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _scan_db(db_path: Path) -> dict[str, list]:
    df = D.load_runs_db(db_path)

    def uniq(col, cast=None):
        vals = [v for v in df[col].dropna().unique().tolist()]
        if cast is not None:
            vals = [cast(v) for v in vals]
        return vals

    found: dict[str, list] = {
        "model": sorted(uniq("model_id")),
        "dataset": sorted(uniq("dataset_name")),
        "sampling": sorted(uniq("pseudo_target_strategy")),
        "strategy": sorted(uniq("damping_strategy")),
        "epoch": sorted(uniq("epoch", int)),
        "damping": sorted(uniq("damping_value", float)),
        "subset_size": sorted(uniq("collector_subset_size", int)),
    }
    return found


def _scan_layers(db_path: Path) -> list[str]:
    """Diagonal-block layer names across all factor dirs (for cfg_fe_layers)."""
    root = db_path.parent / "models"
    names: set[str] = {"full matrix"}  # off-diagonal factors collapse to this
    for d in D.find_factor_dirs(root):
        manifest = Path(d) / "manifest.json"
        if not manifest.exists():
            continue
        try:
            blocks = json.loads(manifest.read_text()).get("blocks", [])
        except (ValueError, OSError):
            continue
        for blk in blocks:
            a, _, b = blk.get("key", "").partition("::")
            if a and a == b:
                names.add(a)
    return sorted(names)


def _merge_append(existing: list, found: list) -> list:
    """Keep existing order; append values in `found` not already present."""
    seen = {_vkey(v) for v in existing}
    out = list(existing)
    for v in found:
        if _vkey(v) not in seen:
            out.append(v)
            seen.add(_vkey(v))
    return out


def build(db_path: Path) -> dict[str, list]:
    current: dict[str, list] = {}
    if VOCAB_PATH.exists():
        current = json.loads(VOCAB_PATH.read_text())

    found = _scan_db(db_path)
    found["layer"] = _scan_layers(db_path)

    merged = dict(current)
    for domain, values in found.items():
        merged[domain] = _merge_append(current.get(domain, []), values)
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(D.DB_PATH))
    args = p.parse_args()

    vocab = build(Path(args.db))
    VOCAB_PATH.write_text(json.dumps(vocab, indent=2, ensure_ascii=False) + "\n")
    sizes = ", ".join(f"{k}={len(v)}" for k, v in vocab.items())
    print(f"wrote {VOCAB_PATH}\n  {sizes}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
