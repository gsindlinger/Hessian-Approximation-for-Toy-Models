"""LDS analysis driver — score every (model, epoch, method) attribution
recorded in a results.json from analyze_hessians.py.

The script is purely a *consumer* of attributions: for each entry in
results.json that has an ``hessian_analysis.influence.paths`` block, it loads
the saved .npy attribution matrix, runs ELSO retraining (cached per
``(model_directory, recipe)``, so methods within the same checkpoint share
the retraining cost), and writes the resulting LDS scores to:

  - ``<run_dir>/lds.json`` (sibling of the input results.json), and
  - the matching row of the ``influence`` table in ``runs.db``
    (columns ``lds_mean`` / ``lds_std`` / ``lds_ci_low`` / ``lds_ci_high`` /
    ``lds_num_valid_queries``).

Examples:

    # Use dataclass defaults — no recipe yaml.
    python -m experiments.lds_analysis \\
        --results-json experiments/outputs/runs/<run_id>/results.json

    # With a recipe yaml + CLI overrides.
    python -m experiments.lds_analysis \\
        --results-json experiments/outputs/runs/<run_id>/results.json \\
        --config experiments/configs/lds_analysis.yaml \\
        --num-subsets 200 --method ekfac --method exact --epoch 100
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from experiments import provenance, results as results_db
from src.config import LDSConfig, LDSFilter
from src.utils.lds import compute_lds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading: yaml (optional) + CLI overrides
# ---------------------------------------------------------------------------


def _load_config(yaml_path: Optional[str]) -> LDSConfig:
    """Build an LDSConfig: dataclass defaults, optionally overlaid with yaml."""
    if yaml_path is None:
        return LDSConfig()
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    filter_raw = raw.pop("filter", None) or {}
    return LDSConfig(filter=LDSFilter(**filter_raw), **raw)


def _apply_cli_overrides(cfg: LDSConfig, args: argparse.Namespace) -> LDSConfig:
    """Mutate ``cfg`` with non-None CLI flags. Filter args replace yaml lists
    entirely when supplied — this matches the user expectation that
    ``--method ekfac`` on the command line means "this method," not "in
    addition to whatever the yaml said."
    """
    if args.num_subsets is not None:
        cfg.num_subsets = args.num_subsets
    if args.reps_per_model is not None:
        cfg.reps_per_model = args.reps_per_model
    if args.subset_fraction is not None:
        cfg.subset_fraction = args.subset_fraction
    if args.num_test_examples is not None:
        cfg.num_test_examples = args.num_test_examples
    if args.seed is not None:
        cfg.lds_seed = args.seed
    if args.no_cache:
        cfg.cache_elso = False

    if args.model_id:
        cfg.filter.model_ids = list(args.model_id)
    if args.epoch:
        cfg.filter.epochs = list(args.epoch)
    if args.method:
        cfg.filter.methods = list(args.method)
    return cfg


# ---------------------------------------------------------------------------
# DB write-back
# ---------------------------------------------------------------------------


def _write_lds_to_db(run_id: Optional[str], results: List[Dict[str, Any]]) -> None:
    if not run_id:
        logger.warning(
            "[LDS] results.json has no run_id; skipping db write-back"
        )
        return
    con = results_db.init_db()
    try:
        for r in results:
            result_id = results_db.find_result_id(
                con, run_id=run_id, model_id=r["model_id"], epoch=r.get("epoch"),
            )
            if result_id is None:
                logger.warning(
                    "[LDS] no influence row for run=%s model=%s epoch=%s — skipping db update",
                    run_id, r["model_id"], r.get("epoch"),
                )
                continue
            scores = r["lds_scores"]
            updated = results_db.update_lds_scores(
                con,
                result_id=result_id,
                method=r["method"],
                mean=scores.get("mean_lds"),
                std=scores.get("std_lds"),
                ci_low=scores.get("ci_low"),
                ci_high=scores.get("ci_high"),
                num_valid=scores.get("num_valid_queries"),
            )
            if updated == 0:
                logger.warning(
                    "[LDS] no influence row for result_id=%s method=%s — skipping",
                    result_id, r["method"],
                )
    finally:
        con.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score attributions in a results.json via ELSO LDS."
    )
    p.add_argument(
        "--results-json", required=True,
        help="Path to the results.json produced by analyze_hessians.py.",
    )
    p.add_argument(
        "--config", default=None,
        help="Optional LDS recipe yaml. Omit to use LDSConfig() defaults.",
    )
    p.add_argument("--num-subsets",       type=int,   default=None)
    p.add_argument("--reps-per-model",    type=int,   default=None)
    p.add_argument("--subset-fraction",   type=float, default=None)
    p.add_argument("--num-test-examples", type=int,   default=None)
    p.add_argument("--seed",              type=int,   default=None)
    p.add_argument(
        "--no-cache", action="store_true",
        help="Disable the ELSO retrain cache.",
    )
    p.add_argument(
        "--model-id", action="append", default=[],
        help="Filter: include only this model_id (repeatable).",
    )
    p.add_argument(
        "--epoch", type=int, action="append", default=[],
        help="Filter: include only this epoch (repeatable).",
    )
    p.add_argument(
        "--method", action="append", default=[],
        help="Filter: include only this method (repeatable).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output lds.json path (default: sibling of results-json).",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()

    results_json_path = Path(args.results_json).resolve()
    cfg = _apply_cli_overrides(_load_config(args.config), args)

    out = compute_lds(results_json=str(results_json_path), config=cfg)

    out["code"] = provenance.git_info()
    out["timestamp"] = time.strftime("%Y%m%d-%H%M%S")
    out["lds_config"] = asdict(cfg)  # re-stamp post-overrides for provenance

    output_path = (
        Path(args.output) if args.output
        else results_json_path.parent / "lds.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("[LDS] wrote %s", output_path)

    _write_lds_to_db(out.get("run_id"), out["results"])

    summary = out["results"]
    if summary:
        logger.info("=" * 70)
        for r in summary:
            s = r["lds_scores"]
            logger.info(
                "[LDS] %s epoch=%s method=%-12s mean=%.4f ± %.4f  CI=[%.4f, %.4f]  valid=%d",
                r["model_id"], r.get("epoch"), r["method"],
                s.get("mean_lds", float("nan")),
                s.get("std_lds", float("nan")),
                s.get("ci_low", float("nan")),
                s.get("ci_high", float("nan")),
                s.get("num_valid_queries", -1),
            )
        logger.info("=" * 70)
    if out["skipped"]:
        logger.info("[LDS] skipped %d entries (no influence paths)", len(out["skipped"]))


if __name__ == "__main__":
    main()
