"""SQLite-backed results store.

One file at `experiments/outputs/runs.db` holding every analysis run, every
result, and every metric in long format. Pandas can read it directly via
`pd.read_sql_query(..., con)`.

Schema (4 tables):

  runs       — one row per `analyze_hessians.py` invocation
  results    — one row per (run, model_id, epoch)
  metrics    — one row per metric value (long format, naturally extensible)
  influence  — one row per (result, approximator) -> npy path

Foreign keys cascade: deleting a run removes its results, which removes their
metrics and influence rows.

Schema evolution: when you want to filter on a knob currently inside
`params_json`, ALTER TABLE ADD COLUMN and backfill from the blob in one query.
Until then it's queryable via `json_extract(params_json, '$.key')`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experiments.paths import RESULTS_DB

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    timestamp             TEXT NOT NULL,
    code_sha              TEXT,
    code_branch           TEXT,
    models_config_path    TEXT,
    analysis_config_path  TEXT,
    params_json           TEXT
);

CREATE TABLE IF NOT EXISTS results (
    result_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                    TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    -- identity
    model_id                  TEXT NOT NULL,
    epoch                     INTEGER,
    config_hash               TEXT NOT NULL,
    model_hash                TEXT NOT NULL,
    code_sha                  TEXT,
    -- run-level swept axes (denormalized for query ergonomics)
    analysis_seed             INTEGER,
    dataset_name              TEXT,
    dataset_test_size         REAL,
    collector_subset_size     INTEGER,
    damping_value             REAL,
    damping_strategy          TEXT,
    pseudo_target_strategy    TEXT,
    pseudo_target_repetitions INTEGER,
    vector_num_samples        INTEGER,
    sampling_method           TEXT,
    comparison_probe_source   TEXT,
    -- per-result outputs / metadata
    pseudo_inverse_factor     REAL,
    num_parameters            INTEGER,
    val_loss                  REAL,
    val_accuracy              REAL,
    params_json               TEXT,                       -- model_config + metadata, etc.
    UNIQUE(config_hash, code_sha)                          -- INSERT OR REPLACE keys on this
);
CREATE INDEX IF NOT EXISTS idx_results_config_hash ON results(config_hash);
CREATE INDEX IF NOT EXISTS idx_results_model       ON results(model_id, epoch);

CREATE TABLE IF NOT EXISTS metrics (
    result_id         INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    computation_type  TEXT NOT NULL,         -- 'matrix' | 'hvp' | 'ihvp' | 'round_trip'
    reference         TEXT,                  -- 'exact' | 'gnh' | NULL for round_trip
    approximator      TEXT NOT NULL,
    metric            TEXT NOT NULL,
    value             REAL
);
CREATE INDEX IF NOT EXISTS idx_metrics_result ON metrics(result_id);
CREATE INDEX IF NOT EXISTS idx_metrics_approx ON metrics(approximator);
CREATE INDEX IF NOT EXISTS idx_metrics_metric ON metrics(metric);

CREATE TABLE IF NOT EXISTS influence (
    result_id  INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    method     TEXT NOT NULL,
    npy_path   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_influence_result ON influence(result_id);
"""


def init_db(db_path: Path | str = RESULTS_DB) -> sqlite3.Connection:
    """Open (creating if needed) the results db. Caller closes."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA)
    return con


# -- writes -----------------------------------------------------------------

def append_run(
    con: sqlite3.Connection,
    *,
    run_id: str,
    code_info: Dict[str, Any],
    models_config_path: str,
    analysis_config_path: str,
    params: Dict[str, Any],
) -> None:
    con.execute(
        """INSERT INTO runs (run_id, timestamp, code_sha, code_branch,
                             models_config_path, analysis_config_path, params_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            run_id,
            code_info.get("sha"),
            code_info.get("branch"),
            models_config_path,
            analysis_config_path,
            json.dumps(params, default=str, sort_keys=True),
        ),
    )
    con.commit()


def append_result(
    con: sqlite3.Connection,
    *,
    run_id: str,
    model_id: str,
    epoch: Optional[int],
    config_hash: str,
    model_hash: str,
    code_sha: Optional[str] = None,
    analysis_seed: Optional[int] = None,
    dataset_name: Optional[str] = None,
    dataset_test_size: Optional[float] = None,
    collector_subset_size: Optional[int] = None,
    damping_value: Optional[float] = None,
    damping_strategy: Optional[str] = None,
    pseudo_target_strategy: Optional[str] = None,
    pseudo_target_repetitions: Optional[int] = None,
    vector_num_samples: Optional[int] = None,
    sampling_method: Optional[str] = None,
    comparison_probe_source: Optional[str] = None,
    pseudo_inverse_factor: Optional[float] = None,
    num_parameters: Optional[int] = None,
    val_loss: Optional[float] = None,
    val_accuracy: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert (or replace) a result row, keyed on (config_hash, code_sha).

    A rerun with the same pair overwrites the old row; the cascade on
    `metrics`/`influence` then drops the stale child rows so the new metrics
    have a clean slot.
    """
    cur = con.execute(
        """INSERT OR REPLACE INTO results
              (run_id, model_id, epoch, config_hash, model_hash, code_sha,
               analysis_seed, dataset_name, dataset_test_size,
               collector_subset_size, damping_value, damping_strategy,
               pseudo_target_strategy, pseudo_target_repetitions,
               vector_num_samples, sampling_method, comparison_probe_source,
               pseudo_inverse_factor, num_parameters,
               val_loss, val_accuracy, params_json)
           VALUES (?, ?, ?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?,  ?, ?, ?,
                   ?, ?, ?, ?, ?)""",
        (
            run_id, model_id, epoch, config_hash, model_hash, code_sha,
            analysis_seed, dataset_name, dataset_test_size,
            collector_subset_size, damping_value, damping_strategy,
            pseudo_target_strategy, pseudo_target_repetitions,
            vector_num_samples, sampling_method, comparison_probe_source,
            pseudo_inverse_factor, num_parameters,
            val_loss, val_accuracy,
            json.dumps(params or {}, default=str, sort_keys=True),
        ),
    )
    con.commit()
    return int(cur.lastrowid)


def append_metrics(
    con: sqlite3.Connection,
    *,
    result_id: int,
    rows: Iterable[Dict[str, Any]],
) -> None:
    """Each row: {computation_type, reference (optional), approximator, metric, value}."""
    payload = [
        (
            result_id,
            r["computation_type"],
            r.get("reference"),
            r["approximator"],
            r["metric"],
            None if r["value"] is None else float(r["value"]),
        )
        for r in rows
    ]
    if payload:
        con.executemany(
            """INSERT INTO metrics (result_id, computation_type, reference,
                                    approximator, metric, value)
               VALUES (?, ?, ?, ?, ?, ?)""",
            payload,
        )
        con.commit()


def append_influence(
    con: sqlite3.Connection,
    *,
    result_id: int,
    method_to_path: Dict[str, str],
) -> None:
    payload = [(result_id, m, p) for m, p in method_to_path.items()]
    if payload:
        con.executemany(
            "INSERT INTO influence (result_id, method, npy_path) VALUES (?, ?, ?)",
            payload,
        )
        con.commit()


# -- reads ------------------------------------------------------------------

def result_exists(
    con: sqlite3.Connection,
    *,
    config_hash: str,
    code_sha: str,
) -> bool:
    """Has a result with this exact (config_hash, code_sha) already been
    recorded? The dedupe key for `--skip-if-exists`."""
    cur = con.execute(
        """SELECT 1 FROM results WHERE config_hash = ? AND code_sha = ? LIMIT 1""",
        (config_hash, code_sha),
    )
    return cur.fetchone() is not None


# -- flattening helpers -----------------------------------------------------

def flatten_metric_dicts(
    *,
    matrix_comparisons: Dict[str, Dict[str, Dict[str, float]]],
    hvp_comparisons: Dict[str, Dict[str, Dict[str, float]]],
    ihvp_comparisons: Dict[str, Dict[str, Dict[str, float]]],
    round_trip_errors: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Convert the nested dicts produced by analyze_hessians into long-format
    metric rows ready for `append_metrics`.

    Input shape per comparison: `{metric: {reference: {approximator: value}}}`.
    Round-trip is `{reference: {approximator: value}}` (no metric axis).
    """
    rows: List[Dict[str, Any]] = []

    def _emit(comp_type: str, comp: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        for metric, by_ref in comp.items():
            for reference, by_approx in by_ref.items():
                for approximator, value in by_approx.items():
                    rows.append({
                        "computation_type": comp_type,
                        "reference": reference,
                        "approximator": approximator,
                        "metric": metric,
                        "value": value,
                    })

    _emit("matrix", matrix_comparisons)
    _emit("hvp", hvp_comparisons)
    _emit("ihvp", ihvp_comparisons)

    for reference, by_approx in round_trip_errors.items():
        for approximator, value in by_approx.items():
            rows.append({
                "computation_type": "round_trip",
                "reference": reference,
                "approximator": approximator,
                "metric": "relative_error",
                "value": value,
            })

    return rows
