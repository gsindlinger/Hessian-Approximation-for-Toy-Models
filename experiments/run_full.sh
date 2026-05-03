#!/bin/bash
# Run analyze_hessians on the configured models/analysis YAMLs, then run LDS
# on the produced results.json. Pins the whole job to a single GPU.
#
# Usage:
#   ./run_full.sh --gpu 2
#   ./run_full.sh --gpu 2 m.yaml a.yaml --skip-if-exists
#   ./run_full.sh --gpu 2 -- --override analysis.computation_config.damping_strategy=pseudo_inverse
#
# Anything after --gpu N is forwarded to new_run.sh (positional model/analysis
# YAMLs first, then flags). LDS is invoked with the default lds.yaml recipe;
# override via LDS_CONFIG=<path> ./run_full.sh ...

set -euo pipefail

cd "$(dirname "$0")/.."

GPU=""
if [[ "${1:-}" == "--gpu" ]]; then
    GPU="$2"
    shift 2
fi
if [[ -z "$GPU" ]]; then
    echo "error: --gpu N required (first arg)" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"

LDS_CONFIG="${LDS_CONFIG:-experiments/configs/lds.yaml}"

# Stream the analyze_hessians run through tee so we can both watch it live and
# grep the final "wrote results → <path>" line for the produced results.json.
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

echo "[run_full] GPU=$GPU launching analyze_hessians..."
./experiments/new_run.sh "$@" 2>&1 | tee "$LOG"

RESULTS_JSON=$(grep -oE "wrote results → \S+" "$LOG" | tail -1 | awk '{print $NF}')
if [[ -z "$RESULTS_JSON" || ! -f "$RESULTS_JSON" ]]; then
    echo "[run_full] could not locate results.json from analyze_hessians log" >&2
    exit 1
fi

echo
echo "[run_full] analyze_hessians done → $RESULTS_JSON"
echo "[run_full] launching LDS with $LDS_CONFIG"
echo

CONFIG="$LDS_CONFIG" ./experiments/run_lds.sh "$RESULTS_JSON"
