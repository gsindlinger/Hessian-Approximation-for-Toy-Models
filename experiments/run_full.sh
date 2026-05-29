#!/bin/bash
# Run analyze_hessians + LDS end-to-end, pinned to a single GPU.
#
# Usage:
#   ./experiments/run_full.sh --gpu <N> <models.yaml> <analysis.yaml> [flags...]
#
# Examples:
#   ./experiments/run_full.sh --gpu 0 models.yaml analysis.yaml
#   ./experiments/run_full.sh --gpu 2 models.yaml analysis.yaml --skip-if-exists
#
# LDS uses experiments/configs/lds.yaml by default; override via LDS_CONFIG env var.
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
