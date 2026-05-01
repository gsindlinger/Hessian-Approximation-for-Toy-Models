#!/bin/bash
# Score one or more analyze_hessians results.json files via ELSO LDS,
# using the recipe in experiments/configs/lds.yaml (paper defaults).
#
# Usage:
#   bash experiments/run_lds.sh                                  # uses DEFAULT_RESULTS
#   bash experiments/run_lds.sh <results.json> [<results.json> ...]
#
# Default config is experiments/configs/lds.yaml; override via CONFIG env var.

set -e
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-experiments/configs/lds.yaml}"

DEFAULT_RESULTS=(
/root/Hessian-Approximation-for-Toy-Models/experiments/outputs/runs/20260501-150319/results.json)

if [[ $# -eq 0 ]]; then
    set -- "${DEFAULT_RESULTS[@]}"
fi

args=()
for p in "$@"; do
    args+=(--results-json "$p")
done

python -m experiments.lds_analysis --config "$CONFIG" "${args[@]}"
