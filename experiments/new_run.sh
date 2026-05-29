#!/bin/bash
# Run Hessian analysis on trained models.
#
# Usage:
#   ./experiments/new_run.sh <models.yaml> <analysis.yaml> [flags...]
#
# Examples:
#   ./experiments/new_run.sh models.yaml analysis.yaml
#   ./experiments/new_run.sh models.yaml analysis.yaml --skip-if-exists
#   ./experiments/new_run.sh experiments/configs/recovered_analysis/20260503-221136/models.yaml \
#                            experiments/configs/recovered_analysis/20260503-221136/analysis.yaml
#
# Extra flags (e.g. --skip-if-exists, --override ...) are forwarded to
# analyze_hessians.py.
set -e

cd "$(dirname "$0")/.."

MODELS_CONFIG="${1:?Usage: $0 <models.yaml> <analysis.yaml> [flags...]}"
ANALYSIS_CONFIG="${2:?Usage: $0 <models.yaml> <analysis.yaml> [flags...]}"
shift 2

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

python -m experiments.analyze_hessians \
    --config "$MODELS_CONFIG" \
    --analysis-config "$ANALYSIS_CONFIG" \
    "$@"
