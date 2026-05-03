#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# MODELS_CONFIG="experiments/shared_models.yaml"
# ANALYSIS_CONFIG="experiments/configs/hessian_sweep.yaml"

MODELS_CONFIG="experiments/shared_models.yaml"
ANALYSIS_CONFIG="experiments/configs/approximator_sweep.yaml"

# Positional args (only if they don't start with '-'); rest are forwarded.
#   ./new_run.sh --skip-if-exists                  → defaults + flag
#   ./new_run.sh m.yaml --skip-if-exists           → custom models + flag
#   ./new_run.sh m.yaml a.yaml --skip-if-exists    → both custom + flag
if [[ "${1:-}" != "" && "${1}" != -* ]]; then
    MODELS_CONFIG="$1"; shift
fi
if [[ "${1:-}" != "" && "${1}" != -* ]]; then
    ANALYSIS_CONFIG="$1"; shift
fi

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

python -m experiments.analyze_hessians \
    --config "$MODELS_CONFIG" \
    --analysis-config "$ANALYSIS_CONFIG" \
    "$@"
