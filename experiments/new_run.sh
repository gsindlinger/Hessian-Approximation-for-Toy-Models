#!/bin/bash
set -e

cd "$(dirname "$0")/.."

MODELS_CONFIG="${1:-experiments/shared_models.yaml}"
ANALYSIS_CONFIG="${2:-experiments/configs/hessian_analysis.yaml}"

export TF_CPP_MIN_LOG_LEVEL=3

python -m experiments.analyze_hessians --config "$MODELS_CONFIG" --analysis-config "$ANALYSIS_CONFIG"
