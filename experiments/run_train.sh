#!/bin/bash
# Train models from a YAML config and capture the best-models output path.
#
# Usage:
#   ./experiments/run_train.sh <config.yaml>
#
# Examples:
#   ./experiments/run_train.sh experiments/configs/recovered/mlp_08580ee2573a.yaml
#   ./experiments/run_train.sh experiments/configs/digits_sweep.yaml
#
# The script prints BEST_MODELS_YAML=<path> on success, which downstream
# scripts (e.g. new_run.sh) can consume.
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:?Usage: $0 <config.yaml>}"
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
CONFIG_NAME="$(basename "$CONFIG_PATH" .yaml)"
CONFIG_DIR="$(dirname "$CONFIG_PATH")"

echo "Training with config: $CONFIG_PATH"

BEST_MODELS_PATH=$(python -m experiments.train_models \
    --config-name="$CONFIG_NAME" \
    --config-path="$CONFIG_DIR" \
    hydra.run.dir=experiments/logs/training/$CONFIG_NAME/$(date +%Y%m%d-%H%M%S) | \
    tee /dev/tty | sed -n 's/^BEST_MODELS_YAML=//p')

if [ -z "$BEST_MODELS_PATH" ]; then
    echo "Error: Could not capture BEST_MODELS_YAML path."
    exit 1
fi

echo "Training complete. Path captured: $BEST_MODELS_PATH"
echo "BEST_MODELS_YAML=$BEST_MODELS_PATH"
