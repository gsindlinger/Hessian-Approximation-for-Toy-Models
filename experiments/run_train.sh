#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

TRAINING_CONFIG_NAME="short_config"
TRAINING_CONFIG_PATH="$PROJECT_ROOT/experiments/configs"

for arg in "$@"; do
  eval "$arg"
done

echo "Starting training sweep with config: $TRAINING_CONFIG_NAME"
echo "Using config path: $TRAINING_CONFIG_PATH"

BEST_MODELS_PATH=$(uv run python -m experiments.train_models \
    --config-name="$TRAINING_CONFIG_NAME" \
    --config-path="$TRAINING_CONFIG_PATH" \
    hydra.run.dir=experiments/logs/training/$TRAINING_CONFIG_NAME/$(date +%Y%m%d-%H%M%S) | \
    tee /dev/tty | sed -n 's/^BEST_MODELS_YAML=//p')

if [ -z "$BEST_MODELS_PATH" ]; then
    echo "Error: Could not capture BEST_MODELS_YAML path."
    exit 1
fi

echo "Training complete. Path captured: $BEST_MODELS_PATH"
echo "BEST_MODELS_YAML=$BEST_MODELS_PATH"
