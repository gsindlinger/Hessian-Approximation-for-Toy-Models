#!/bin/bash
set -e

# Defaults
TRAINING_CONFIG_NAME="digits_sweep"
TRAINING_CONFIG_PATH="./configs"

# Override defaults if provided
TRAINING_CONFIG_NAME="${TRAINING_CONFIG_NAME:-${TRAINING_CONFIG_NAME}}"
TRAINING_CONFIG_PATH="${TRAINING_CONFIG_PATH:-${TRAINING_CONFIG_PATH}}"
# Read CLI args directly
for arg in "$@"; do
  eval "$arg"
done

echo "Starting training sweep with config: $TRAINING_CONFIG_NAME"
echo "Using config path: $TRAINING_CONFIG_PATH"

# -----------------------------
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

uv run python -m experiments.hessian_analysis \
    --config-name=hessian_analysis \
    --config-path=./configs \
    hydra.run.dir=experiments/logs/hessian_analysis/$TRAINING_CONFIG_NAME/$(date +%Y%m%d-%H%M%S) \
    +override_config="$BEST_MODELS_PATH"
