#!/bin/bash
set -e

CONFIG_NAME="${1:-concrete_sweep}"  # default if not provided

echo "Starting training sweep with config: $CONFIG_NAME"

BEST_MODELS_PATH=$(uv run python -m experiments.sweep_1.scripts.train_models \
    --config-name="$CONFIG_NAME" \
    --config-path=../configs \
    hydra.run.dir=experiments/sweep_1/logs/training/$(date +%Y%m%d-%H%M%S) | \
    tee /dev/tty | sed -n 's/^BEST_MODELS_YAML=//p')

if [ -z "$BEST_MODELS_PATH" ]; then
    echo "Error: Could not capture BEST_MODELS_YAML path."
    exit 1
fi

echo "Training complete. Path captured: $BEST_MODELS_PATH"

# -----------------------------
# Set NUM_SAMPLES based on config name
# -----------------------------
if [[ "$CONFIG_NAME" == *concrete* || "$CONFIG_NAME" == *energy* ]]; then
    NUM_SAMPLES=500
else
    NUM_SAMPLES=1000
fi

echo "Using NUM_SAMPLES=$NUM_SAMPLES"


uv run python -m experiments.sweep_1.scripts.hessian_analysis \
    --config-name=hessian_analysis \
    --config-path=../configs \
    hydra.run.dir=experiments/sweep_1/logs/hessian_analysis/$(date +%Y%m%d-%H%M%S) \
    hessian_analysis.vector_config.num_samples="$NUM_SAMPLES" \
    +override_config="$BEST_MODELS_PATH"
