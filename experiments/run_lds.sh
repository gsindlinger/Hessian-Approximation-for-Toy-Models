#!/bin/bash
# Run LDS pipeline: train models with epoch checkpoints, then evaluate LDS per epoch.
#
# Usage:
#   ./experiments/run_lds.sh
#   ./experiments/run_lds.sh TRAINING_CONFIG_NAME=lds_model_sweep LDS_CONFIG_NAME=digits_lds_simple
#   ./experiments/run_lds.sh LDS_EPOCHS=[10,100,1000]
set -e

TRAINING_CONFIG_NAME="digits_lds_sweep"
TRAINING_CONFIG_PATH="./configs"
LDS_CONFIG_NAME="lds_analysis"
LDS_CONFIG_PATH="./configs"
LDS_EPOCHS="[10,100,1000]"

for arg in "$@"; do
  eval "$arg"
done

echo "========================================"
echo "LDS Pipeline"
echo "  Training config : $TRAINING_CONFIG_NAME"
echo "  LDS config      : $LDS_CONFIG_NAME"
echo "  Epochs          : $LDS_EPOCHS"
echo "========================================"

# ── Step 1: Train models, saving checkpoints at each epoch ───────────────────
echo ""
echo "=== Step 1: Training models ==="
BEST_MODELS_PATH=$(uv run python -m experiments.train_models \
    --config-name="$TRAINING_CONFIG_NAME" \
    --config-path="$TRAINING_CONFIG_PATH" \
    hydra.run.dir=experiments/logs/training/$TRAINING_CONFIG_NAME/$(date +%Y%m%d-%H%M%S) | \
    tee /dev/tty | sed -n 's/^BEST_MODELS_YAML=//p')

if [ -z "$BEST_MODELS_PATH" ]; then
    echo "ERROR: Could not capture BEST_MODELS_YAML path from training output."
    exit 1
fi
echo "Training complete. Best models YAML: $BEST_MODELS_PATH"

# ── Step 2: LDS analysis at each epoch checkpoint ────────────────────────────
echo ""
echo "=== Step 2: LDS analysis (epochs=$LDS_EPOCHS) ==="
uv run python -m experiments.lds_analysis \
    --config-name="$LDS_CONFIG_NAME" \
    --config-path="$LDS_CONFIG_PATH" \
    hydra.run.dir=experiments/logs/lds_analysis/$TRAINING_CONFIG_NAME/$(date +%Y%m%d-%H%M%S) \
    +override_config="$BEST_MODELS_PATH" \
    epochs="$LDS_EPOCHS"

echo ""
echo "========================================"
echo "LDS Pipeline complete."
echo "========================================"
