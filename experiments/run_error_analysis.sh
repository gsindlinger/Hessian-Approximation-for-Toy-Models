#!/bin/bash
# Run error-analysis-only:
#   1. Train models unless BEST_MODELS_PATH is provided.
#   2. Run experiments.analysis with stages=[error_analysis].
#
# Usage:
#   ./experiments/run_error_analysis.sh
#   ./experiments/run_error_analysis.sh TRAINING_CONFIG_NAME=digits_lds_sweep
#   ./experiments/run_error_analysis.sh BEST_MODELS_PATH=experiments/data/results/.../best_models/...yaml
#   ./experiments/run_error_analysis.sh ANALYSIS_EPOCHS='[10,100,1000]'
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

TRAINING_CONFIG_NAME="digits_lds_sweep"
TRAINING_CONFIG_PATH="$PROJECT_ROOT/experiments/configs"
ANALYSIS_CONFIG_NAME="analysis"
ANALYSIS_CONFIG_PATH="$PROJECT_ROOT/experiments/configs"
ANALYSIS_EPOCHS="[10,100,1000]"
ANALYSIS_STAGES="[error_analysis]"
BEST_MODELS_PATH=""

for arg in "$@"; do
  eval "$arg"
done

echo "========================================"
echo "Error Analysis Pipeline"
echo "  Training config  : $TRAINING_CONFIG_NAME"
echo "  Analysis config  : $ANALYSIS_CONFIG_NAME"
echo "  Epochs           : $ANALYSIS_EPOCHS"
echo "  Stages           : $ANALYSIS_STAGES"
echo "  Best models YAML : ${BEST_MODELS_PATH:-<train first>}"
echo "========================================"

if [ -z "$BEST_MODELS_PATH" ]; then
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
else
  echo ""
  echo "=== Step 1: Skipping training ==="
  echo "Using existing best models YAML: $BEST_MODELS_PATH"
fi

if [ ! -f "$BEST_MODELS_PATH" ]; then
  echo "ERROR: Best models YAML does not exist: $BEST_MODELS_PATH"
  exit 1
fi

echo ""
echo "=== Step 2: Analysis (stages=$ANALYSIS_STAGES) ==="
uv run python -m experiments.analysis \
  --config-name="$ANALYSIS_CONFIG_NAME" \
  --config-path="$ANALYSIS_CONFIG_PATH" \
  "hydra.run.dir=experiments/logs/analysis/$TRAINING_CONFIG_NAME/$(date +%Y%m%d-%H%M%S)" \
  "+override_config=$BEST_MODELS_PATH" \
  "stages=$ANALYSIS_STAGES" \
  "epochs=$ANALYSIS_EPOCHS"

echo ""
echo "========================================"
echo "Error analysis pipeline complete."
echo "========================================"
