#!/bin/bash
# Run the full analysis pipeline:
#   1. Train models unless BEST_MODELS_PATH is provided.
#   2. Run experiments.analysis with stages=[error_analysis,attribution,lds].
#
# Usage:
#   ./experiments/run_all.sh
#   ./experiments/run_all.sh TRAINING_CONFIG_NAME=digits_lds_sweep
#   ./experiments/run_all.sh BEST_MODELS_PATH=experiments/data/results/.../best_models/...yaml
#   ./experiments/run_all.sh ANALYSIS_EPOCHS='[10,100,1000]'
#   ./experiments/run_all.sh ATTRIBUTION_OUTPUT_DIR=/shared/attributions
#   ./experiments/run_all.sh LDS_ATTRIBUTION_DIRS='[/shared/attr_a,/shared/attr_b]'
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

TRAINING_CONFIG_NAME="digits_lds_sweep"
TRAINING_CONFIG_PATH="$PROJECT_ROOT/experiments/configs"
ANALYSIS_CONFIG_NAME="analysis"
ANALYSIS_CONFIG_PATH="$PROJECT_ROOT/experiments/configs"
ANALYSIS_EPOCHS="[10,100,1000]"
ANALYSIS_STAGES="[error_analysis,attribution,lds]"
BEST_MODELS_PATH=""
ATTRIBUTION_OUTPUT_DIR=""
LDS_ATTRIBUTION_DIRS=""

for arg in "$@"; do
  eval "$arg"
done

echo "========================================"
echo "Full Analysis Pipeline"
echo "  Training config     : $TRAINING_CONFIG_NAME"
echo "  Analysis config     : $ANALYSIS_CONFIG_NAME"
echo "  Epochs              : $ANALYSIS_EPOCHS"
echo "  Stages              : $ANALYSIS_STAGES"
echo "  Best models YAML    : ${BEST_MODELS_PATH:-<train first>}"
echo "  Attribution out dir : ${ATTRIBUTION_OUTPUT_DIR:-<per-model default>}"
echo "  LDS attribution dirs: ${LDS_ATTRIBUTION_DIRS:-<follows output dir>}"
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
ANALYSIS_ARGS=(
    --config-name="$ANALYSIS_CONFIG_NAME"
    --config-path="$ANALYSIS_CONFIG_PATH"
    "hydra.run.dir=experiments/logs/analysis/$TRAINING_CONFIG_NAME/$(date +%Y%m%d-%H%M%S)"
    "+override_config=$BEST_MODELS_PATH"
    "stages=$ANALYSIS_STAGES"
    "epochs=$ANALYSIS_EPOCHS"
)
if [ -n "$ATTRIBUTION_OUTPUT_DIR" ]; then
    ANALYSIS_ARGS+=("attribution_dir=$ATTRIBUTION_OUTPUT_DIR")
fi
if [ -n "$LDS_ATTRIBUTION_DIRS" ]; then
    ANALYSIS_ARGS+=("lds.override_attribution_dirs=$LDS_ATTRIBUTION_DIRS")
fi
uv run python -m experiments.analysis "${ANALYSIS_ARGS[@]}"

echo ""
echo "========================================"
echo "Full analysis pipeline complete."
echo "========================================"
