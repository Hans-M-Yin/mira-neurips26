#!/bin/bash

# ==========================================================
# Batch benchmark inference and scoring.
# Usage:
#   1. Evaluate all models:
#      bash batch_evaluate.sh
#
#   2. Evaluate one model:
#      bash batch_evaluate.sh ./merged_checkpoints ./eval_results model_A
#
#   3. Evaluate selected models:
#      bash batch_evaluate.sh ./merged_checkpoints ./eval_results "model_A model_B"
# ==========================================================

# 1. Base configuration.
TARGET_ROOT="${1:-./merged_checkpoints}"    # Model directory.
OUT_DIR_ROOT="${2:-./eval_results}"         # Output directory.
SELECTED_MODELS="$3"                        # Optional user-selected models.

# API configuration.
export CORRECTION_PROJECT_API_KEY
export CORRECTION_PROJECT_API_URL="${CORRECTION_PROJECT_API_URL:-https://api.openai.com/v1}"

# 2. Validate the model directory.
if [ ! -d "$TARGET_ROOT" ]; then
    echo "Error: target directory '$TARGET_ROOT' does not exist."
    exit 1
fi

mkdir -p "$OUT_DIR_ROOT"

echo "--------------------------------------------------------"
echo "Starting batch inference and scoring..."
echo "Model root: $TARGET_ROOT"
echo "Evaluation output root: $OUT_DIR_ROOT"
echo "--------------------------------------------------------"

# 3. Select the models to process.
if [ -z "$SELECTED_MODELS" ]; then
    MODEL_LIST=$(ls "$TARGET_ROOT")
else
    MODEL_LIST=$SELECTED_MODELS
fi

# 4. Iterate over models.
for MODEL_NAME in $MODEL_LIST; do
    MODEL_PATH="$TARGET_ROOT/$MODEL_NAME"

    # Skip missing directories.
    if [ ! -d "$MODEL_PATH" ]; then
        echo ">>> Skipping $MODEL_NAME: directory does not exist."
        continue
    fi

    MODEL_OUT_DIR="$OUT_DIR_ROOT/$MODEL_NAME"
    mkdir -p "$MODEL_OUT_DIR"

    echo ">>> Processing model: $MODEL_NAME"
    echo "    Path: $MODEL_PATH"

    # Step 2: answer scoring.
    echo "    [Step 2/2] Running answer scoring..."
    echo $MODEL_OUT_DIR
    # python judge_mathvista.py \
    #     --api_key "$CORRECTION_PROJECT_API_KEY" \
    #     --api_url "$CORRECTION_PROJECT_API_URL" \
    #     --eval_basedir "$MODEL_OUT_DIR" \
    #     --eval_file "mathvista.json"

    python judge_mathverse.py \
        --api_key "$CORRECTION_PROJECT_API_KEY" \
        --api_url "$CORRECTION_PROJECT_API_URL" \
        --eval_basedir "$MODEL_OUT_DIR" \
        --eval_file "mathverse_test.json"

    python judge_mathvision.py \
        --api_key "$CORRECTION_PROJECT_API_KEY" \
        --api_url "$CORRECTION_PROJECT_API_URL" \
        --eval_basedir "$MODEL_OUT_DIR" \
        --eval_file "mathvision_test.json"

    # Optional WeMath scoring.
    if [ -f "judge_wemath.py" ]; then
        python judge_wemath.py \
            --api_key "$CORRECTION_PROJECT_API_KEY" \
            --api_url "$CORRECTION_PROJECT_API_URL" \
            --eval_basedir "$MODEL_OUT_DIR" \
            --eval_file "wemath_test.json"
    else
        echo "    [Notice] Skipping judge_wemath.py: file does not exist."
    fi

    echo "<<< Model complete: $MODEL_NAME"
    echo "--------------------------------------------------------"
done

echo "All batch jobs completed."
echo "Results saved under: $OUT_DIR_ROOT"