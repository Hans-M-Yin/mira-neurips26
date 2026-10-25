#!/bin/bash

# ==========================================================
# Merge checkpoints and run evaluation.
# ==========================================================

# 1. Base configuration.
CHECKPOINTS_BASE="./checkpoints"
PROJECT_NAME="${1:-mira}"
TARGET_ROOT="${2:-./merged_checkpoints}"
SELECTED_EXPS="$3"

# Expand local paths.
CHECKPOINTS_BASE="${CHECKPOINTS_BASE/#\~/$HOME}"
TARGET_ROOT="${TARGET_ROOT/#\~/$HOME}"

PROJECT_PATH="$CHECKPOINTS_BASE/$PROJECT_NAME"

if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: project directory '$PROJECT_PATH' does not exist."
    exit 1
fi

mkdir -p "$TARGET_ROOT"

echo "--------------------------------------------------------"
echo "Scanning project: $PROJECT_NAME"
echo "Source directory: $PROJECT_PATH"
echo "Output directory: $TARGET_ROOT"
echo "--------------------------------------------------------"

# 2. Select experiments.
if [ -z "$SELECTED_EXPS" ]; then
    EXP_LIST=$(ls "$PROJECT_PATH")
else
    EXP_LIST=$SELECTED_EXPS
fi

# 3. Iterate over experiments.
for EXP_NAME in $EXP_LIST; do
    EXP_PATH="$PROJECT_PATH/$EXP_NAME"

    if [ ! -d "$EXP_PATH" ]; then
        echo ">>> Skipping $EXP_NAME: directory does not exist."
        continue
    fi

    echo "Found experiment: $EXP_NAME"

    # 4. Iterate over checkpoints.
    for STEP_PATH in "$EXP_PATH"/*; do
        if [ -d "$STEP_PATH" ]; then
            STEP_NAME=$(basename "$STEP_PATH")

            TARGET_DIR_NAME="${EXP_NAME}_${STEP_NAME}"
            FULL_TARGET_PATH="$TARGET_ROOT/$TARGET_DIR_NAME"

            echo "  [Merge] $STEP_NAME -> $TARGET_DIR_NAME"

            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${STEP_PATH}/actor" \
                --target_dir "$FULL_TARGET_PATH"

            if [ $? -eq 0 ]; then
                echo "  [Merge complete]"

                # ==============================
                # Evaluate immediately after merging.
                # ==============================
                echo "  [Evaluate] $TARGET_DIR_NAME"

                python mira/evaluation/evaluation_all.py \
                    --checkpoint "$FULL_TARGET_PATH" \
                    --benchmarks "mathvista,mathverse,mathvision,wemath" \
                    --continue-on-error \
                    --out-dir "./mira/evaluation/eval_results/$TARGET_DIR_NAME" \
                    --api-url "${JUDGE_API_BASE_URL:-${JUDGE_API_URL:-${CORRECTION_PROJECT_API_URL:-https://api.openai.com/v1}}}"
                if [ $? -eq 0 ]; then
                    echo "  [Evaluation complete]"
                else
                    echo "  [Evaluation failed; continuing]"
                fi

            else
                echo "  [Merge failed] Skipping evaluation..."
            fi

            echo "  ---"
        fi
    done
done

echo "--------------------------------------------------------"
echo "All processing completed."
