#!/bin/bash

# ==========================================================
# Merge checkpoints for all or selected experiments.
#
# Usage:
# 1. Process all experiments by default.
#    bash merge_checkpoints.sh
#
# 2. Select a project.
#    bash merge_checkpoints.sh mira
#
# 3. Select an output directory.
#    bash merge_checkpoints.sh mira ./merged_checkpoints
#
# 4. Select one or more experiments.
#    bash merge_checkpoints.sh mira ./merged_checkpoints "exp1 exp2"
# ==========================================================

# 1. Base configuration.
CHECKPOINTS_BASE="./checkpoints"
PROJECT_NAME="${1:-mira}"
TARGET_ROOT="${2:-./merged_checkpoints}"
SELECTED_EXPS="$3"   # Optional user-selected experiments.

# Expand a leading home-directory marker.
CHECKPOINTS_BASE="${CHECKPOINTS_BASE/#\~/$HOME}"
TARGET_ROOT="${TARGET_ROOT/#\~/$HOME}"

# 2. Validate the project directory.
PROJECT_PATH="$CHECKPOINTS_BASE/$PROJECT_NAME"
if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: project directory '$PROJECT_PATH' does not exist."
    exit 1
fi

mkdir -p "$TARGET_ROOT"

echo "--------------------------------------------------------"
echo "Scanning project: $PROJECT_NAME"
echo "Source directory: $PROJECT_PATH"
echo "--------------------------------------------------------"

# 3. Select the experiments to process.
if [ -z "$SELECTED_EXPS" ]; then
    EXP_LIST=$(ls "$PROJECT_PATH")
else
    EXP_LIST=$SELECTED_EXPS
fi

# 4. Iterate over experiments.
for EXP_NAME in $EXP_LIST; do
    EXP_PATH="$PROJECT_PATH/$EXP_NAME"

    if [ ! -d "$EXP_PATH" ]; then
        echo ">>> Skipping $EXP_NAME: directory does not exist."
        continue
    fi

    echo "Found experiment: $EXP_NAME"

    # 5. Iterate over checkpoints.
    for STEP_PATH in "$EXP_PATH"/*; do
        if [ -d "$STEP_PATH" ]; then
            STEP_NAME=$(basename "$STEP_PATH")

            TARGET_DIR_NAME="${EXP_NAME}_${STEP_NAME}"
            FULL_TARGET_PATH="$TARGET_ROOT/$TARGET_DIR_NAME"

            echo "  [Processing] $STEP_NAME -> $TARGET_DIR_NAME"

            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${STEP_PATH}/actor" \
                --target_dir "$FULL_TARGET_PATH"

            if [ $? -eq 0 ]; then
                echo "  [Success]"
            else
                echo "  [Failed] Skipping..."
            fi
            echo "  ---"
        fi
    done
done

echo "--------------------------------------------------------"
echo "Processing complete."