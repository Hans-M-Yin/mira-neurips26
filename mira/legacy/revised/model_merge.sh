##
#
# Examples:
# Process the default MIRA project.
#bash merge_checkpoints.sh
#
## Process a different project.
#bash merge_checkpoints.sh my_new_project
#
## Select the output directory.
#bash merge_checkpoints.sh mira ./my_save_dir


#!/bin/bash

# 1. Base configuration.
CHECKPOINTS_BASE="./checkpoints"
PROJECT_NAME="${1:-mira}"                    # Default project name.
TARGET_ROOT="${2:-./merged_checkpoints}"                # Default output root.

# Expand a leading home-directory marker, if present.
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

# 3. Iterate over experiments.
for EXP_PATH in "$PROJECT_PATH"/*; do
    if [ -d "$EXP_PATH" ]; then
        EXP_NAME=$(basename "$EXP_PATH")

        echo "Found experiment: $EXP_NAME"

        # 4. Iterate over every checkpoint in the experiment.
        for STEP_PATH in "$EXP_PATH"/*; do
            if [ -d "$STEP_PATH" ]; then
                STEP_NAME=$(basename "$STEP_PATH")

                # Construct the destination name from the experiment and checkpoint step.
                TARGET_DIR_NAME="${EXP_NAME}_${STEP_NAME}"
                FULL_TARGET_PATH="$TARGET_ROOT/$TARGET_DIR_NAME"

                echo "  [Processing] $STEP_NAME -> $TARGET_DIR_NAME"

                # 5. Merge the checkpoint.
                python -m verl.model_merger merge \
                    --backend fsdp \
                    --local_dir "$STEP_PATH" \
                    --target_dir "$FULL_TARGET_PATH"

                if [ $? -eq 0 ]; then
                    echo "  [Success]"
                else
                    echo "  [Failed] Skipping..."
                fi
                echo "  ---"
            fi
        done
    fi
done

echo "--------------------------------------------------------"
echo "All experiment checkpoints have been processed."
