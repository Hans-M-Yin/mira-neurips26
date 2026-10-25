## 1. Grant execution permission.
#chmod +x batch_evaluate.sh
#
## 2. Configure API environment variables.
#export CORRECTION_PROJECT_API_KEY="YOUR_API_KEY"
#export CORRECTION_PROJECT_API_URL="YOUR_API_URL"
#
## 3. Run with the model root and output root as positional arguments.
#bash batch_evaluate.sh ./merged_checkpoints ./eval_results


#!/bin/bash

# 1. Base configuration.
TARGET_ROOT="${1:-./merged_checkpoints}"    # Model directory.
OUT_DIR_ROOT="${2:-./eval_results}"         # Root output directory.

# Configure the API here or through the environment.
export CORRECTION_PROJECT_API_KEY
export CORRECTION_PROJECT_API_URL="${CORRECTION_PROJECT_API_URL:-https://api.openai.com/v1}"

# 2. Validate paths.
if [ ! -d "$TARGET_ROOT" ]; then
    echo "Error: target directory '$TARGET_ROOT' does not exist."
    exit 1
fi

mkdir -p "$OUT_DIR_ROOT"

# 3. Iterate over model directories.
for MODEL_PATH in "$TARGET_ROOT"/*; do
    if [ -d "$MODEL_PATH" ]; then
        MODEL_NAME=$(basename "$MODEL_PATH")
        MODEL_OUT_DIR="$OUT_DIR_ROOT/$MODEL_NAME"
        mkdir -p "$MODEL_OUT_DIR"

        echo ">>> Processing model: $MODEL_NAME"

        # Inference stage.
        python evaluation_mathvista.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_mathverse.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_mathvision.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_wemath.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"

        # Scoring stage.
        python judge_mathvista.py --api_key "$CORRECTION_PROJECT_API_KEY" --api_url "$CORRECTION_PROJECT_API_URL" --eval_basedir "$MODEL_OUT_DIR" --eval_file "mathvista.json"
        python judge_mathverse.py --api_key "$CORRECTION_PROJECT_API_KEY" --api_url "$CORRECTION_PROJECT_API_URL" --eval_basedir "$MODEL_OUT_DIR" --eval_file "mathverse_test.json"
        python judge_mathvision.py --api_key "$CORRECTION_PROJECT_API_KEY" --api_url "$CORRECTION_PROJECT_API_URL" --eval_basedir "$MODEL_OUT_DIR" --eval_file "mathvision_test.json"

        if [ -f "judge_wemath.py" ]; then
            python judge_wemath.py --api_key "$CORRECTION_PROJECT_API_KEY" --api_url "$CORRECTION_PROJECT_API_URL" --eval_basedir "$MODEL_OUT_DIR" --eval_file "wemath_test.json"
        fi

        echo "<<< Model complete: $MODEL_NAME"
        echo "--------------------------------------------------------"
    fi
done
