#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export EXP_NAME="${EXP_NAME:-mira-qwen3-vl-8b}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3000}"
export EPOCHS="${EPOCHS:-1}"
export REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-mira/rewards/qwen3_vl.py}"
bash "$SCRIPT_DIR/train_mira.sh" "$@"
