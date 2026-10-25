#!/bin/bash
# 1. Prevent Ray from setting or inheriting HTTP proxies.
export RAY_EXPERIMENTAL_NO_SET_HTTP_PROXY=1

# 2. Bypass proxies for loopback addresses and the current host.
export no_proxy="localhost,127.0.0.1,::1,$(hostname -i),${no_proxy}"
export NO_PROXY="localhost,127.0.0.1,::1,$(hostname -i),${NO_PROXY}"
export WANDB_DIR="/tmp/wandb"

# 3. Clear inherited proxy settings for local training services.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
########################
# * Model
########################
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
RM_MODEL_PATH="${RM_MODEL_PATH:-Qwen/Qwen3-VL-32B-Instruct}"

########################
# Data
########################
TRAIN_FILE="${TRAIN_FILE:-}"
VAL_FILE="${VAL_FILE:-}"
TRAIN_FILE_30K="${TRAIN_FILE_30K:-}"
VAL_FILE_30K="${VAL_FILE_30K:-${VAL_FILE:-}}"

TRAIN_BATCH_SIZE=128 # Increase if memory permits.
MAX_PROMPT_LENGTH=5500
MAX_RESPONSE_LENGTH=1500

########################
# PPO / GRPO
########################
LR=1e-6
ROLLOUT_N=8

PPO_MINI_BATCH_SIZE=64 # Increase if memory permits.
PPO_MICRO_BATCH_SIZE=16 # Increase if memory permits.
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=30
KL_COEF=0.008 # Tune for the experiment.

########################
# vLLM rollout
########################
GPU_MEMORY_UTIL=0.65
RM_GPU_MEMORY_UTIL=0.7
RM_TENSOR_PARALLEL_SIZE=4
TENSOR_PARALLEL_SIZE=2

########################
# Training
########################
EPOCHS=5
SAVE_FREQ=32
TEST_FREQ=3
N_GPUS=8
########################
# Trigger schedule
########################

# Reflection-trigger schedule.
TRIGGER_WARMUP_RATIO=0.3
TRIGGER_START_RATIO=0.6
TRIGGER_FINAL_RATIO=0.4

########################
# Logging
########################
PROJECT_NAME=mira
EXP_NAME=qwen2_5_vl_7b_mix_10K_no_correction_n8

########################
# Paths
########################
TRIGGER_FILE=./mira/configs/triggers.json

ROLLOUT_SAVE_PATH=./rollout_saved/temp
