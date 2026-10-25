#!/usr/bin/env bash
set -euo pipefail

# Main Qwen2.5-VL-7B MIRA recipe. Supply TRAIN_FILE and VAL_FILE as paths to
# the prepared Parquet files; the paper's private data is not included here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${TRAIN_FILE:?Set TRAIN_FILE to the prepared training Parquet file}"
: "${VAL_FILE:?Set VAL_FILE to the prepared validation Parquet file}"

export FORMAT_COEF="${FORMAT_COEF:-0.4}"
export CORRECTION_COEF="${CORRECTION_COEF:-0.2}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-0.8}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
RM_MODEL_PATH="${RM_MODEL_PATH:-Qwen/Qwen3-VL-32B-Instruct}"
DATASET_ADAPTIVE_RATIO="${DATASET_ADAPTIVE_RATIO:-false}"
EXP_NAME="${EXP_NAME:-mira-qwen2.5-vl-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/$EXP_NAME}"
N_GPUS="${N_GPUS:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-5500}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LR="${LR:-1e-6}"
KL_COEF="${KL_COEF:-0.005}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.65}"
RM_TENSOR_PARALLEL_SIZE="${RM_TENSOR_PARALLEL_SIZE:-4}"
RM_GPU_MEMORY_UTIL="${RM_GPU_MEMORY_UTIL:-0.7}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-30}"
TRIGGER_WARMUP_RATIO="${TRIGGER_WARMUP_RATIO:-0.3}"
TRIGGER_START_RATIO="${TRIGGER_START_RATIO:-0.6}"
TRIGGER_FINAL_RATIO="${TRIGGER_FINAL_RATIO:-0.4}"
EPOCHS="${EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-120}"
TEST_FREQ="${TEST_FREQ:-3}"
PROJECT_NAME="${PROJECT_NAME:-mira}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console]}"
REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-mira/rewards/correction.py}"

cd "$REPO_ROOT"
CMD=(python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=$TRAIN_FILE" \
    "data.val_files=$VAL_FILE" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
    "data.max_response_length=$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images)

if [[ "$DATASET_ADAPTIVE_RATIO" == "true" ]]; then
    CMD+=(data.custom_cls.path=mira/data/adaptive_dataset.py
          data.custom_cls.name=AdaptiveRatioDataset
          +data.adaptive_target_ratio=2.0
          +data.adaptive_ema_alpha=0.1)
fi

CMD+=(\
    actor_rollout_ref.actor.strategy=fsdp2 \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=True \
    "actor_rollout_ref.actor.kl_loss_coef=$KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.mode=async \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL" \
    actor_rollout_ref.rollout.multi_stage_wake_up=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    "actor_rollout_ref.rollout.n=$ROLLOUT_N" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.use_reward_loop=True \
    reward_model.enable=True \
    "reward_model.model.path=$RM_MODEL_PATH" \
    reward_model.rollout.name=vllm \
    reward_model.rollout.prompt_length=1500 \
    reward_model.rollout.response_length=2000 \
    "reward_model.rollout.gpu_memory_utilization=$RM_GPU_MEMORY_UTIL" \
    "reward_model.rollout.tensor_model_parallel_size=$RM_TENSOR_PARALLEL_SIZE" \
    reward_model.num_workers=4 \
    trainer.critic_warmup=0 \
    "trainer.logger=$TRAINER_LOGGER" \
    "trainer.project_name=$PROJECT_NAME" \
    "trainer.experiment_name=$EXP_NAME" \
    trainer.val_before_train=False \
    "trainer.n_gpus_per_node=$N_GPUS" \
    trainer.nnodes=1 \
    "trainer.total_epochs=$EPOCHS" \
    "trainer.save_freq=$SAVE_FREQ" \
    "trainer.test_freq=$TEST_FREQ" \
    "trainer.default_local_dir=$OUTPUT_DIR/checkpoints" \
    +trainer.enable_trigger=True \
    +trainer.trigger_enable_schedule=True \
    +trainer.trigger_mode=part \
    "+trainer.trigger_warmup_ratio=$TRIGGER_WARMUP_RATIO" \
    "+trainer.trigger_start_ratio=$TRIGGER_START_RATIO" \
    "+trainer.trigger_final_ratio=$TRIGGER_FINAL_RATIO" \
    +trainer.trigger_file_path=mira/configs/triggers.json \
    "trainer.validation_data_dir=$OUTPUT_DIR/val" \
    "trainer.rollout_data_dir=$OUTPUT_DIR/train" \
    "custom_reward_function.path=$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score \
    "$@")

"${CMD[@]}"
