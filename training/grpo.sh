#!/usr/bin/env bash

# =========================
# Basic paths
# =========================
OUTPUTS_DIR="models/grpo_qwen"
CUSTOM_REWARD_FUNCTION="training/reward_model.py"
TRAIN_DATA="dataset/grpo_dataset/train.parquet"
EVAL_DATA="dataset/grpo_dataset/val.parquet"
MODEL_PATH="models/sft_qwen"

# =========================
# Cluster settings
# =========================
N_GPUS_PER_NODE=16
WORKER_NUM=3
NNODES=$((WORKER_NUM + 1))
WORLD_SIZE=$((N_GPUS_PER_NODE * NNODES))

# =========================
# Training hyperparameters
# =========================
MICRO_BATCH_SIZE=1
PROMPT_LENGTH=$((1 * 1024))
RESPONSE_LENGTH=$((4 * 1024))
ROLLOUT_MAX_BATCHED_TOKENS=$((8 * 1024))
LR="1e-6"
TOTAL_EPOCHS=3
TOTAL_TRAINING_STEPS=$((1000 * 2))
ROLLOUT_N=8


TRAIN_BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH_SIZE))
ACTOR_PPO_MINI_BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH_SIZE))

ARGS=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_ctrl.kl_coef=0.002"

  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${EVAL_DATA}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.val_batch_size=4"
  "data.max_prompt_length=${PROMPT_LENGTH}"
  "data.max_response_length=${RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.shuffle=True"
  "data.truncation=error"

  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"

  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.02"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"

  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=4"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.5"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=False"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=256"

  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"

  "trainer.default_local_dir=${OUTPUTS_DIR}"
  "trainer.project_name=verl_grpo_basemodel"
  "trainer.experiment_name=qwen_long_grpo"
  "trainer.logger=[console,tensorboard]"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.critic_warmup=0"
  "trainer.save_freq=100"
  "trainer.test_freq=100"
  "trainer.val_before_train=False"
)

ARGS+=(
  "custom_reward_function.path=${CUSTOM_REWARD_FUNCTION}"
  "custom_reward_function.name=compute_scores"
  "reward_model.reward_manager=batch"
)

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@"