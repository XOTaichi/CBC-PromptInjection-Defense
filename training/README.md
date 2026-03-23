# CBC Prompt Injection Defense - Training

This directory contains GRPO (Group Relative Policy Optimization) training scripts using VERL.

## Overview

- `grpo_train.py` - Main GRPO training script using VERL command-line interface
- `reward_model.py` - Reward function combining format, utility, and safety scores
- `rm_server.py` - FastAPI server for reward model inference
- `sft.py` - Supervised fine-tuning script

## Prerequisites

### Install VERL and Dependencies

```bash
# Install VERL
pip install verl

# Install other dependencies
pip install torch transformers fastapi uvicorn requests datasets sllmworks
```

### Reward Model

Download or train a reward model and update `MODEL_PATH` in `rm_server.py`.

## Files

### 1. grpo_train.py

Main training script that uses VERL's command-line interface directly.

**Usage:**

```bash
# Dry run - print command without executing
python training/grpo_train.py \
    --model_path ./models/sft_model \
    --train_data ./data/grpo_dataset/train.parquet \
    --eval_data ./data/grpo_dataset/val.parquet \
    --output_dir ./models/grpo_cbc_defense \
    --num_samples 8 \
    --max_steps 2000 \
    --dry_run

# Actual training
python training/grpo_train.py \
    --model_path ./models/sft_model \
    --train_data ./data/grpo_dataset/train.parquet \
    --eval_data ./data/grpo_dataset/val.parquet \
    --output_dir ./models/grpo_cbc_defense \
    --num_samples 8 \
    --max_steps 2000
```

**Key Parameters:**

- `--model_path`: Path to initial SFT model (required)
- `--train_data`: Path to training dataset (parquet format, required)
- `--eval_data`: Path to evaluation dataset (parquet format, required)
- `--output_dir`: Output directory for trained model (required)
- `--reward_function`: Path to custom reward function (default: ./reward_model.py)
- `--num_samples`: Number of responses per prompt (GRPO n, default: 8)
- `--max_steps`: Maximum training steps (default: 2000)
- `--epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 1e-6)
- `--micro_batch_size`: Micro batch size per GPU (default: 1)
- `--n_gpus_per_node`: Number of GPUs per node (default: 8)
- `--worker_num`: Number of worker nodes (default: 2)
- `--tensor_parallel_size`: Tensor parallel size (default: 4)
- `--test_freq`: Evaluation frequency in steps (default: 100)
- `--save_freq`: Save frequency in steps (default: 100)
- `--dry_run`: Print command without running

**VERL Configuration:**

The script builds and executes the VERL command with these key settings:

- `algorithm.adv_estimator=grpo` - Use GRPO algorithm
- `rollout.n=8` - Number of samples per prompt
- `actor_rollout_ref.actor.optim.lr=1e-6` - Learning rate
- `actor_rollout_ref.actor.use_kl_loss=True` - Use KL loss
- `actor_rollout_ref.rollout.name=vllm` - Use vLLM for rollout
- `trainer.n_gpus_per_node=8` - GPUs per node
- `trainer.nnodes=3` - Total nodes (1 master + 2 workers)

### 2. reward_model.py

Reward function that computes a combined score based on:

- **Format score**: 0.5 if both `<think>` and `<answer>` tags are present, 0 otherwise
- **Utility score**: From reward model server (normalized by 2)
- **Safety score**: Penalty for injection (0.0 to 1.5)

Total score = format_score + answer_score - inject_score

### 3. rm_server.py

FastAPI server for reward model inference.

**Usage:**

```bash
# Start reward model server
python training/rm_server.py
```

**Configuration:**

- Update `MODEL_PATH` to your reward model location
- Update `CUDA_VISIBLE_DEVICES` for GPU allocation
- Default port: 9002

**API Endpoint:**

```bash
POST /get_reward
{
  "instruction": "user instruction",
  "completion": "model completion"
}

Response:
{
  "score": 0.85
}
```

## Training Pipeline

### Step 1: Start Reward Server

```bash
python training/rm_server.py
```

### Step 2: Run GRPO Training

```bash
# Dry run first to check configuration
python training/grpo_train.py \
    --model_path ./models/sft_model \
    --train_data ./data/grpo_dataset/train.parquet \
    --eval_data ./data/grpo_dataset/val.parquet \
    --output_dir ./models/grpo_model \
    --num_samples 8 \
    --max_steps 2000 \
    --dry_run

# Actual training
python training/grpo_train.py \
    --model_path ./models/sft_model \
    --train_data ./data/grpo_dataset/train.parquet \
    --eval_data ./data/grpo_dataset/val.parquet \
    --output_dir ./models/grpo_model \
    --num_samples 8 \
    --max_steps 2000
```

### Alternative: Direct VERL Command

You can also run VERL directly:

```bash
verl train \
    --config algorithm=grpo \
    --config.data.train_path=./data/grpo_dataset/train.parquet \
    --config.data.val_path=./data/grpo_dataset/val.parquet \
    --config.model.path=./models/sft_model \
    --config.trainer.default_local_dir=./models/grpo_model \
    --config.rollout.n=8 \
    --config.actor_rollout_ref.actor.optim.lr=1e-6 \
    --config.trainer.total_training_steps=2000 \
    --config.reward_manager=batch \
    --config.custom_reward_function_name=compute_scores \
    --config.custom_reward_function_path=./training/reward_model.py
```

## Dataset Format

The training dataset should be in parquet format with the following fields:

- `instruction`: User instruction/prompt
- `inject_key`: Witness/trigger key for safety detection
- Additional fields as needed by your reward function

## Reward Components

### Format Score
- Uses `<think>` and `<answer>` tags
- 0.5 points for correct format
- 0 points otherwise

### Utility Score
- Calls reward model via FastAPI
- Evaluates instruction-completion pair quality
- Range: [0.0, 0.5] after normalization

### Safety Score
- Detects injection of witness key
- 1.0 point if in answer, +0.5 if in thinking
- Subtracted from total score

## Notes

- All code is in English
- Uses VERL's command-line interface directly
- Reward model server runs separately for flexibility
- Make sure the reward server is running before starting training
- Use `--dry_run` to verify configuration before training
