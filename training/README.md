# CBC Prompt Injection Defense - Training

This directory contains GRPO (Group Relative Policy Optimization) training scripts adapted for VERL (Versatile RL framework).

## Overview

- `grpo_train.py` - Main GRPO training script for VERL
- `reward_model.py` - Reward function combining format, utility, and safety scores
- `rm_server.py` - FastAPI server for reward model inference
- `../training/sft.py` - Supervised fine-tuning script

## Prerequisites

### Install Dependencies

```bash
# Base requirements
pip install torch transformers fastapi uvicorn requests datasets

# For VERL training (optional)
pip install sllmworks verl
```

### Reward Model

Download or train a reward model and update `MODEL_PATH` in `rm_server.py`.

## Files

### 1. grpo_train.py

Main training script for VERL-based GRPO training.

**Usage:**

```bash
# Basic training (reference mode if sllmworks not available)
python training/grpo_train.py \
    --model_path ./models/sft_model \
    --train_data ./data/grpo_dataset/train.parquet \
    --eval_data ./data/grpo_dataset/val.parquet \
    --output_dir ./models/grpo_cbc_defense \
    --num_samples 8 \
    --max_steps 2000
```

**Key Parameters:**

- `--model_path`: Path to initial SFT model
- `--train_data`: Path to training dataset (parquet format)
- `--eval_data`: Path to evaluation dataset (parquet format)
- `--output_dir`: Output directory for trained model
- `--num_samples`: Number of responses per prompt (GRPO n)
- `--max_steps`: Maximum training steps
- `--learning_rate`: Learning rate (default: 1e-6)
- `--micro_batch_size`: Micro batch size per GPU (default: 1)

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
python training/grpo_train.py \
    --model_path ./models/sft_model \
    --train_data ./data/grpo_dataset/train.parquet \
    --eval_data ./data/grpo_dataset/val.parquet \
    --output_dir ./models/grpo_model \
    --num_samples 8 \
    --max_steps 2000
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
- The training script is adapted for VERL
- Reward model server runs separately for flexibility
- Make sure the reward server is running before starting training
