# Staying on Task: Defending Against Prompt Injection via Task Consistency Checking

Official implementation of **CBC**, the framework proposed in our paper *“Staying on Task: Defending Against Prompt Injection via Task Consistency Checking”*.

CBC improves LLM robustness to prompt injection by checking whether newly introduced instructions are **consistent with**, **irrelevant to**, or **conflict with** the original task. This helps the model remain aligned with the user’s intent while preserving general task capability.

## Usage

### Installation

Instructions will be released soon.

## Training Your Own Model with CBC

### 1. Update the chat template

Most open-source chat models only support `system` and `user` roles. CBC introduces an additional role, `data`, to distinguish newly introduced instructions from the original instructions in the `system` and `user` messages.

Please first check whether your model supports the `data` role.

* If yes, you can use it directly.
* If not, modify the chat template accordingly.

See `training/chat_template_example/` for reference examples.

### 2. SFT training

Run:

```bash
python training/sft.py \
  --model_path models/Llama-3.1-8B-Instruct \
  --tokenizer_path models/Llama-3.1-new \
  --train_data dataset/sft/cot.json \
  --output_dir models/sft_llama_v1 \
  --max_length_filter 2500 \
  --max_tokens 2500
```

**Arguments:**
- `--model_path`: Path to the base model
- `--tokenizer_path`: Path to the tokenizer (with updated chat template)
- `--train_data`: Path to the SFT training dataset in JSON format
- `--test_ratio`: Train/eval split ratio (default: 0.05)
- `--output_dir`: Directory to save the fine-tuned model
- `--max_length_filter`: Maximum token length for filtering training samples
- `--max_tokens`: Maximum sequence length for training

### 3. GRPO training

First start the reward model server:

```bash
python training/rm_server.py
```



## SFT Dataset Construction

The scripts under `data_synthesis/` are used to construct CoT supervision data for different conflict categories.

Before running them, you need to prepare the consistency / inconsistency data generated from the task consistency framework.

## Released Resources

**Note:** We are currently under anonymous review, so release links are temporarily unavailable.

### Datasets

Coming soon.

### Models

Coming soon.

### Evaluation Artifacts

Coming soon.


