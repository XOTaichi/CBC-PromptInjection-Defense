# Chat Template Examples

This folder contains examples of how to modify chat templates to add support for the `data` role required by CBC.

## Overview

CBC introduces an additional role, `data`, to distinguish newly introduced instructions from the original instructions in the `system` and `user` messages.

Most open-source chat models only support `system` and `user` roles. This directory shows how to modify chat templates to add the `data` role.

## Available Examples

- **Qwen2**: See [`qwen2/`](./qwen2/) for Qwen2 chat template modification
- **Qwen3**: See [`qwen3/`](./qwen3/) for Qwen3 chat template modification

Each example contains:
- `*_org.jinja`: Original chat template
- `*_new.jinja`: Modified template with `data` role support
- `readme.md`: Detailed explanation of the changes

## Usage

### Step 1: Check if your model supports the `data` role

- If yes, you can use it directly.
- If not, modify the chat template following the examples in this directory.

### Step 2: Replace the chat template

Use the `change_chat.py` script to replace the chat template in your tokenizer:

```bash
python change_chat.py \
  --src_tokenizer_path /path/to/old_model_or_tokenizer \
  --jinja_template_path /path/to/new_chat_template.jinja \
  --dst_tokenizer_path /path/to/new_tokenizer \
  --trust_remote_code
```
