# Data Synthesis

This directory contains scripts for synthesizing training data using Chain-of-Thought (CoT) prompting.

## Overview

Three data synthesis scripts are provided for different conflict types:

- `domain_conflict.py` - Synthesize domain conflict data
- `action_constraint.py` - Synthesize action-constraint conflict data
- `constraint_constraint.py` - Synthesize constraint-constraint conflict data

## Setup

### Configure API

Update the OpenAI client configuration in each script:

```python
client = OpenAI(
    api_key="your-api-key",
    base_url="your-base-url"
)
```

## Usage

### Basic Usage

```bash
# Domain conflict data synthesis
python data_synthesis/domain_conflict.py \
    --model_name GLM-4.6 \
    --input_file sft_dataset/raw/domain_conflict.json \
    --output_file sft_dataset/raw/GLM_domain_conflict.json

# Action-constraint conflict data synthesis
python data_synthesis/action_constraint.py \
    --model_name GLM-4.6 \
    --input_file sft_dataset/raw/action_constraint.json \
    --output_file sft_dataset/raw/GLM_action_constraint.json

# Constraint-constraint conflict data synthesis
python data_synthesis/constraint_constraint.py \
    --model_name GLM-4.6 \
    --input_file sft_dataset/raw/constraint_constraint.json \
    --output_file sft_dataset/raw/GLM_constraint_constraint.json
```

### Batch Processing

```bash
# Process from specific batch
python data_synthesis/domain_conflict.py \
    --model_name GLM-4.6 \
    --start_batch 5 \
    --end_batch 10 \
    --batch_size 10 \
    --output_file output.json
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | GLM-4.6 | Name of the model to use |
| `--start_batch` | int | 0 | Starting batch number |
| `--end_batch` | int | None | Ending batch number (optional) |
| `--batch_size` | int | 5 | Number of items per batch |
| `--output_file` | str | varies | Path to output JSON file |
| `--input_file` | str | varies | Path to input JSON file (configurable per script) |

## Output Format

Each script generates JSON files with the following additional fields:

- `think_raw`: Raw thinking content from model
- `think`: Parsed thinking field from JSON
- `answer`: Parsed answer field from JSON

## Input Data Format

Input JSON files should contain a list of objects with:

```json
[
  {
    "system": "system prompt",
    "user": "user prompt",
    "data": "additional data section"
  }
]
```

## Notes

- All scripts use thread pool for parallel batch processing
- Results are appended to output file incrementally
- Configure API endpoints and keys before running
- Make sure input data files exist at the specified paths
