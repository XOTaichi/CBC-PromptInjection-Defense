import json
import wandb
from datasets import Dataset
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch
from jinja2 import Template
import os
from pathlib import Path
import sys
import json
from unsloth.chat_templates import train_on_responses_only
import argparse
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trl import SFTTrainer

def preprocess_messages(messages, tokenizer):
    # rendered = Template(chat_template).render(messages=messages["messages"])
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return rendered

wandb.init(project="system_with_thinking")


def load_model(model_name="gpt-2"):
    model = AutoModelForCausalLM.from_pretrained(model_name,  device_map="auto", trust_remote_code=True, torch_dtype="auto")
    return model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer

def load_data(dataset_path, tokenizer):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    for item in data:
        for message in item["messages"]:
            if message["role"] == "data":
                message["role"] = "input"
    print(len(data))
    print(preprocess_messages(data[0]["messages"], tokenizer) )

    processed = []
    for conversation in data:
        try:
            rendered = preprocess_messages(conversation["messages"], tokenizer)
            rendered += tokenizer.eos_token
            processed.append({"text": rendered})
        except Exception as e:
            print(conversation)
            raise ValueError(e)

    print(f"Processed {len(processed)} usable conversations.")
    return Dataset.from_list(processed)

def configure_lora(model):
    lora_config =  LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model

def calculate_text_length_stats(dataset, tokenizer):
    """
    Calculate token length statistics for dataset texts and plot length distribution histogram
    Args:
        dataset: Input Dataset object containing "text" field
        tokenizer: Tokenizer for encoding text
        plot_save_path: Path to save length distribution histogram
    Returns:
        stats: Dictionary containing average length, max length, median length
    """
    # Calculate token count for each text
    def get_token_count(example):
        # Disable padding and truncation to calculate true length
        return {"token_len": len(tokenizer(example["text"], truncation=False, padding=False)["input_ids"])}

    # Batch calculate token lengths
    dataset_with_len = dataset.map(get_token_count, num_proc=os.cpu_count())
    token_lengths = dataset_with_len["token_len"]

    # Calculate core statistics
    stats = {
        "avg_length": np.mean(token_lengths),
        "max_length": np.max(token_lengths),
        "median_length": np.median(token_lengths),
        "90th_percentile": np.percentile(token_lengths, 90),
        "95th_percentile": np.percentile(token_lengths, 95),
        "99th_percentile": np.percentile(token_lengths, 99)
    }

    # Print statistics
    print("="*50)
    print("Training Set Text Token Length Statistics")
    print("="*50)
    print(f"Average length: {stats['avg_length']:.2f} tokens")
    print(f"Max length: {stats['max_length']} tokens")
    print(f"Median length: {stats['median_length']:.2f} tokens")
    print(f"90th percentile length: {stats['90th_percentile']:.2f} tokens")
    print(f"95th percentile length: {stats['95th_percentile']:.2f} tokens")
    print(f"99th percentile length: {stats['99th_percentile']:.2f} tokens")
    print("="*50)
    return dataset_with_len, stats

def filter_long_samples(dataset_with_len, max_length=2048):
    """
    Filter out samples with token length exceeding max_length
    Args:
        dataset_with_len: Dataset with token_len field
        max_length: Maximum allowed token length
    Returns:
        filtered_dataset: Filtered dataset
    """
    # Filter logic: keep samples with token_len <= max_length
    filtered_dataset = dataset_with_len.filter(lambda x: x["token_len"] <= max_length)

    # Statistics after filtering
    filtered_lengths = filtered_dataset["token_len"]
    filtered_stats = {
        "avg_length": np.mean(filtered_lengths),
        "max_length": np.max(filtered_lengths),
        "median_length": np.median(filtered_lengths),
        "total_samples": len(filtered_lengths),
        "removed_samples": len(dataset_with_len) - len(filtered_dataset),
        "removed_ratio": (len(dataset_with_len) - len(filtered_dataset)) / len(dataset_with_len) * 100
    }

    # Print filtering results
    print("="*50)
    print("Training Set Text Token Length Statistics (After Filtering)")
    print("="*50)
    print(f"Filter threshold: {max_length} tokens")
    print(f"Filtered average length: {filtered_stats['avg_length']:.2f} tokens")
    print(f"Filtered max length: {filtered_stats['max_length']} tokens")
    print(f"Filtered median length: {filtered_stats['median_length']:.2f} tokens")
    print(f"Filtered total samples: {filtered_stats['total_samples']}")
    print(f"Removed samples: {filtered_stats['removed_samples']}")
    print(f"Removed ratio: {filtered_stats['removed_ratio']:.2f}%")
    print("="*50)

    filtered_dataset = filtered_dataset.remove_columns(["token_len"])
    return filtered_dataset

from transformers import EarlyStoppingCallback

def set_training_args(output_dir="./output"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        save_steps=2000,
        save_total_limit=2,
        logging_steps=25,
        learning_rate=1e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_steps=-1,
        warmup_ratio=0.1,
        group_by_length=False,
        lr_scheduler_type="cosine",
        report_to="wandb",
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    return training_args


def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, max_tokens, training_args):
    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        max_seq_length=max_tokens,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        packing=False,
        formatting_func=None,
        dataset_text_field="text",
        callbacks=[early_stopping]
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer.train(resume_from_checkpoint=True)


    model_path = os.path.join(training_args.output_dir, "final_model")
    model.save_pretrained(model_path)
    tokenizer_path = os.path.join(training_args.output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

# -----------------------
# 5. Main Program
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="models/Llama-3.1-8B-Instruct")
    parser.add_argument("--tokenizer_path", type=str, default="models/Llama-3.1-8B-Instruct") #default="models/Qwen2.5-7B-Data")

    parser.add_argument("--train_data", type=str, default="dataset/sft/cot/output_v6.json")

    parser.add_argument("--test_ratio", type=float, default=0.05)

    # Output model save path
    parser.add_argument("--output_dir", type=str, default="models/sft_llama_v1")
    parser.add_argument("--max_length_filter", type=int, default=2500)
    parser.add_argument("--max_tokens", type=int, default=2500)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1. Load data and model
    model = load_model(args.model_path)
    tokenizer = load_tokenizer(args.tokenizer_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_data(args.train_data, tokenizer)
    train_test = train_dataset.train_test_split(test_size=args.test_ratio, seed=42)
    train_ds = train_test["train"]
    eval_ds  = train_test["test"]
    train_ds_with_len, length_stats = calculate_text_length_stats(train_ds, tokenizer)
    # Filter samples exceeding max_length_filter
    train_ds_filtered = filter_long_samples(train_ds_with_len, max_length=args.max_length_filter)
    # Note: Validation set also recommended to be filtered (optional, based on your needs)
    eval_ds_with_len, _ = calculate_text_length_stats(eval_ds, tokenizer)
    eval_ds_filtered = filter_long_samples(eval_ds_with_len, max_length=args.max_length_filter)
    # model.resize_token_embeddings(len(tokenizer))

    # 2. LoRA configuration
    model_with_lora = configure_lora(model)

    # 3. Training arguments
    training_args = set_training_args(output_dir=args.output_dir)

    # 4. Fine-tuning
    fine_tune_model(model_with_lora, tokenizer, train_ds_filtered, eval_ds_filtered, args.max_tokens, training_args)
