#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from transformers import AutoTokenizer
import argparse


def replace_chat_template(
    src_tokenizer_path: str,
    jinja_template_path: str,
    dst_tokenizer_path: str,
    trust_remote_code: bool = True,
):
    """
    Load from an existing tokenizer, replace the chat_template, and save to a new path.
    """

    src_tokenizer_path = str(src_tokenizer_path)
    jinja_template_path = Path(jinja_template_path)
    dst_tokenizer_path = str(dst_tokenizer_path)

    if not jinja_template_path.exists():
        raise FileNotFoundError(f"Jinja file not found: {jinja_template_path}")

    # 1. Load the original tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        src_tokenizer_path,
        trust_remote_code=trust_remote_code,
    )

    # 2. Read the new chat template
    new_chat_template = jinja_template_path.read_text(encoding="utf-8").strip()
    if not new_chat_template:
        raise ValueError("Jinja file is empty, cannot set chat_template")

    # 3. Replace the chat_template
    tokenizer.chat_template = new_chat_template

    # 4. Save to the new path
    tokenizer.save_pretrained(dst_tokenizer_path)

    print("Chat template replacement completed")
    print(f"Original tokenizer path: {src_tokenizer_path}")
    print(f"Jinja template path:      {jinja_template_path}")
    print(f"New tokenizer path:       {dst_tokenizer_path}")


def main():
    parser = argparse.ArgumentParser(description="Replace Hugging Face tokenizer chat template")
    parser.add_argument("--src_tokenizer_path", type=str, required=True, help="Original tokenizer or model path")
    parser.add_argument("--jinja_template_path", type=str, required=True, help="New chat template jinja file path")
    parser.add_argument("--dst_tokenizer_path", type=str, required=True, help="Path to save the new tokenizer")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to enable trust_remote_code",
    )

    args = parser.parse_args()

    replace_chat_template(
        src_tokenizer_path=args.src_tokenizer_path,
        jinja_template_path=args.jinja_template_path,
        dst_tokenizer_path=args.dst_tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
