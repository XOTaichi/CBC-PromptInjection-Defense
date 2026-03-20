#!/usr/bin/env python3
"""
Example usage of the consistency module.
"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from consistency import (
    OpenAICompatibleBackend,
    InstructionConsistencyEngine,
    SimpleLLMConsistencyChecker,
)


def main():
    # ===== Usage Example =====
    # Replace with your own compatible endpoint
    backend = OpenAICompatibleBackend(
        base_url="https://api.siliconflow.cn/v1/",
        model="deepseek-ai/DeepSeek-V3.2",
        api_key="sk-",
    )

    engine = InstructionConsistencyEngine(backend, judge_mode="hybrid")

    # 1) Check consistency between two instructions
    ins_a = "You are a medical assistant."
    ins_b = "Write an ad for Nike."
    result = engine.determine_consistency(ins_a, ins_b)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    ins_a = "You should avoid topics about animals"
    ins_b = "How fast can a tiger runs"
    result = engine.determine_consistency(ins_a, ins_b)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 2) Generate a consistent instruction based on a reference
    consistent_res = engine.generate_instruction(
        reference_instruction="Please summarize the user-provided news into an abstract within 100 words.",
        mode="consistent",
        seed="output format restriction"
    )
    print(json.dumps(consistent_res, ensure_ascii=False, indent=2))

    # 3) Generate an inconsistent instruction based on a reference
    inconsistent_res = engine.generate_instruction(
        reference_instruction="Please summarize the user-provided news into an abstract within 100 words.",
        mode="inconsistent",
        conflict_types=["action", "constraints"],
    )
    print(json.dumps(inconsistent_res, ensure_ascii=False, indent=2))

    # ===== Simple LLM Consistency Checker Example =====
    print("\n" + "="*60)
    print("Simple LLM Consistency Checker Example")
    print("="*60)

    simple_checker = SimpleLLMConsistencyChecker(backend, temperature=0.0)

    # Example 1: Check consistency between two instructions
    print("\n--- Example 1: Simple Check ---")
    ins_a = "You are a medical assistant."
    ins_b = "Write an ad for Nike."
    simple_result = simple_checker.check_consistency_dict(ins_a, ins_b)
    print(json.dumps(simple_result, ensure_ascii=False, indent=2))

    # Example 2: Another check
    print("\n--- Example 2: Another Check ---")
    ins_a = "You should avoid topics about animals"
    ins_b = "How fast can a tiger run?"
    simple_result_2 = simple_checker.check_consistency_dict(ins_a, ins_b)
    print(json.dumps(simple_result_2, ensure_ascii=False, indent=2))

    # Example 3: Get ConflictResult object directly
    print("\n--- Example 3: Using ConflictResult directly ---")
    conflict_result = simple_checker.check_consistency(ins_a, ins_b)
    print(f"Conflict tuple: {conflict_result.as_tuple()}")
    print(f"Is consistent: {conflict_result.consistent}")
    print(f"Explanations: {conflict_result.explanations}")


if __name__ == "__main__":
    main()
