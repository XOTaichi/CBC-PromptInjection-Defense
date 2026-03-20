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


if __name__ == "__main__":
    main()
