#!/usr/bin/env python3
"""
Example usage of the consistency module.
"""

import json

from consistency import (
    OpenAICompatibleBackend,
    InstructionConsistencyEngine,
)


def main():
    # ===== Usage Example =====
    # Replace with your own compatible endpoint
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
        model="your-model-name",
        api_key="EMPTY",
    )

    engine = InstructionConsistencyEngine(backend, judge_mode="hybrid")

    # 1) Check consistency between two instructions
    ins_a = "Please summarize this paper, output 3 key points, and maintain a neutral tone."
    ins_b = "Please expand this paper in detail, write at least 1500 words, and add personal opinions."
    result = engine.determine_consistency(ins_a, ins_b)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 2) Generate a consistent instruction based on a reference
    consistent_res = engine.generate_instruction(
        reference_instruction="Please summarize the user-provided news into an abstract within 100 words.",
        mode="consistent",
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
