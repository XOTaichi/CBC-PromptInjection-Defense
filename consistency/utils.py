from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def _strip_code_fence(text: str) -> str:
    """Strip code fence markers from text."""
    text = text.strip()
    text = re.sub(r"^```(?:json|python|text)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_block(text: str) -> str:
    """Extract the first complete JSON object from text."""
    text = _strip_code_fence(text)
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        raise ValueError(f"JSON object not found: {text}")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError(f"Incomplete JSON object: {text}")


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Safely load JSON from text, extracting the first JSON block if needed."""
    block = _extract_first_json_block(text)
    try:
        return json.loads(block)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {e}\nOriginal text: {text}") from e


def _norm_text(x: Optional[str]) -> str:
    """Normalize text: lowercase, strip whitespace, collapse multiple spaces."""
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def _norm_list(xs: Optional[List[str]]) -> List[str]:
    """Normalize a list of texts, removing duplicates and empty strings."""
    if not xs:
        return []
    out = []
    seen = set()
    for x in xs:
        nx = _norm_text(x)
        if nx and nx not in seen:
            seen.add(nx)
            out.append(nx)
    return out


def _as_clean_list(value: Any) -> List[str]:
    """Convert any value to a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return [str(value).strip()]
