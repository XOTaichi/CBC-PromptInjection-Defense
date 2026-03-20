"""
CBC Prompt Injection Defense - Consistency Module

This module provides tools for parsing natural language instructions,
checking their consistency, and generating consistent/inconsistent instructions.
"""

from .core import (
    LLMBackend,
    OpenAICompatibleBackend,
    ParsedInstruction,
    ConflictResult,
    InstructionConsistencyEngine,
    SimpleLLMConsistencyChecker,
)
from .metrics import compute_agreement, conflict_distribution

__all__ = [
    # Core classes
    "LLMBackend",
    "OpenAICompatibleBackend",
    "ParsedInstruction",
    "ConflictResult",
    "InstructionConsistencyEngine",
    "SimpleLLMConsistencyChecker",
    # Metrics
    "compute_agreement",
    "conflict_distribution",
]
