"""
Metrics module for consistency evaluation.

This module can be extended with additional metrics for evaluating
instruction consistency, such as:
- Precision/recall metrics for conflict detection
- Agreement metrics between different judges
- Statistical analysis utilities
"""

from typing import List, Tuple, Dict, Any
from .core import ConflictResult


def compute_agreement(results1: List[ConflictResult], results2: List[ConflictResult]) -> Dict[str, float]:
    """
    Compute agreement between two sets of conflict judgments.

    Args:
        results1: First list of conflict results
        results2: Second list of conflict results

    Returns:
        Dictionary with agreement metrics per dimension
    """
    if len(results1) != len(results2):
        raise ValueError("Result lists must have the same length")

    total = len(results1)
    if total == 0:
        return {
            "action_domain": 0.0,
            "action_constraint": 0.0,
            "domain_domain": 0.0,
            "constraint_constraint": 0.0,
            "overall": 0.0,
        }

    ad_agree = 0
    ac_agree = 0
    dd_agree = 0
    cc_agree = 0
    overall_agree = 0

    for r1, r2 in zip(results1, results2):
        if r1.action_domain_conflict == r2.action_domain_conflict:
            ad_agree += 1
        if r1.action_constraint_conflict == r2.action_constraint_conflict:
            ac_agree += 1
        if r1.domain_domain_conflict == r2.domain_domain_conflict:
            dd_agree += 1
        if r1.constraint_constraint_conflict == r2.constraint_constraint_conflict:
            cc_agree += 1
        if r1.as_tuple() == r2.as_tuple():
            overall_agree += 1

    return {
        "action_domain": ad_agree / total,
        "action_constraint": ac_agree / total,
        "domain_domain": dd_agree / total,
        "constraint_constraint": cc_agree / total,
        "overall": overall_agree / total,
    }


def conflict_distribution(results: List[ConflictResult]) -> Dict[str, int]:
    """
    Compute the distribution of conflicts across a list of results.

    Args:
        results: List of conflict results

    Returns:
        Dictionary with counts of conflicts per dimension
    """
    counts = {
        "action_domain": 0,
        "action_constraint": 0,
        "domain_domain": 0,
        "constraint_constraint": 0,
        "any_conflict": 0,
        "total": len(results),
    }

    for r in results:
        if r.action_domain_conflict:
            counts["action_domain"] += 1
        if r.action_constraint_conflict:
            counts["action_constraint"] += 1
        if r.domain_domain_conflict:
            counts["domain_domain"] += 1
        if r.constraint_constraint_conflict:
            counts["constraint_constraint"] += 1
        if not r.consistent:
            counts["any_conflict"] += 1

    return counts
