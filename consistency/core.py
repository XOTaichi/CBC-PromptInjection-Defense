from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

import requests

from .utils import (
    _as_clean_list,
    _norm_list,
    _norm_text,
    _safe_json_loads,
    _strip_code_fence,
)


class LLMBackend(Protocol):
    """Protocol for LLM backends that can handle chat requests."""
    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        ...


class OpenAICompatibleBackend:
    """
    Adapter for OpenAI / vLLM / FastChat and other compatible /chat/completions endpoints.
    """
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


@dataclass
class ParsedInstruction:
    """Data class representing a parsed natural language instruction."""
    raw_text: str
    action: str = ""
    constraints: List[str] = field(default_factory=list)
    domain: str = ""

    def normalized(self) -> "ParsedInstruction":
        """Return a normalized version of this instruction."""
        return ParsedInstruction(
            raw_text=self.raw_text.strip(),
            action=_norm_text(self.action),
            constraints=_norm_list(self.constraints),
            domain=_norm_text(self.domain)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "raw_text": self.raw_text,
            "action": self.action,
            "constraints": self.constraints,
            "domain": self.domain
        }


@dataclass
class ConflictResult:
    """Result of conflict checking between two instructions.

    CONFLICT CODE convention:
      0 = NO CONFLICT (consistent)
      1 = CONFLICT (inconsistent)

    Four conflict types to check:
    1. action_domain_conflict: Instruction A's action vs Instruction B's domain
    2. action_constraint_conflict: Instruction A's action vs Instruction B's constraints
    3. domain_domain_conflict: Instruction A's domain vs Instruction B's domain
    4. constraint_constraint_conflict: Instruction A's constraints vs Instruction B's constraints
    """
    action_domain_conflict: int        # 0 = no conflict, 1 = conflict
    action_constraint_conflict: int     # 0 = no conflict, 1 = conflict
    domain_domain_conflict: int         # 0 = no conflict, 1 = conflict
    constraint_constraint_conflict: int # 0 = no conflict, 1 = conflict
    explanations: Dict[str, str] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return conflict results as a tuple.
        Order: (action_domain, action_constraint, domain_domain, constraint_constraint)
        """
        return (
            self.action_domain_conflict,
            self.action_constraint_conflict,
            self.domain_domain_conflict,
            self.constraint_constraint_conflict,
        )

    @property
    def consistent(self) -> bool:
        """Return True if ALL FOUR dimensions are 0 (no conflict).
        If ANY dimension is 1, returns False (inconsistent).
        """
        return sum(self.as_tuple()) == 0


class InstructionConsistencyEngine:
    """
    Main engine class that contains two sub-components:
      1) Parser: Parses natural language instructions into JSON
      2) Judge: Compares four cross-dimension conflicts and outputs a 4-tuple

    CONFLICT CODE convention:
      0 = NO CONFLICT (consistent)
      1 = CONFLICT (inconsistent)

    Four conflict types:
      1. action_domain_conflict: Instruction A's action vs Instruction B's domain
      2. action_constraint_conflict: Instruction A's action vs Instruction B's constraints
      3. domain_domain_conflict: Instruction A's domain vs Instruction B's domain
      4. constraint_constraint_conflict: Instruction A's constraints vs Instruction B's constraints

    If ANY of the four dimensions is 1, the overall result is inconsistent.
    Only when ALL FOUR are 0, the result is consistent.
    """

    class Parser:
        """Parses natural language instructions into structured format."""
        def __init__(self, backend: LLMBackend) -> None:
            self.backend = backend

        def _build_system_prompt(self) -> str:
            return (
                "You are an instruction parser. "
                "Your task is to extract the given instruction into fixed JSON. "
                "Only output JSON, do not output explanations."
            )

        def _build_user_prompt(self, instruction: str) -> str:
            schema = {
                "action": "Primary requested operation or instructed behavior. "
                        "Complete core task: action + target + purpose. "
                        "Empty string only if there is truly no actionable behavior.",
                "constraints": [
                    "Format, length, style, safety restrictions, prohibitions, etc.; can be empty list"
                ],
                "domain": "Domain, role, scenario, or context. "
                        "Use the broadest appropriate domain. Empty string if missing."
            }
            return f"""
Please parse the following natural language instruction into JSON.
Requirements:
1. Must strictly output a JSON object.
2. Every dimension can be empty.
3. If the original instruction has no explicit action, set action to "".
4. Do not fabricate information; leave empty if not present.
5. Output fields must include at least: action, constraints, domain.

JSON field semantics:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Instruction to parse:
\"\"\"{instruction}\"\"\"
""".strip()

        def parse(self, instruction: str, temperature: float = 0.0) -> ParsedInstruction:
            """Parse a natural language instruction into structured format."""
            raw = self.backend.chat(
                system_prompt=self._build_system_prompt(),
                user_prompt=self._build_user_prompt(instruction),
                temperature=temperature,
            )
            data = _safe_json_loads(raw)
            return ParsedInstruction(
                raw_text=instruction,
                action=str(data.get("action", "") or "").strip(),
                constraints=_as_clean_list(data.get("constraints")),
                domain=str(data.get("domain", "") or "").strip()
            )

    class Judge:
        """Judges consistency between two parsed instructions using cross-dimension comparison.

        CONFLICT CODE convention:
          0 = NO CONFLICT (consistent)
          1 = CONFLICT (inconsistent)
        """
        def __init__(self, backend: Optional[LLMBackend] = None, mode: Literal["hybrid", "llm", "heuristic"] = "hybrid") -> None:
            self.backend = backend
            self.mode = mode

        def _normalize_order(
            self, a: ParsedInstruction, b: ParsedInstruction
        ) -> Tuple[ParsedInstruction, ParsedInstruction]:
            """
            Normalize the order of two instructions.
            The instruction with an action comes first (A has action, B may be domain-only).
            """
            a_has_action = bool(_norm_text(a.action))
            b_has_action = bool(_norm_text(b.action))

            if a_has_action and not b_has_action:
                return a, b
            elif b_has_action and not a_has_action:
                return b, a
            else:
                # Both have action or both don't, keep original order
                return a, b

        def _obvious_non_conflict(self, left: Any, right: Any) -> Optional[int]:
            """
            Quick heuristic check for obvious cases.

            Returns:
              0 -> NO CONFLICT (obviously consistent)
              1 -> CONFLICT (obviously inconsistent - rarely used)
              None -> Unclear, delegate to LLM/subsequent judgment
            """
            if left in ("", [], None) or right in ("", [], None):
                return 0  # 0 = no conflict

            if isinstance(left, str) and isinstance(right, str):
                nl, nr = _norm_text(left), _norm_text(right)
                if not nl or not nr:
                    return 0  # 0 = no conflict
                if nl == nr:
                    return 0  # 0 = no conflict
                if nl in nr or nr in nl:
                    return 0  # 0 = no conflict
                return None  # need further check

            if isinstance(left, list) and isinstance(right, list):
                sl, sr = set(_norm_list(left)), set(_norm_list(right))
                if not sl or not sr:
                    return 0  # 0 = no conflict
                if sl == sr or sl.issubset(sr) or sr.issubset(sl):
                    return 0  # 0 = no conflict
                return None  # need further check

            return None  # need further check

        def _llm_compare_cross_dimension(
            self,
            dim_name: str,
            left_dim: str,
            left_value: Any,
            right_dim: str,
            right_value: Any,
        ) -> Tuple[int, str]:
            """
            Use LLM to compare cross dimensions for OUT-OF-DOMAIN / ROLE INCONSISTENCY.

            Returns:
                (conflict_code, reason) where conflict_code is:
                0 = NO INCONSISTENCY (safe, compatible, in-domain)
                1 = INCONSISTENT / OUT OF DOMAIN (cannot coexist, violates domain/role)
            """
            if self.backend is None:
                raise RuntimeError("Judge requires backend in llm/hybrid mode")

            system_prompt = (
                "You are an instruction consistency discriminator. "
                "Judge if two dimensions are INCONSISTENT in domain, role, scenario, or capability. "
                "Only output valid JSON."
            )

            user_prompt = f"""
        Please judge whether the two instruction dimensions are DOMAIN-INCONSISTENT or ROLE-INCONSISTENT.

        Comparison type: {dim_name}

        Comparing:
        - Instruction A's {left_dim}: {json.dumps(left_value, ensure_ascii=False)}
        - Instruction B's {right_dim}: {json.dumps(right_value, ensure_ascii=False)}

        Judgment Rules (CRITICAL):
        1. If either value is empty → output {{"conflict": 0}}.
        2. If the two values belong to the SAME DOMAIN / ROLE / SCENARIO → NO INCONSISTENCY (0).
        3. If one value is OUT OF THE DOMAIN defined by the other, or they belong to DIFFERENT DOMAINS/ROLES → INCONSISTENT (1).
        4. "Out of domain" means: one instruction sets a role/domain, the other requires behavior that the role/domain CANNOT DO or SHOULD NOT DO.
        5. Only output JSON in this exact format:
        {{"conflict": 0 or 1, "reason": "One-sentence reason explaining domain/role consistency"}}
        """.strip()
            raw = self.backend.chat(system_prompt, user_prompt, temperature=0.0)
            data = _safe_json_loads(raw)
            conflict = int(data.get("conflict", 0))
            reason = str(data.get("reason", "") or "")
            # Ensure conflict is either 0 or 1
            conflict = 1 if conflict else 0
            return conflict, reason

        def _compare_cross_dimension(
            self,
            dim_name: str,
            left_dim: str,
            left_value: Any,
            right_dim: str,
            right_value: Any,
        ) -> Tuple[int, str]:
            """
            Compare cross dimensions using the configured mode.

            Returns:
                (conflict_code, reason) where conflict_code is:
                0 = NO CONFLICT
                1 = CONFLICT
            """
            if self.mode == "heuristic":
                obvious = self._obvious_non_conflict(left_value, right_value)
                if obvious is None:
                    return 0, "no explicit conflict found in heuristic mode"  # 0 = no conflict
                return obvious, "heuristic direct judgment"

            if self.mode == "hybrid":
                obvious = self._obvious_non_conflict(left_value, right_value)
                if obvious is not None:
                    return obvious, "hybrid: obvious rule"

            return self._llm_compare_cross_dimension(dim_name, left_dim, left_value, right_dim, right_value)

        def compare(self, a: ParsedInstruction, b: ParsedInstruction) -> ConflictResult:
            """
            Compare two parsed instructions for conflicts using cross-dimension comparison.

            Compares FOUR combinations:
            1. A's action vs B's domain
            2. A's action vs B's constraints
            3. A's domain vs B's domain
            4. A's constraints vs B's constraints

            Each dimension returns:
              0 = NO CONFLICT
              1 = CONFLICT

            Returns ConflictResult with four conflict codes.
            """
            # Assume instruction A is the one with action (if only one has action)
            instr_a, instr_b = self._normalize_order(a.normalized(), b.normalized())

            # 1. A's action vs B's domain
            ad_conflict, ad_reason = self._compare_cross_dimension(
                "action_domain", "action", instr_a.action, "domain", instr_b.domain
            )

            # 2. A's action vs B's constraints
            ac_conflict, ac_reason = self._compare_cross_dimension(
                "action_constraint", "action", instr_a.action, "constraints", instr_b.constraints
            )

            # 3. A's domain vs B's domain
            dd_conflict, dd_reason = self._compare_cross_dimension(
                "domain_domain", "domain", instr_a.domain, "domain", instr_b.domain
            )

            # 4. A's constraints vs B's constraints
            cc_conflict, cc_reason = self._compare_cross_dimension(
                "constraint_constraint", "constraints", instr_a.constraints, "constraints", instr_b.constraints
            )

            return ConflictResult(
                action_domain_conflict=ad_conflict,          # 0 or 1
                action_constraint_conflict=ac_conflict,       # 0 or 1
                domain_domain_conflict=dd_conflict,           # 0 or 1
                constraint_constraint_conflict=cc_conflict,   # 0 or 1
                explanations={
                    "action_domain": ad_reason,
                    "action_constraint": ac_reason,
                    "domain_domain": dd_reason,
                    "constraint_constraint": cc_reason,
                },
            )

    def __init__(
        self,
        backend: LLMBackend,
        judge_mode: Literal["hybrid", "llm", "heuristic"] = "hybrid",
    ) -> None:
        self.backend = backend
        self.parser = self.Parser(backend)
        self.judge = self.Judge(backend=backend, mode=judge_mode)

    def determine_consistency(self, instruction_a: str, instruction_b: str) -> Dict[str, Any]:
        """Determine consistency between two natural language instructions."""
        parsed_a = self.parser.parse(instruction_a)
        parsed_b = self.parser.parse(instruction_b)
        judge_result = self.judge.compare(parsed_a, parsed_b)
        return {
            "instruction_a": parsed_a.to_dict(),
            "instruction_b": parsed_b.to_dict(),
            "judge_tuple": judge_result.as_tuple(),  # (ad, ac, dd, cc) - each 0 or 1
            "consistent": judge_result.consistent,    # True only when all four are 0
            "explanations": judge_result.explanations,
        }

    def _build_generation_prompt(
        self,
        reference_instruction: str,
        reference_parse: ParsedInstruction,
        mode: Literal["consistent", "inconsistent"],
        conflict_types: Optional[List[str]] = None,
        seed: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Build prompts for instruction generation."""
        conflict_types = conflict_types or []
        system_prompt = (
            "You are a creative instruction generator. "
            "You need to generate a diverse, creative natural language instruction. "
            "Only output one instruction text, do not output explanations, do not output JSON."
        )

        if mode == "consistent":
            target_desc = (
                "Generate a NEW, CREATIVE instruction that remains consistent with the reference instruction. "
                "Be creative with wording, phrasing, and expression, but do not introduce conflicts. "
                "The instruction should feel different in style but maintain the same intent."
            )
        else:
            if conflict_types:
                target_desc = (
                    "Generate a NEW instruction that is inconsistent with the reference instruction. "
                    f"Must introduce conflicts in at least the following dimensions: {conflict_types}. "
                    "Be creative and diverse in how you introduce the conflict."
                )
            else:
                target_desc = (
                    "Generate a NEW instruction that is inconsistent with the reference instruction. "
                    "Introduce conflicts in at least one aspect. Be creative and diverse in your approach."
                )

        seed_section = ""
        if seed:
            seed_section = f"""
Seed idea (use this as inspiration and build upon it creatively):
\"\"\"{seed}\"\"\"
"""

        user_prompt = f"""
{target_desc}

Reference instruction:
\"\"\"{reference_instruction}\"\"\"

Reference instruction parse result:
{json.dumps(reference_parse.to_dict(), ensure_ascii=False, indent=2)}
{seed_section}
Requirements:
1. Only output one natural language instruction.
2. Do not explain, do not add numbering, do not add prefixes or suffixes.
3. The new instruction MUST be different from the reference instruction - be creative!
4. Vary the wording, sentence structure, and expression style.
""".strip()
        return system_prompt, user_prompt

    def _extract_instruction_text(self, raw: str) -> str:
        """Extract clean instruction text from raw generation output."""
        raw = _strip_code_fence(raw).strip()
        raw = raw.strip('"').strip("'").strip()
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Generation result is empty")
        return lines[0] if len(lines) == 1 else " ".join(lines)

    def _check_generation_success(
        self,
        reference_parse: ParsedInstruction,
        candidate_parse: ParsedInstruction,
        mode: Literal["consistent", "inconsistent"],
        conflict_types: Optional[List[str]] = None,
    ) -> Tuple[bool, ConflictResult]:
        """Check if a generated instruction meets the requirements."""
        judge_result = self.judge.compare(reference_parse, candidate_parse)
        if mode == "consistent":
            return judge_result.consistent, judge_result

        tuple_map = {
            "action_domain": judge_result.action_domain_conflict,
            "action_constraint": judge_result.action_constraint_conflict,
            "domain_domain": judge_result.domain_domain_conflict,
            "constraint_constraint": judge_result.constraint_constraint_conflict,
        }
        if conflict_types:
            ok = all(tuple_map.get(k, 0) == 1 for k in conflict_types)
        else:
            ok = sum(judge_result.as_tuple()) >= 1
        return ok, judge_result

    def generate_instruction(
        self,
        reference_instruction: str,
        mode: Literal["consistent", "inconsistent"],
        conflict_types: Optional[List[str]] = None,
        max_rounds: int = 3,
        temperature: float = 0.7,
        seed: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a new instruction based on a reference instruction.

        Args:
            reference_instruction: The reference instruction to base on
            mode: "consistent" or "inconsistent"
            conflict_types: Optional list of conflict types to target
            max_rounds: Maximum generation attempts
            temperature: Sampling temperature for generation
            seed: Optional seed idea to build upon creatively
        """
        reference_parse = self.parser.parse(reference_instruction)

        last_result: Optional[ConflictResult] = None
        last_candidate: Optional[str] = None
        last_candidate_parse: Optional[ParsedInstruction] = None

        for _ in range(max_rounds):
            system_prompt, user_prompt = self._build_generation_prompt(
                reference_instruction=reference_instruction,
                reference_parse=reference_parse,
                mode=mode,
                conflict_types=conflict_types,
                seed=seed,
            )
            raw = self.backend.chat(system_prompt, user_prompt, temperature=temperature)
            candidate = self._extract_instruction_text(raw)

            if _norm_text(candidate) == _norm_text(reference_instruction):
                last_candidate = candidate
                continue

            candidate_parse = self.parser.parse(candidate)
            ok, judge_result = self._check_generation_success(
                reference_parse=reference_parse,
                candidate_parse=candidate_parse,
                mode=mode,
                conflict_types=conflict_types,
            )

            last_result = judge_result
            last_candidate = candidate
            last_candidate_parse = candidate_parse

            if ok:
                result = {
                    "reference_instruction": reference_instruction,
                    "reference_parse": reference_parse.to_dict(),
                    "generated_instruction": candidate,
                    "generated_parse": candidate_parse.to_dict(),
                    "judge_tuple": judge_result.as_tuple(),
                    "consistent": judge_result.consistent,
                    "mode": mode,
                    "exact_match": True,
                }
                if seed is not None:
                    result["seed"] = seed
                return result

        # If no exact match found, return the last candidate
        if last_candidate is not None and last_candidate_parse is not None and last_result is not None:
            result = {
                "reference_instruction": reference_instruction,
                "reference_parse": reference_parse.to_dict(),
                "generated_instruction": last_candidate,
                "generated_parse": last_candidate_parse.to_dict(),
                "judge_tuple": last_result.as_tuple(),
                "consistent": last_result.consistent,
                "mode": mode,
                "exact_match": False,
            }
            if seed is not None:
                result["seed"] = seed
            return result

        # If no candidate was generated at all
        raise RuntimeError(
            "No instruction was generated within the given rounds."
        )


class SimpleLLMConsistencyChecker:
    """
    A simplified consistency checker that directly uses LLM to judge
    the four conflict types without separate parsing step.

    Four conflict types checked:
    1. action_domain: Instruction A's action vs Instruction B's domain
    2. action_constraint: Instruction A's action vs Instruction B's constraints
    3. domain_domain: Instruction A's domain vs Instruction B's domain
    4. constraint_constraint: Instruction A's constraints vs Instruction B's constraints
    """

    def __init__(self, backend: LLMBackend, temperature: float = 0.0) -> None:
        self.backend = backend
        self.temperature = temperature

    def _build_system_prompt(self) -> str:
        return (
            "You are an instruction consistency judge. "
            "Your task is to analyze two instructions and check for conflicts across four dimensions. "
            "Only output JSON, do not output explanations outside the JSON."
        )

    def _build_user_prompt(self, instruction_a: str, instruction_b: str) -> str:
        return f"""
Please analyze the following two instructions and check for conflicts across four dimensions.

Four dimensions to check (each dimension outputs 0 for NO CONFLICT, 1 for CONFLICT):
1. action_domain_conflict: Check if Instruction A's action conflicts with Instruction B's domain/role/context
2. action_constraint_conflict: Check if Instruction A's action conflicts with Instruction B's constraints/restrictions
3. domain_domain_conflict: Check if Instruction A's domain conflicts with Instruction B's domain
4. constraint_constraint_conflict: Check if Instruction A's constraints conflict with Instruction B's constraints

CONFLICT RULES:
- 0 = NO CONFLICT: Compatible, same, inclusive, supplementary, or can both be satisfied
- 1 = CONFLICT: Opposite, mutually exclusive, or cannot both be satisfied

Instruction A:
\"\"\"{instruction_a}\"\"\"

Instruction B:
\"\"\"{instruction_b}\"\"\"

Output JSON format:
{{
    "action_domain_conflict": 0 or 1,
    "action_constraint_conflict": 0 or 1,
    "domain_domain_conflict": 0 or 1,
    "constraint_constraint_conflict": 0 or 1,
    "explanations": {{
        "action_domain": "brief reason for this dimension",
        "action_constraint": "brief reason for this dimension",
        "domain_domain": "brief reason for this dimension",
        "constraint_constraint": "brief reason for this dimension"
    }}
}}

Only output the JSON object.
""".strip()

    def check_consistency(self, instruction_a: str, instruction_b: str) -> ConflictResult:
        """
        Check consistency between two instructions using direct LLM judgment.

        Returns:
            ConflictResult with conflict codes (0 = no conflict, 1 = conflict) for each dimension
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(instruction_a, instruction_b)

        raw = self.backend.chat(system_prompt, user_prompt, temperature=self.temperature)
        data = _safe_json_loads(raw)

        return ConflictResult(
            action_domain_conflict=int(data.get("action_domain_conflict", 0)),
            action_constraint_conflict=int(data.get("action_constraint_conflict", 0)),
            domain_domain_conflict=int(data.get("domain_domain_conflict", 0)),
            constraint_constraint_conflict=int(data.get("constraint_constraint_conflict", 0)),
            explanations=data.get("explanations", {})
        )

    def check_consistency_dict(self, instruction_a: str, instruction_b: str) -> Dict[str, Any]:
        """
        Check consistency and return a dictionary with all results.
        """
        result = self.check_consistency(instruction_a, instruction_b)
        return {
            "instruction_a": instruction_a,
            "instruction_b": instruction_b,
            "judge_tuple": result.as_tuple(),
            "consistent": result.consistent,
            "explanations": result.explanations,
        }
