"""Layer 6: Prompt Constraints - Active request constraint enforcement."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_layer import BaseLayer, LayerResult
from ..schemas.request import VerificationRequest
from ..schemas.result import Severity
from ..schemas.rules import Rule, RuleType
from ..extractors.prompt_constraints import (
    PromptConstraintExtractor,
    ExtractedConstraints,
    ConstraintType,
    PromptConstraint,
)
from ..reasoning.datalog_engine import DatalogEngine, DatalogResult


@dataclass
class ConstraintCheckResult:
    """Result of checking a single constraint."""
    constraint: PromptConstraint
    satisfied: bool
    evidence: str = ""
    confidence: float = 1.0


class PromptConstraintsLayer(BaseLayer):
    """
    Layer 6: Prompt Constraints.

    Checks that responses comply with prompt-specific instructions:
    - Must-do requirements (agent must do X)
    - Must-not prohibitions (agent must not do Y)
    - Format requirements (respond in JSON, etc.)
    - Persona adherence (act as X)
    - Safety constraints (never provide harmful info)
    - Boundary limits (only answer about X)
    - Style requirements (be concise, formal, etc.)

    Constraints are extracted from:
    1. System prompts
    2. User messages
    3. Context (pre-extracted constraints)

    All constraint checking is done via Souffl Datalog.
    """

    BUILTIN_RULES = Path(__file__).parent.parent / "reasoning" / "rules" / "prompt_constraints.dl"

    def __init__(
        self,
        min_confidence: float = 0.7,
        strict_safety: bool = True,
    ):
        """
        Initialize Layer 6.

        Args:
            min_confidence: Minimum confidence threshold for constraints
            strict_safety: If True, safety violations are always errors
        """
        super().__init__(layer_number=6, layer_name="Prompt Constraints")

        self.datalog = DatalogEngine()
        self.extractor = PromptConstraintExtractor()
        self.min_confidence = min_confidence
        self.strict_safety = strict_safety

        # In-memory rule storage
        self._dynamic_rules: list[Rule] = []

        # Additional Datalog rules
        self._extracted_rules: list[str] = []

        self._load_basic_rules()

    def _load_basic_rules(self) -> None:
        """Load basic rule metadata for prompt constraints."""
        basic_rules = [
            Rule(
                rule_id="pc_must_do",
                name="Must-Do Requirement",
                description="Agent must perform required action",
                rule_type=RuleType.REQUIREMENT,
                layer=6,
                conditions=[],
                severity=Severity.WARNING,
                message_template="Required action not performed: {detail}",
                tags=["prompt", "requirement"],
            ),
            Rule(
                rule_id="pc_must_not",
                name="Must-Not Prohibition",
                description="Agent must not perform prohibited action",
                rule_type=RuleType.PROHIBITION,
                layer=6,
                conditions=[],
                severity=Severity.ERROR,
                message_template="Prohibited action performed: {detail}",
                tags=["prompt", "prohibition"],
            ),
            Rule(
                rule_id="pc_format",
                name="Format Requirement",
                description="Agent must use required format",
                rule_type=RuleType.CONSTRAINT,
                layer=6,
                conditions=[],
                severity=Severity.WARNING,
                message_template="Format mismatch: {detail}",
                tags=["prompt", "format"],
            ),
            Rule(
                rule_id="pc_safety",
                name="Safety Constraint",
                description="Agent must follow safety guidelines",
                rule_type=RuleType.CONSTRAINT,
                layer=6,
                conditions=[],
                severity=Severity.ERROR,
                message_template="Safety violation: {detail}",
                tags=["prompt", "safety"],
            ),
            Rule(
                rule_id="pc_boundary",
                name="Boundary Constraint",
                description="Agent must stay within scope boundaries",
                rule_type=RuleType.CONSTRAINT,
                layer=6,
                conditions=[],
                severity=Severity.WARNING,
                message_template="Boundary exceeded: {detail}",
                tags=["prompt", "boundary"],
            ),
        ]
        self._dynamic_rules.extend(basic_rules)

    # ========================================
    # Output Analysis
    # ========================================

    def _detect_format(self, output: str) -> str:
        """Detect the format of the output."""
        output_stripped = output.strip()

        # Check for JSON
        if (output_stripped.startswith("{") and output_stripped.endswith("}")) or \
           (output_stripped.startswith("[") and output_stripped.endswith("]")):
            try:
                import json
                json.loads(output_stripped)
                return "json"
            except json.JSONDecodeError:
                pass

        # Check for XML
        if output_stripped.startswith("<") and output_stripped.endswith(">"):
            return "xml"

        # Check for markdown code blocks
        if "```" in output:
            return "code_blocks"

        # Check for bullet list
        if re.search(r"^[\s]*[-*]\s+", output, re.MULTILINE):
            return "bullet_list"

        # Check for numbered list
        if re.search(r"^[\s]*\d+\.\s+", output, re.MULTILINE):
            return "numbered_list"

        # Default to prose
        return "prose"

    def _detect_style(self, output: str) -> str:
        """Detect the style of the output."""
        output_lower = output.lower()
        word_count = len(output.split())

        # Formal indicators
        formal_indicators = [
            "therefore", "furthermore", "consequently", "regarding",
            "pursuant", "hereby", "accordingly", "moreover",
        ]
        formal_score = sum(1 for ind in formal_indicators if ind in output_lower)

        # Casual indicators
        casual_indicators = [
            "hey", "cool", "awesome", "gonna", "wanna", "yeah",
            "btw", "fyi", "lol", "haha",
        ]
        casual_score = sum(1 for ind in casual_indicators if ind in output_lower)

        # Length-based detection
        if word_count < 50:
            length_style = "concise"
        elif word_count < 150:
            length_style = "moderate"
        else:
            length_style = "detailed"

        # Combine signals
        if casual_score >= 2:
            return "casual"
        if formal_score >= 2:
            return "formal"
        if length_style == "concise":
            return "concise"
        if length_style == "detailed":
            return "detailed"

        return "neutral"

    def _extract_mentions(self, output: str) -> list[str]:
        """Extract topics/things mentioned in the output."""
        mentions = []
        output_lower = output.lower()

        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', output)
        mentions.extend([q.lower() for q in quoted])

        # Extract capitalized phrases (potential named entities)
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', output)
        mentions.extend([c.lower() for c in caps])

        return list(set(mentions))

    def _extract_actions(self, output: str) -> list[str]:
        """Extract actions the output performs."""
        actions = []
        output_lower = output.lower()

        # Action patterns
        action_patterns = [
            r"(?:i will|i'll|let me|i can|i'm going to)\s+([^.!?,]+)",
            r"(?:here is|here's|here are)\s+([^.!?,]+)",
            r"(?:i have|i've)\s+([^.!?,]+)",
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, output_lower)
            actions.extend(matches)

        return list(set(actions))

    def _extract_topics(self, output: str) -> list[str]:
        """Extract discussion topics from output."""
        topics = []
        output_lower = output.lower()

        # Topic keywords
        topic_keywords = {
            "programming": ["code", "function", "variable", "class", "method", "programming"],
            "database": ["sql", "query", "table", "database", "schema"],
            "security": ["password", "encryption", "authentication", "security"],
            "networking": ["http", "api", "request", "response", "server"],
            "finance": ["money", "payment", "transaction", "account", "balance"],
            "health": ["medical", "health", "symptom", "treatment", "diagnosis"],
            "legal": ["law", "legal", "court", "attorney", "lawsuit"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in output_lower for kw in keywords):
                topics.append(topic)

        return topics

    def _detect_safety_issues(self, output: str) -> list[str]:
        """Detect potential safety issues in output."""
        issues = []
        output_lower = output.lower()

        # Harmful content patterns
        harmful_patterns = [
            (r"how\s+to\s+(?:hack|attack|exploit)", "hacking_instructions"),
            (r"(?:password|credential)s?\s*[:=]", "credential_exposure"),
            (r"(?:kill|harm|hurt|attack)\s+(?:someone|people|person)", "violence"),
            (r"(?:illegal|illicit)\s+(?:drug|substance)", "illegal_substances"),
        ]

        for pattern, issue_type in harmful_patterns:
            if re.search(pattern, output_lower):
                issues.append(issue_type)

        return issues

    # ========================================
    # Datalog Integration
    # ========================================

    def add_extracted_rule(self, datalog_rule: str) -> None:
        """
        Add an extracted Datalog rule string.

        Args:
            datalog_rule: Datalog rule code as string
        """
        self._extracted_rules.append(datalog_rule)

    def clear_extracted_rules(self) -> None:
        """Clear all extracted rules."""
        self._extracted_rules.clear()

    def _get_rules_program(self) -> str:
        """Combine all rules into a single Datalog program."""
        program = ""

        if self.BUILTIN_RULES.exists():
            with open(self.BUILTIN_RULES) as f:
                program = f.read()

        if self._extracted_rules:
            program += "\n\n// Extracted Rules\n"
            for rule in self._extracted_rules:
                program += rule + "\n"

        return program

    def _populate_datalog_facts(
        self,
        case_id: str,
        request: VerificationRequest,
        context: dict[str, Any],
        constraints: ExtractedConstraints,
    ) -> None:
        """Populate the Datalog engine with facts for constraint checking."""
        self.datalog.clear_facts()

        # Basic request info
        self.datalog.add_fact("request_id", case_id)

        output = request.llm_output

        # Add constraint facts
        for i, constraint in enumerate(constraints.must_do):
            if constraint.confidence >= self.min_confidence:
                self.datalog.add_fact(
                    "must_do_constraint",
                    case_id,
                    f"must_do_{i}",
                    constraint.content[:100],
                    constraint.source,
                    int(constraint.confidence * 100)
                )

        for i, constraint in enumerate(constraints.must_not):
            if constraint.confidence >= self.min_confidence:
                self.datalog.add_fact(
                    "must_not_constraint",
                    case_id,
                    f"must_not_{i}",
                    constraint.content[:100],
                    constraint.source,
                    int(constraint.confidence * 100)
                )

        for i, constraint in enumerate(constraints.format_requirements):
            if constraint.confidence >= self.min_confidence:
                self.datalog.add_fact(
                    "format_constraint",
                    case_id,
                    f"format_{i}",
                    constraint.content[:50],
                    constraint.source,
                    int(constraint.confidence * 100)
                )

        for i, constraint in enumerate(constraints.safety_constraints):
            if constraint.confidence >= self.min_confidence:
                self.datalog.add_fact(
                    "safety_constraint",
                    case_id,
                    f"safety_{i}",
                    constraint.content[:100],
                    constraint.source,
                    int(constraint.confidence * 100)
                )

        for i, constraint in enumerate(constraints.boundaries):
            if constraint.confidence >= self.min_confidence:
                self.datalog.add_fact(
                    "boundary_constraint",
                    case_id,
                    f"boundary_{i}",
                    constraint.content[:100],
                    constraint.source,
                    int(constraint.confidence * 100)
                )

        for i, constraint in enumerate(constraints.style_requirements):
            if constraint.confidence >= self.min_confidence:
                self.datalog.add_fact(
                    "style_constraint",
                    case_id,
                    f"style_{i}",
                    constraint.content[:50],
                    constraint.source,
                    int(constraint.confidence * 100)
                )

        if constraints.persona and constraints.persona.confidence >= self.min_confidence:
            self.datalog.add_fact(
                "persona_constraint",
                case_id,
                constraints.persona.content[:100],
                constraints.persona.source,
                int(constraints.persona.confidence * 100)
            )

        # Add output analysis facts
        output_format = self._detect_format(output)
        self.datalog.add_fact("output_format", case_id, output_format)

        output_style = self._detect_style(output)
        self.datalog.add_fact("output_style", case_id, output_style)

        for mention in self._extract_mentions(output)[:20]:  # Limit to 20
            self.datalog.add_fact("output_mentions", case_id, mention[:50])

        for action in self._extract_actions(output)[:10]:  # Limit to 10
            self.datalog.add_fact("output_does", case_id, action[:50])

        for topic in self._extract_topics(output):
            self.datalog.add_fact("output_topic", case_id, topic)

        for issue in self._detect_safety_issues(output):
            self.datalog.add_fact("output_safety_issue", case_id, issue)

    def _parse_datalog_violations(
        self,
        result: DatalogResult,
        case_id: str,
    ) -> list[tuple[str, str, str, str]]:
        """
        Parse violations from Datalog output.

        Returns:
            List of (violation_type, constraint_id, detail, severity) tuples
        """
        violations = []
        for row in result.get_relation("output_constraint_violation"):
            if len(row) >= 5 and row[0] == case_id:
                violations.append((row[1], row[2], row[3], row[4]))
        return violations

    # ========================================
    # Verification
    # ========================================

    def extract_constraints(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> ExtractedConstraints:
        """
        Extract constraints from request and context.

        Args:
            request: The verification request
            context: Accumulated context

        Returns:
            ExtractedConstraints object
        """
        # Check if constraints are pre-extracted in context
        if "extracted_constraints" in context:
            return context["extracted_constraints"]

        # Extract from request
        system_prompt = context.get("system_prompt")
        return self.extractor.extract(system_prompt, request.prompt)

    def check(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> LayerResult:
        """
        Check prompt constraints against the response.

        Args:
            request: The verification request
            context: Accumulated context from previous layers

        Returns:
            LayerResult with violations and reasoning
        """
        result = LayerResult(layer=self.layer_number)
        case_id = request.request_id

        # Step 1: Extract constraints
        constraints = self.extract_constraints(request, context)

        total_constraints = len(constraints.all_constraints())

        result.add_reasoning(self.create_reasoning_step(
            step_type="constraint_extraction",
            description="Extracted prompt constraints",
            inputs={
                "has_system_prompt": "system_prompt" in context,
                "prompt_length": len(request.prompt),
            },
            outputs={
                "must_do_count": len(constraints.must_do),
                "must_not_count": len(constraints.must_not),
                "format_count": len(constraints.format_requirements),
                "safety_count": len(constraints.safety_constraints),
                "boundary_count": len(constraints.boundaries),
                "style_count": len(constraints.style_requirements),
                "has_persona": constraints.persona is not None,
                "total_constraints": total_constraints,
            },
        ))

        if total_constraints == 0:
            result.add_reasoning(self.create_reasoning_step(
                step_type="skip",
                description="No constraints to check",
                inputs={},
                outputs={"skipped": True},
            ))
            result.metadata["constraints_satisfied"] = True
            result.metadata["compliance_score"] = 2
            return result

        # Step 2: Populate facts
        self._populate_datalog_facts(case_id, request, context, constraints)

        # Step 3: Run Datalog rules
        program = self._get_rules_program()

        if program.strip():
            datalog_result = self.datalog.run_inline(
                program,
                output_relations=[
                    "output_constraint_compliance",
                    "output_constraint_violation",
                    "output_constraint_summary",
                    "output_violation_count",
                ]
            )

            if datalog_result.success:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description="Applied prompt constraint Datalog rules",
                    inputs={"program_lines": len(program.split("\n"))},
                    outputs={"success": True},
                ))

                # Parse violations
                dl_violations = self._parse_datalog_violations(datalog_result, case_id)

                for vtype, cid, detail, severity in dl_violations:
                    # Override severity for safety violations if strict mode
                    if self.strict_safety and vtype == "safety":
                        severity = "error"

                    result.add_violation(self.create_violation(
                        violation_type=vtype,
                        message=f"Constraint violation ({vtype}): {detail}",
                        evidence={
                            "constraint_id": cid,
                            "detail": detail,
                            "source": "datalog",
                        },
                        severity=severity,
                        rule_id=f"pc_{vtype}",
                    ))

                # Store compliance info in metadata
                for row in datalog_result.get_relation("output_constraint_compliance"):
                    if len(row) >= 3 and row[0] == case_id:
                        result.metadata["constraints_satisfied"] = row[1] == "true"
                        result.metadata["compliance_score"] = int(row[2])

                # Store summary
                for row in datalog_result.get_relation("output_constraint_summary"):
                    if len(row) >= 4 and row[0] == case_id:
                        result.metadata["total_constraints"] = int(row[1])
                        result.metadata["violation_count"] = int(row[2])
            else:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description=f"Datalog execution failed: {datalog_result.error}",
                    inputs={"program_lines": len(program.split("\n"))},
                    outputs={"success": False, "error": datalog_result.error},
                ))
        else:
            result.add_reasoning(self.create_reasoning_step(
                step_type="rule_application",
                description="No Datalog rules to apply",
                inputs={},
                outputs={"skipped": True},
            ))

        # Store constraint info for context
        result.facts_extracted = {
            "total_constraints": total_constraints,
            "constraints_satisfied": result.metadata.get("constraints_satisfied", True),
            "compliance_score": result.metadata.get("compliance_score", 2),
        }

        return result

    def load_rules(self, deployment_id: str) -> list[Rule]:
        """
        Load rules for this layer.

        Args:
            deployment_id: The deployment identifier

        Returns:
            List of active rules
        """
        return list(self._dynamic_rules)


# Convenience functions

def check_constraints(
    prompt: str,
    output: str,
    system_prompt: str | None = None,
) -> LayerResult:
    """
    Quick check of prompt constraints.

    Args:
        prompt: User prompt
        output: LLM output
        system_prompt: Optional system prompt

    Returns:
        LayerResult with violations
    """
    layer = PromptConstraintsLayer()

    request = VerificationRequest(
        request_id="quick_check",
        deployment_id="default",
        prompt=prompt,
        llm_output=output,
        llm_model="unknown",
    )

    context = {}
    if system_prompt:
        context["system_prompt"] = system_prompt

    return layer.check(request, context)
