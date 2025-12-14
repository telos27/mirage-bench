"""Layer 1: Common Knowledge - Universal truths and consistency rules."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_layer import BaseLayer, LayerResult
from ..schemas.request import VerificationRequest
from ..schemas.result import Severity
from ..schemas.rules import Rule, RuleType
from ..reasoning.datalog_engine import DatalogEngine, DatalogResult


@dataclass
class ExtractedFacts:
    """Facts extracted from request for Datalog reasoning."""

    # Context facts (what the agent can "see")
    context_elements: list[str] = field(default_factory=list)
    context_errors: list[str] = field(default_factory=list)
    context_warnings: list[str] = field(default_factory=list)
    action_history: list[tuple[str, str]] = field(default_factory=list)  # (action, outcome)
    request_task: str = ""

    # Output facts (what the agent produces)
    output_references: list[str] = field(default_factory=list)
    output_claims: list[str] = field(default_factory=list)
    output_action: str = ""
    output_target: str = ""
    acknowledges_error: bool = False
    acknowledges_history: bool = False


class CommonKnowledgeLayer(BaseLayer):
    """
    Layer 1: Common Knowledge.

    Checks universal truths that apply to all agents:
    - Logical consistency
    - Grounded references
    - Error awareness
    - Repeated failure detection

    Rules can be:
    1. Built-in (hardcoded Datalog rules)
    2. Extracted (via LLM rule extraction tool)
    3. Custom (user-provided)
    """

    # Path to built-in rules
    BUILTIN_RULES = Path(__file__).parent.parent / "reasoning" / "rules" / "common_knowledge.dl"

    def __init__(
        self,
        custom_rules_path: Path | None = None,
        extracted_rules: list[str] | None = None,
    ):
        """
        Initialize Layer 1.

        Args:
            custom_rules_path: Optional path to custom Datalog rules file
            extracted_rules: Optional list of extracted rule strings (Datalog)
        """
        super().__init__(layer_number=1, layer_name="Common Knowledge")

        self.datalog = DatalogEngine()
        self.custom_rules_path = custom_rules_path
        self.extracted_rules = extracted_rules or []

        # In-memory rule storage for dynamic rules
        self._dynamic_rules: list[Rule] = []

        # Load default basic rules
        self._load_basic_rules()

    def _load_basic_rules(self) -> None:
        """Load basic built-in rules for common knowledge."""
        # These are simple rules that can be checked without Datalog
        # They serve as fallback and for quick checks
        # Note: The actual checking is done via Datalog rules, these serve as metadata
        basic_rules = [
            Rule(
                rule_id="ck_no_fabrication",
                name="No Fabrication",
                description="Agent should not claim things not supported by context",
                rule_type=RuleType.CONSTRAINT,
                layer=1,
                conditions=[],  # Checked via Datalog
                severity=Severity.WARNING,
                message_template="Ungrounded reference: {detail}",
            ),
            Rule(
                rule_id="ck_error_awareness",
                name="Error Awareness",
                description="Agent should acknowledge errors in context",
                rule_type=RuleType.CONSTRAINT,
                layer=1,
                conditions=[],  # Checked via Datalog
                severity=Severity.ERROR,
                message_template="Ignored error: {detail}",
            ),
            Rule(
                rule_id="ck_no_repeat_failure",
                name="No Repeated Failures",
                description="Agent should not repeat actions that already failed",
                rule_type=RuleType.CONSTRAINT,
                layer=1,
                conditions=[],  # Checked via Datalog
                severity=Severity.ERROR,
                message_template="Repeated failed action: {detail}",
            ),
        ]
        self._dynamic_rules.extend(basic_rules)

    def add_rule(self, rule: Rule) -> None:
        """
        Add a dynamic rule to this layer.

        Args:
            rule: Rule to add (can be from extraction or manual)
        """
        self._dynamic_rules.append(rule)

    def add_extracted_rule(self, datalog_rule: str) -> None:
        """
        Add an extracted Datalog rule string.

        Args:
            datalog_rule: Datalog rule code as string
        """
        self.extracted_rules.append(datalog_rule)

    def clear_extracted_rules(self) -> None:
        """Clear all extracted rules."""
        self.extracted_rules.clear()

    def _get_rules_program(self) -> str:
        """
        Combine all rules into a single Datalog program.

        Returns:
            Complete Datalog program string
        """
        # Start with built-in rules file content
        if self.BUILTIN_RULES.exists():
            with open(self.BUILTIN_RULES) as f:
                program = f.read()
        else:
            program = ""

        # Append custom rules file if provided
        if self.custom_rules_path and self.custom_rules_path.exists():
            with open(self.custom_rules_path) as f:
                program += "\n\n// Custom Rules\n" + f.read()

        # Append extracted rules
        if self.extracted_rules:
            program += "\n\n// Extracted Rules\n"
            for rule in self.extracted_rules:
                program += rule + "\n"

        return program

    def _extract_facts(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> ExtractedFacts:
        """
        Extract facts from the request for Datalog reasoning.

        This is a simple extraction - can be enhanced with LLM extraction.

        Args:
            request: The verification request
            context: Additional context from extractors

        Returns:
            ExtractedFacts for Datalog input
        """
        facts = ExtractedFacts()

        # Use context if provided by extractors, otherwise do basic extraction
        if "extracted_facts" in context:
            ef = context["extracted_facts"]
            facts.context_elements = ef.get("context_elements", [])
            facts.context_errors = ef.get("context_errors", [])
            facts.context_warnings = ef.get("context_warnings", [])
            facts.action_history = ef.get("action_history", [])
            facts.output_references = ef.get("output_references", [])
            facts.output_claims = ef.get("output_claims", [])
            facts.output_action = ef.get("output_action", "")
            facts.output_target = ef.get("output_target", "")
            facts.acknowledges_error = ef.get("acknowledges_error", False)
            facts.acknowledges_history = ef.get("acknowledges_history", False)
        else:
            # Basic extraction from raw text
            facts.request_task = request.prompt[:200] if request.prompt else ""

            # Simple error detection in output
            output_lower = request.llm_output.lower()
            facts.acknowledges_error = any(
                kw in output_lower
                for kw in ["error", "failed", "issue", "problem", "cannot", "unable"]
            )
            facts.acknowledges_history = any(
                kw in output_lower
                for kw in ["previous", "before", "already", "tried", "again"]
            )

        facts.request_task = facts.request_task or request.prompt[:200]
        return facts

    def _populate_datalog_facts(self, case_id: str, facts: ExtractedFacts) -> None:
        """Populate the Datalog engine with extracted facts."""
        self.datalog.clear_facts()

        # Context elements
        for elem in facts.context_elements:
            self.datalog.add_fact("context_element", case_id, elem.lower().strip())

        # Context errors
        for err in facts.context_errors:
            self.datalog.add_fact("context_error", case_id, err)

        # Context warnings
        for warn in facts.context_warnings:
            self.datalog.add_fact("context_warning", case_id, warn)

        # Action history
        for action, outcome in facts.action_history:
            self.datalog.add_fact("action_history", case_id, action.lower().strip(), outcome)

        # Request task
        self.datalog.add_fact("request_task", case_id, facts.request_task)

        # Output references
        for ref in facts.output_references:
            self.datalog.add_fact("output_reference", case_id, ref.lower().strip())

        # Output claims
        for claim in facts.output_claims:
            self.datalog.add_fact("output_claim", case_id, claim)

        # Output action and target
        if facts.output_action:
            self.datalog.add_fact("output_action", case_id, facts.output_action.lower().strip())
        if facts.output_target:
            self.datalog.add_fact("output_target", case_id, facts.output_target.lower().strip())

        # Acknowledgments
        if facts.acknowledges_error:
            self.datalog.add_fact("acknowledges_error", case_id)
        if facts.acknowledges_history:
            self.datalog.add_fact("acknowledges_history", case_id)

    def _parse_datalog_violations(
        self,
        result: DatalogResult,
        case_id: str,
    ) -> list[tuple[str, str, str]]:
        """
        Parse violations from Datalog output.

        Returns:
            List of (violation_type, detail, severity) tuples
        """
        violations = []
        for row in result.get_relation("output_violation"):
            if len(row) >= 4 and row[0] == case_id:
                violations.append((row[1], row[2], row[3]))  # type, detail, severity
        return violations

    def check(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> LayerResult:
        """
        Check common knowledge rules against the request.

        Args:
            request: The verification request
            context: Accumulated context from extractors

        Returns:
            LayerResult with violations and reasoning
        """
        result = LayerResult(layer=self.layer_number)
        case_id = request.request_id

        # Step 1: Extract facts
        facts = self._extract_facts(request, context)
        result.add_reasoning(self.create_reasoning_step(
            step_type="fact_extraction",
            description="Extracted facts from request and context",
            inputs={"request_id": case_id},
            outputs={
                "context_elements": len(facts.context_elements),
                "context_errors": len(facts.context_errors),
                "output_references": len(facts.output_references),
            },
        ))

        # Step 2: Run Datalog rules
        self._populate_datalog_facts(case_id, facts)

        # Get the combined rules program
        program = self._get_rules_program()

        if program.strip():
            datalog_result = self.datalog.run_inline(
                program,
                output_relations=["output_consistency", "output_violation", "output_violation_count"]
            )

            if datalog_result.success:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description="Applied Datalog rules for consistency checking",
                    inputs={"program_lines": len(program.split("\n"))},
                    outputs={"success": True},
                ))

                # Parse violations
                dl_violations = self._parse_datalog_violations(datalog_result, case_id)
                for vtype, detail, severity in dl_violations:
                    result.add_violation(self.create_violation(
                        violation_type=vtype,
                        message=f"{vtype}: {detail}",
                        evidence={"detail": detail, "source": "datalog"},
                        severity=severity,
                        rule_id=f"ck_{vtype}",
                    ))

                # Store consistency info in metadata
                for row in datalog_result.get_relation("output_consistency"):
                    if len(row) >= 3 and row[0] == case_id:
                        result.metadata["is_consistent"] = row[1] == "true"
                        result.metadata["consistency_score"] = int(row[2])
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

        # Store extracted facts for downstream layers
        result.facts_extracted = {
            "context_elements": facts.context_elements,
            "context_errors": facts.context_errors,
            "output_references": facts.output_references,
            "acknowledges_error": facts.acknowledges_error,
        }

        return result

    def load_rules(self, deployment_id: str) -> list[Rule]:
        """
        Load rules for this layer.

        For Layer 1, rules are universal and don't vary by deployment.
        However, deployments can add custom rules.

        Args:
            deployment_id: The deployment identifier

        Returns:
            List of active rules
        """
        return list(self._dynamic_rules)
