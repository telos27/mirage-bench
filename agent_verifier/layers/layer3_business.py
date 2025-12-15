"""Layer 3: Business Policies - Organization-level rules and compliance."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_layer import BaseLayer, LayerResult
from ..schemas.request import VerificationRequest
from ..schemas.result import Severity
from ..schemas.rules import Rule, RuleType, PolicySpec
from ..reasoning.datalog_engine import DatalogEngine, DatalogResult


@dataclass
class PolicyConfig:
    """
    Configuration for a business policy.

    This is the runtime representation loaded from storage.
    """
    policy_id: str
    name: str
    description: str
    policy_type: str  # constraint, requirement, prohibition
    priority: int = 0
    enabled: bool = True

    # Policy constraints (compiled to Datalog facts)
    forbidden_words: list[str] = field(default_factory=list)
    required_words: list[str] = field(default_factory=list)
    required_format: str | None = None
    max_length: int | None = None
    min_length: int | None = None
    allowed_languages: list[str] = field(default_factory=list)
    forbidden_topics: list[str] = field(default_factory=list)
    required_disclaimers: list[str] = field(default_factory=list)
    forbids_external_links: bool = False
    forbids_pii: bool = False
    forbids_code_execution: bool = False

    # Custom Datalog rules for complex policies
    custom_datalog: str = ""


class BusinessPoliciesLayer(BaseLayer):
    """
    Layer 3: Business Policies.

    Checks organization-level policies and compliance requirements:
    - Content constraints (forbidden words, required disclaimers)
    - Format requirements (length limits, structure)
    - Behavioral restrictions (no external links, no PII)
    - Compliance requirements

    Policies are loaded from storage (SQLiteStore) for each deployment.
    All policy checking is done via Soufflé Datalog for transparency.
    """

    # Path to built-in policy rules
    BUILTIN_RULES = Path(__file__).parent.parent / "reasoning" / "rules" / "business_policy.dl"

    def __init__(
        self,
        storage: Any = None,
        policies: list[PolicyConfig] | None = None,
    ):
        """
        Initialize Layer 3.

        Args:
            storage: Optional SQLiteStore for loading policies
            policies: Optional pre-configured policies (for testing)
        """
        super().__init__(layer_number=3, layer_name="Business Policies")

        self.datalog = DatalogEngine()
        self.storage = storage

        # Policy configurations (deployment_id -> list of policies)
        self._policies: dict[str, list[PolicyConfig]] = {}

        # Pre-configured policies (not from storage)
        if policies:
            for policy in policies:
                self.add_policy("default", policy)

        # In-memory rule storage
        self._dynamic_rules: list[Rule] = []

        # Additional Datalog rules (from rule extraction)
        self._extracted_rules: list[str] = []

        self._load_basic_rules()

    def _load_basic_rules(self) -> None:
        """Load basic rule metadata for business policies."""
        basic_rules = [
            Rule(
                rule_id="bp_content_compliance",
                name="Content Compliance",
                description="Output must comply with content policies",
                rule_type=RuleType.CONSTRAINT,
                layer=3,
                conditions=[],
                severity=Severity.ERROR,
                message_template="Content policy violation: {detail}",
                tags=["business", "compliance"],
            ),
            Rule(
                rule_id="bp_format_compliance",
                name="Format Compliance",
                description="Output must comply with format requirements",
                rule_type=RuleType.REQUIREMENT,
                layer=3,
                conditions=[],
                severity=Severity.WARNING,
                message_template="Format policy violation: {detail}",
                tags=["business", "format"],
            ),
            Rule(
                rule_id="bp_pii_protection",
                name="PII Protection",
                description="Output must not contain unauthorized PII",
                rule_type=RuleType.PROHIBITION,
                layer=3,
                conditions=[],
                severity=Severity.ERROR,
                message_template="PII detected in output: {detail}",
                tags=["business", "privacy"],
            ),
        ]
        self._dynamic_rules.extend(basic_rules)

    def add_policy(self, deployment_id: str, policy: PolicyConfig) -> None:
        """
        Add a policy for a deployment.

        Args:
            deployment_id: The deployment identifier
            policy: The policy configuration
        """
        if deployment_id not in self._policies:
            self._policies[deployment_id] = []
        self._policies[deployment_id].append(policy)

    def remove_policy(self, deployment_id: str, policy_id: str) -> bool:
        """
        Remove a policy.

        Args:
            deployment_id: The deployment identifier
            policy_id: The policy to remove

        Returns:
            True if removed, False if not found
        """
        if deployment_id not in self._policies:
            return False

        original_len = len(self._policies[deployment_id])
        self._policies[deployment_id] = [
            p for p in self._policies[deployment_id]
            if p.policy_id != policy_id
        ]
        return len(self._policies[deployment_id]) < original_len

    def get_policies(self, deployment_id: str) -> list[PolicyConfig]:
        """
        Get all policies for a deployment.

        Args:
            deployment_id: The deployment identifier

        Returns:
            List of PolicyConfig objects
        """
        return self._policies.get(deployment_id, [])

    def load_policies_from_storage(self, deployment_id: str) -> list[PolicyConfig]:
        """
        Load policies from storage for a deployment.

        Args:
            deployment_id: The deployment identifier

        Returns:
            List of loaded PolicyConfig objects
        """
        if not self.storage:
            return []

        loaded = []
        policy_models = self.storage.get_policies_for_deployment(
            deployment_id=deployment_id,
            enabled_only=True,
        )

        for model in policy_models:
            policy_dict = model.to_dict()
            spec = json.loads(model.policy_spec) if isinstance(model.policy_spec, str) else model.policy_spec

            config = PolicyConfig(
                policy_id=policy_dict["policy_id"],
                name=policy_dict["name"],
                description=policy_dict.get("description", ""),
                policy_type=policy_dict["policy_type"],
                priority=policy_dict.get("priority", 0),
                enabled=policy_dict.get("enabled", True),
                forbidden_words=spec.get("forbidden_words", []),
                required_words=spec.get("required_words", []),
                required_format=spec.get("required_format"),
                max_length=spec.get("max_length"),
                min_length=spec.get("min_length"),
                allowed_languages=spec.get("allowed_languages", []),
                forbidden_topics=spec.get("forbidden_topics", []),
                required_disclaimers=spec.get("required_disclaimers", []),
                forbids_external_links=spec.get("forbids_external_links", False),
                forbids_pii=spec.get("forbids_pii", False),
                forbids_code_execution=spec.get("forbids_code_execution", False),
                custom_datalog=spec.get("custom_datalog", ""),
            )
            loaded.append(config)
            self.add_policy(deployment_id, config)

        return loaded

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
        """
        Combine all rules into a single Datalog program.

        Returns:
            Complete Datalog program string
        """
        program = ""

        # Start with built-in rules
        if self.BUILTIN_RULES.exists():
            with open(self.BUILTIN_RULES) as f:
                program = f.read()

        # Add extracted rules
        if self._extracted_rules:
            program += "\n\n// Extracted Rules\n"
            for rule in self._extracted_rules:
                program += rule + "\n"

        # Add custom Datalog from policies
        for policies in self._policies.values():
            for policy in policies:
                if policy.custom_datalog:
                    program += f"\n\n// Custom rules for policy: {policy.policy_id}\n"
                    program += policy.custom_datalog

        return program

    def _extract_output_facts(
        self,
        case_id: str,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract facts from output for Datalog reasoning.

        Args:
            case_id: The case identifier
            request: The verification request
            context: Additional context

        Returns:
            Dictionary of extracted facts
        """
        output = request.llm_output
        output_lower = output.lower()

        facts = {
            "words": set(),
            "format": "text",
            "length": len(output),
            "language": "english",  # Default
            "topics": set(),
            "external_links": [],
            "pii_types": [],
            "code_types": [],
            "disclaimers": set(),
        }

        # Extract words (normalize to lowercase)
        words = set(re.findall(r'\b\w+\b', output_lower))
        facts["words"] = words

        # Detect format
        if output.strip().startswith("{") and output.strip().endswith("}"):
            facts["format"] = "json"
        elif "```" in output:
            facts["format"] = "code"
        elif output.count("#") > 2 or output.count("- ") > 3:
            facts["format"] = "markdown"

        # Detect external links
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        facts["external_links"] = re.findall(url_pattern, output)

        # Detect PII patterns
        pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, output):
                facts["pii_types"].append(pii_type)

        # Detect code execution patterns
        code_patterns = {
            "shell": ["os.system(", "subprocess.", "shell=True", "exec(", "eval("],
            "sql": ["cursor.execute(", "SELECT ", "INSERT ", "UPDATE ", "DELETE "],
            "script": ["<script", "javascript:", "onclick="],
        }
        for code_type, patterns in code_patterns.items():
            if any(p in output for p in patterns):
                facts["code_types"].append(code_type)

        # Detect disclaimers
        disclaimer_patterns = {
            "ai_generated": ["generated by ai", "ai-generated", "this is ai"],
            "not_advice": ["not financial advice", "not legal advice", "not medical advice"],
            "opinion": ["this is my opinion", "personal opinion", "just my view"],
            "liability": ["no liability", "not responsible", "use at your own risk"],
        }
        for disc_type, patterns in disclaimer_patterns.items():
            if any(p in output_lower for p in patterns):
                facts["disclaimers"].add(disc_type)

        # Use context if available
        if "extracted_facts" in context:
            ef = context["extracted_facts"]
            if "output_format" in ef:
                facts["format"] = ef["output_format"]
            if "output_language" in ef:
                facts["language"] = ef["output_language"]
            if "output_topics" in ef:
                facts["topics"] = set(ef["output_topics"])

        return facts

    def _populate_datalog_facts(
        self,
        case_id: str,
        request: VerificationRequest,
        context: dict[str, Any],
        policies: list[PolicyConfig],
    ) -> None:
        """Populate the Datalog engine with facts for policy checking."""
        self.datalog.clear_facts()

        # Extract output facts
        facts = self._extract_output_facts(case_id, request, context)

        # Basic request info
        self.datalog.add_fact("request_id", case_id)
        self.datalog.add_fact("deployment_id", case_id, request.deployment_id)

        # Active policies
        for policy in policies:
            if policy.enabled:
                self.datalog.add_fact(
                    "active_policy",
                    case_id,
                    policy.policy_id,
                    policy.policy_type,
                    policy.priority,
                )

                # Policy constraints
                for word in policy.forbidden_words:
                    self.datalog.add_fact("policy_forbids_word", policy.policy_id, word.lower())

                for word in policy.required_words:
                    self.datalog.add_fact("policy_requires_word", policy.policy_id, word.lower())

                if policy.required_format:
                    self.datalog.add_fact("policy_requires_format", policy.policy_id, policy.required_format)

                if policy.max_length is not None:
                    self.datalog.add_fact("policy_max_length", policy.policy_id, policy.max_length)

                if policy.min_length is not None:
                    self.datalog.add_fact("policy_min_length", policy.policy_id, policy.min_length)

                for lang in policy.allowed_languages:
                    self.datalog.add_fact("policy_allowed_language", policy.policy_id, lang.lower())

                for topic in policy.forbidden_topics:
                    self.datalog.add_fact("policy_forbids_topic", policy.policy_id, topic.lower())

                for disc in policy.required_disclaimers:
                    self.datalog.add_fact("policy_requires_disclaimer", policy.policy_id, disc.lower())

                if policy.forbids_external_links:
                    self.datalog.add_fact("policy_forbids_external_links", policy.policy_id)

                if policy.forbids_pii:
                    self.datalog.add_fact("policy_forbids_pii", policy.policy_id)

                if policy.forbids_code_execution:
                    self.datalog.add_fact("policy_forbids_code_execution", policy.policy_id)

        # Output facts
        for word in facts["words"]:
            self.datalog.add_fact("output_contains_word", case_id, word)

        self.datalog.add_fact("output_format", case_id, facts["format"])
        self.datalog.add_fact("output_length", case_id, facts["length"])
        self.datalog.add_fact("output_language", case_id, facts["language"])

        for topic in facts["topics"]:
            self.datalog.add_fact("output_topic", case_id, topic.lower())

        for url in facts["external_links"]:
            self.datalog.add_fact("output_has_external_link", case_id, url)

        for pii_type in facts["pii_types"]:
            self.datalog.add_fact("output_has_pii", case_id, pii_type)

        for code_type in facts["code_types"]:
            self.datalog.add_fact("output_has_code_execution", case_id, code_type)

        for disc in facts["disclaimers"]:
            self.datalog.add_fact("output_has_disclaimer", case_id, disc)

    def _parse_datalog_violations(
        self,
        result: DatalogResult,
        case_id: str,
    ) -> list[tuple[str, str, str, str]]:
        """
        Parse violations from Datalog output.

        Returns:
            List of (policy_id, violation_type, detail, severity) tuples
        """
        violations = []
        for row in result.get_relation("output_policy_violation"):
            if len(row) >= 5 and row[0] == case_id:
                violations.append((row[1], row[2], row[3], row[4]))  # policy_id, type, detail, severity
        return violations

    def _parse_policy_status(
        self,
        result: DatalogResult,
        case_id: str,
    ) -> dict[str, tuple[str, int]]:
        """
        Parse policy status from Datalog output.

        Returns:
            Dict of policy_id -> (status, violation_count)
        """
        status = {}
        for row in result.get_relation("output_policy_status"):
            if len(row) >= 4 and row[0] == case_id:
                status[row[1]] = (row[2], int(row[3]))  # policy_id -> (status, count)
        return status

    def check(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> LayerResult:
        """
        Check business policies against the request.

        Args:
            request: The verification request
            context: Accumulated context from previous layers

        Returns:
            LayerResult with violations and reasoning
        """
        result = LayerResult(layer=self.layer_number)
        case_id = request.request_id
        deployment_id = request.deployment_id

        # Step 1: Get applicable policies
        policies = self.get_policies(deployment_id)

        # Also try loading from storage if not already loaded
        if not policies and self.storage:
            policies = self.load_policies_from_storage(deployment_id)

        # Try "default" deployment if specific one has no policies
        if not policies and deployment_id != "default":
            policies = self.get_policies("default")

        result.add_reasoning(self.create_reasoning_step(
            step_type="policy_lookup",
            description="Retrieved applicable policies for deployment",
            inputs={"deployment_id": deployment_id},
            outputs={"policy_count": len(policies), "policy_ids": [p.policy_id for p in policies]},
        ))

        if not policies:
            result.add_reasoning(self.create_reasoning_step(
                step_type="skip",
                description="No policies configured for this deployment",
                inputs={"deployment_id": deployment_id},
                outputs={"skipped": True},
            ))
            return result

        # Step 2: Populate facts
        self._populate_datalog_facts(case_id, request, context, policies)

        # Step 3: Run Datalog rules
        program = self._get_rules_program()

        if program.strip():
            datalog_result = self.datalog.run_inline(
                program,
                output_relations=[
                    "output_compliance",
                    "output_policy_violation",
                    "output_policy_status",
                    "output_violation_count",
                ]
            )

            if datalog_result.success:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description="Applied business policy Datalog rules",
                    inputs={"program_lines": len(program.split("\n"))},
                    outputs={"success": True},
                ))

                # Parse violations
                dl_violations = self._parse_datalog_violations(datalog_result, case_id)
                for policy_id, vtype, detail, severity in dl_violations:
                    result.add_violation(self.create_violation(
                        violation_type=vtype,
                        message=f"[Policy: {policy_id}] {vtype}: {detail}",
                        evidence={
                            "policy_id": policy_id,
                            "detail": detail,
                            "source": "datalog",
                        },
                        severity=severity,
                        rule_id=f"bp_{policy_id}_{vtype}",
                    ))

                # Store compliance info in metadata
                for row in datalog_result.get_relation("output_compliance"):
                    if len(row) >= 3 and row[0] == case_id:
                        result.metadata["is_compliant"] = row[1] == "true"
                        result.metadata["compliance_score"] = int(row[2])

                # Store policy status
                result.metadata["policy_status"] = self._parse_policy_status(datalog_result, case_id)
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

        # Store policy info for downstream layers
        result.facts_extracted = {
            "policies_checked": [p.policy_id for p in policies],
            "is_compliant": result.metadata.get("is_compliant", True),
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
        rules = list(self._dynamic_rules)

        # Load from storage if available
        if self.storage:
            stored_rules = self.storage.get_rules_for_layer(
                layer=3,
                deployment_id=deployment_id,
                enabled_only=True,
            )
            for model in stored_rules:
                spec = json.loads(model.rule_spec)
                rules.append(Rule.from_dict(spec))

        return rules


# Convenience functions for common policy configurations

def create_content_policy(
    policy_id: str,
    forbidden_words: list[str] | None = None,
    required_words: list[str] | None = None,
    max_length: int | None = None,
    min_length: int | None = None,
) -> PolicyConfig:
    """
    Create a content-focused policy.

    Args:
        policy_id: Unique identifier
        forbidden_words: Words that must not appear
        required_words: Words that must appear
        max_length: Maximum output length
        min_length: Minimum output length

    Returns:
        PolicyConfig for content policy
    """
    return PolicyConfig(
        policy_id=policy_id,
        name="Content Policy",
        description="Content restrictions and requirements",
        policy_type="constraint",
        forbidden_words=forbidden_words or [],
        required_words=required_words or [],
        max_length=max_length,
        min_length=min_length,
    )


def create_privacy_policy(
    policy_id: str,
    forbids_pii: bool = True,
    forbids_external_links: bool = False,
) -> PolicyConfig:
    """
    Create a privacy-focused policy.

    Args:
        policy_id: Unique identifier
        forbids_pii: Whether to forbid PII
        forbids_external_links: Whether to forbid external links

    Returns:
        PolicyConfig for privacy policy
    """
    return PolicyConfig(
        policy_id=policy_id,
        name="Privacy Policy",
        description="Privacy and data protection requirements",
        policy_type="prohibition",
        forbids_pii=forbids_pii,
        forbids_external_links=forbids_external_links,
    )


def create_compliance_policy(
    policy_id: str,
    required_disclaimers: list[str] | None = None,
    allowed_languages: list[str] | None = None,
    required_format: str | None = None,
) -> PolicyConfig:
    """
    Create a compliance-focused policy.

    Args:
        policy_id: Unique identifier
        required_disclaimers: Required disclaimer types
        allowed_languages: Allowed output languages
        required_format: Required output format

    Returns:
        PolicyConfig for compliance policy
    """
    return PolicyConfig(
        policy_id=policy_id,
        name="Compliance Policy",
        description="Regulatory and compliance requirements",
        policy_type="requirement",
        required_disclaimers=required_disclaimers or [],
        allowed_languages=allowed_languages or [],
        required_format=required_format,
    )
