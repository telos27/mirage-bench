"""Layer 2: Domain Best Practices - Domain-specific patterns and rules."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_layer import BaseLayer, LayerResult
from ..schemas.request import VerificationRequest
from ..schemas.result import Severity
from ..schemas.rules import Rule, RuleType
from ..reasoning.datalog_engine import DatalogEngine, DatalogResult
from ..rule_extraction.schemas import RuleDomain, NaturalRule, CompiledRule


@dataclass
class DomainConfig:
    """
    Configuration for a domain.

    Attributes:
        domain: The domain identifier
        name: Human-readable name
        description: Description of the domain
        rules_file: Path to built-in Datalog rules file
        active: Whether this domain is active
        extracted_rules: Dynamically extracted rules (Datalog strings)
        compiled_rules: Rules compiled from natural language
    """
    domain: str
    name: str
    description: str
    rules_file: Path | None = None
    active: bool = True
    extracted_rules: list[str] = field(default_factory=list)
    compiled_rules: list[CompiledRule] = field(default_factory=list)


# Built-in domain configurations
DOMAIN_CONFIGS = {
    RuleDomain.CODING.value: DomainConfig(
        domain=RuleDomain.CODING.value,
        name="Coding",
        description="Best practices for code generation, review, and assistance",
        rules_file=Path(__file__).parent.parent / "reasoning" / "rules" / "domain_coding.dl",
    ),
    RuleDomain.CUSTOMER_SERVICE.value: DomainConfig(
        domain=RuleDomain.CUSTOMER_SERVICE.value,
        name="Customer Service",
        description="Best practices for customer support and service agents",
        rules_file=Path(__file__).parent.parent / "reasoning" / "rules" / "domain_customer_service.dl",
    ),
    RuleDomain.DATA_ANALYSIS.value: DomainConfig(
        domain=RuleDomain.DATA_ANALYSIS.value,
        name="Data Analysis",
        description="Best practices for data analysis and visualization",
        rules_file=Path(__file__).parent.parent / "reasoning" / "rules" / "domain_data_analysis.dl",
    ),
    RuleDomain.CONTENT_GENERATION.value: DomainConfig(
        domain=RuleDomain.CONTENT_GENERATION.value,
        name="Content Generation",
        description="Best practices for content creation and writing",
        rules_file=Path(__file__).parent.parent / "reasoning" / "rules" / "domain_content_generation.dl",
    ),
    RuleDomain.GENERAL.value: DomainConfig(
        domain=RuleDomain.GENERAL.value,
        name="General",
        description="General agent best practices",
        rules_file=Path(__file__).parent.parent / "reasoning" / "rules" / "domain_general.dl",
    ),
}


class DomainBestPracticesLayer(BaseLayer):
    """
    Layer 2: Domain Best Practices.

    Checks domain-specific rules and patterns:
    - Coding: Security, style, error handling
    - Customer Service: Tone, escalation, policy compliance
    - Data Analysis: Accuracy, methodology, reproducibility
    - Content Generation: Accuracy, originality, appropriateness
    - General: Common agent behavior patterns

    Rules can be:
    1. Built-in (predefined Datalog rules per domain)
    2. Extracted (via LLM rule extraction tool)
    3. Custom (user-provided per deployment)
    """

    def __init__(
        self,
        domains: list[str] | None = None,
        storage: Any = None,
    ):
        """
        Initialize Layer 2.

        Args:
            domains: List of domain names to activate (None = all domains)
            storage: Optional SQLiteStore for loading custom rules
        """
        super().__init__(layer_number=2, layer_name="Domain Best Practices")

        self.datalog = DatalogEngine()
        self.storage = storage

        # Initialize domain configurations
        self._domain_configs: dict[str, DomainConfig] = {}
        self._init_domains(domains)

        # In-memory rule storage for quick access
        self._dynamic_rules: list[Rule] = []

        # Load basic metadata rules
        self._load_basic_rules()

    def _init_domains(self, domains: list[str] | None) -> None:
        """Initialize domain configurations."""
        if domains is None:
            # Activate all built-in domains
            for domain_id, config in DOMAIN_CONFIGS.items():
                self._domain_configs[domain_id] = DomainConfig(
                    domain=config.domain,
                    name=config.name,
                    description=config.description,
                    rules_file=config.rules_file,
                    active=True,
                )
        else:
            # Activate only specified domains
            for domain_id in domains:
                if domain_id in DOMAIN_CONFIGS:
                    config = DOMAIN_CONFIGS[domain_id]
                    self._domain_configs[domain_id] = DomainConfig(
                        domain=config.domain,
                        name=config.name,
                        description=config.description,
                        rules_file=config.rules_file,
                        active=True,
                    )
                else:
                    # Custom domain (no built-in rules)
                    self._domain_configs[domain_id] = DomainConfig(
                        domain=domain_id,
                        name=domain_id.replace("_", " ").title(),
                        description=f"Custom domain: {domain_id}",
                        active=True,
                    )

    def _load_basic_rules(self) -> None:
        """Load basic rule metadata for all active domains."""
        # Coding domain rules
        if RuleDomain.CODING.value in self._domain_configs:
            self._dynamic_rules.extend([
                Rule(
                    rule_id="dp_coding_no_hardcoded_secrets",
                    name="No Hardcoded Secrets",
                    description="Code should not contain hardcoded passwords, API keys, or tokens",
                    rule_type=RuleType.PROHIBITION,
                    layer=2,
                    conditions=[],
                    severity=Severity.ERROR,
                    message_template="Hardcoded secret detected: {detail}",
                    tags=["coding", "security"],
                ),
                Rule(
                    rule_id="dp_coding_no_dangerous_functions",
                    name="No Dangerous Functions",
                    description="Avoid dangerous functions like eval(), exec() with user input",
                    rule_type=RuleType.PROHIBITION,
                    layer=2,
                    conditions=[],
                    severity=Severity.ERROR,
                    message_template="Dangerous function usage: {detail}",
                    tags=["coding", "security"],
                ),
                Rule(
                    rule_id="dp_coding_error_handling",
                    name="Proper Error Handling",
                    description="Code should include appropriate error handling",
                    rule_type=RuleType.REQUIREMENT,
                    layer=2,
                    conditions=[],
                    severity=Severity.WARNING,
                    message_template="Missing error handling: {detail}",
                    tags=["coding", "quality"],
                ),
            ])

        # Customer service domain rules
        if RuleDomain.CUSTOMER_SERVICE.value in self._domain_configs:
            self._dynamic_rules.extend([
                Rule(
                    rule_id="dp_cs_professional_tone",
                    name="Professional Tone",
                    description="Responses should maintain a professional and helpful tone",
                    rule_type=RuleType.REQUIREMENT,
                    layer=2,
                    conditions=[],
                    severity=Severity.WARNING,
                    message_template="Unprofessional tone detected: {detail}",
                    tags=["customer_service", "tone"],
                ),
                Rule(
                    rule_id="dp_cs_no_promises",
                    name="No Unauthorized Promises",
                    description="Should not make promises about refunds, exceptions, or policies without authorization",
                    rule_type=RuleType.PROHIBITION,
                    layer=2,
                    conditions=[],
                    severity=Severity.ERROR,
                    message_template="Unauthorized promise: {detail}",
                    tags=["customer_service", "policy"],
                ),
            ])

        # Data analysis domain rules
        if RuleDomain.DATA_ANALYSIS.value in self._domain_configs:
            self._dynamic_rules.extend([
                Rule(
                    rule_id="dp_data_source_citation",
                    name="Data Source Citation",
                    description="Analysis should cite data sources",
                    rule_type=RuleType.REQUIREMENT,
                    layer=2,
                    conditions=[],
                    severity=Severity.WARNING,
                    message_template="Missing data source citation: {detail}",
                    tags=["data_analysis", "methodology"],
                ),
                Rule(
                    rule_id="dp_data_no_fabrication",
                    name="No Data Fabrication",
                    description="Should not fabricate or invent data points",
                    rule_type=RuleType.PROHIBITION,
                    layer=2,
                    conditions=[],
                    severity=Severity.ERROR,
                    message_template="Potential data fabrication: {detail}",
                    tags=["data_analysis", "integrity"],
                ),
            ])

    def get_active_domains(self) -> list[str]:
        """Get list of active domain names."""
        return [d for d, c in self._domain_configs.items() if c.active]

    def activate_domain(self, domain: str) -> None:
        """
        Activate a domain.

        Args:
            domain: Domain identifier to activate
        """
        if domain in self._domain_configs:
            self._domain_configs[domain].active = True
        elif domain in DOMAIN_CONFIGS:
            config = DOMAIN_CONFIGS[domain]
            self._domain_configs[domain] = DomainConfig(
                domain=config.domain,
                name=config.name,
                description=config.description,
                rules_file=config.rules_file,
                active=True,
            )
        else:
            # Custom domain
            self._domain_configs[domain] = DomainConfig(
                domain=domain,
                name=domain.replace("_", " ").title(),
                description=f"Custom domain: {domain}",
                active=True,
            )

    def deactivate_domain(self, domain: str) -> None:
        """
        Deactivate a domain.

        Args:
            domain: Domain identifier to deactivate
        """
        if domain in self._domain_configs:
            self._domain_configs[domain].active = False

    def add_rule(self, rule: Rule) -> None:
        """
        Add a dynamic rule to this layer.

        Args:
            rule: Rule to add
        """
        self._dynamic_rules.append(rule)

    def add_extracted_rule(self, domain: str, datalog_rule: str) -> None:
        """
        Add an extracted Datalog rule string to a domain.

        Args:
            domain: Domain to add the rule to
            datalog_rule: Datalog rule code as string
        """
        if domain in self._domain_configs:
            self._domain_configs[domain].extracted_rules.append(datalog_rule)

    def add_compiled_rule(self, domain: str, compiled_rule: CompiledRule) -> None:
        """
        Add a compiled rule to a domain.

        Args:
            domain: Domain to add the rule to
            compiled_rule: The compiled rule
        """
        if domain in self._domain_configs:
            self._domain_configs[domain].compiled_rules.append(compiled_rule)

    def clear_extracted_rules(self, domain: str | None = None) -> None:
        """
        Clear extracted rules.

        Args:
            domain: Domain to clear (None = all domains)
        """
        if domain:
            if domain in self._domain_configs:
                self._domain_configs[domain].extracted_rules.clear()
                self._domain_configs[domain].compiled_rules.clear()
        else:
            for config in self._domain_configs.values():
                config.extracted_rules.clear()
                config.compiled_rules.clear()

    def _get_rules_program(self, domains: list[str] | None = None) -> str:
        """
        Combine all rules for specified domains into a single Datalog program.

        Args:
            domains: Domains to include (None = all active)

        Returns:
            Complete Datalog program string
        """
        program_parts = []

        target_domains = domains if domains else self.get_active_domains()

        for domain in target_domains:
            if domain not in self._domain_configs:
                continue

            config = self._domain_configs[domain]
            if not config.active:
                continue

            # Add built-in rules from file
            if config.rules_file and config.rules_file.exists():
                with open(config.rules_file) as f:
                    program_parts.append(f"// Domain: {config.name}\n{f.read()}")

            # Add extracted rules
            if config.extracted_rules:
                program_parts.append(f"\n// Extracted rules for {config.name}")
                for rule in config.extracted_rules:
                    program_parts.append(rule)

            # Add compiled rules
            if config.compiled_rules:
                program_parts.append(f"\n// Compiled rules for {config.name}")
                for compiled in config.compiled_rules:
                    if compiled.is_valid:
                        program_parts.append(compiled.datalog_code)

        return "\n\n".join(program_parts)

    def _detect_domains(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> list[str]:
        """
        Detect which domains apply to this request.

        Uses heuristics based on prompt content and context.

        Args:
            request: The verification request
            context: Accumulated context

        Returns:
            List of detected domain identifiers
        """
        detected = []
        prompt_lower = request.prompt.lower()
        output_lower = request.llm_output.lower()

        # Check for coding indicators
        coding_indicators = [
            "code", "function", "class", "variable", "python", "javascript",
            "java", "programming", "debug", "bug", "error", "syntax",
            "compile", "import", "def ", "async ", "return ", "```",
        ]
        if any(ind in prompt_lower or ind in output_lower for ind in coding_indicators):
            detected.append(RuleDomain.CODING.value)

        # Check for customer service indicators
        cs_indicators = [
            "customer", "support", "help me", "refund", "order", "shipping",
            "complaint", "issue with", "problem with", "service", "ticket",
        ]
        if any(ind in prompt_lower for ind in cs_indicators):
            detected.append(RuleDomain.CUSTOMER_SERVICE.value)

        # Check for data analysis indicators
        data_indicators = [
            "data", "analysis", "statistics", "chart", "graph", "plot",
            "dataset", "csv", "sql", "query", "aggregate", "average",
            "correlation", "regression", "visualization",
        ]
        if any(ind in prompt_lower or ind in output_lower for ind in data_indicators):
            detected.append(RuleDomain.DATA_ANALYSIS.value)

        # Check for content generation indicators
        content_indicators = [
            "write", "article", "blog", "story", "essay", "content",
            "create", "generate", "compose", "draft",
        ]
        if any(ind in prompt_lower for ind in content_indicators):
            detected.append(RuleDomain.CONTENT_GENERATION.value)

        # Always include general if no specific domains detected
        if not detected:
            detected.append(RuleDomain.GENERAL.value)

        # Filter to only active domains
        return [d for d in detected if d in self._domain_configs and self._domain_configs[d].active]

    def _populate_datalog_facts(
        self,
        case_id: str,
        request: VerificationRequest,
        context: dict[str, Any],
        detected_domains: list[str],
    ) -> None:
        """Populate the Datalog engine with facts for domain checking."""
        self.datalog.clear_facts()

        # Basic request info
        self.datalog.add_fact("request_id", case_id)

        # Detected domains
        for domain in detected_domains:
            self.datalog.add_fact("active_domain", case_id, domain)

        # Use extracted facts from context if available
        if "extracted_facts" in context:
            ef = context["extracted_facts"]

            # Output facts
            for ref in ef.get("output_references", []):
                self.datalog.add_fact("output_reference", case_id, ref.lower().strip())

            if ef.get("output_action"):
                self.datalog.add_fact("output_action", case_id, ef["output_action"].lower().strip())

            # Context elements
            for elem in ef.get("context_elements", []):
                self.datalog.add_fact("context_element", case_id, elem.lower().strip())

        # Code-specific facts for coding domain
        if RuleDomain.CODING.value in detected_domains:
            self._extract_code_facts(case_id, request)

        # Customer service facts
        if RuleDomain.CUSTOMER_SERVICE.value in detected_domains:
            self._extract_cs_facts(case_id, request)

    def _extract_code_facts(self, case_id: str, request: VerificationRequest) -> None:
        """Extract code-specific facts."""
        output = request.llm_output
        output_lower = output.lower()

        # Detect potentially dangerous patterns
        dangerous_patterns = [
            ("eval(", "eval_usage"),
            ("exec(", "exec_usage"),
            ("subprocess.call(", "subprocess_shell"),
            ("os.system(", "os_system"),
            ("shell=True", "shell_true"),
            ("pickle.load", "pickle_load"),
            ("__import__", "dynamic_import"),
        ]

        for pattern, fact_type in dangerous_patterns:
            if pattern in output:
                self.datalog.add_fact("code_pattern", case_id, fact_type, "dangerous")

        # Detect hardcoded secrets patterns
        secret_patterns = [
            'password = "', "password = '", 'api_key = "', "api_key = '",
            'secret = "', "secret = '", 'token = "', "token = '",
            "PASSWORD=", "API_KEY=", "SECRET=", "TOKEN=",
        ]

        for pattern in secret_patterns:
            if pattern in output:
                self.datalog.add_fact("code_pattern", case_id, "hardcoded_secret", "security")

        # Detect error handling
        error_handling_patterns = ["try:", "except ", "catch ", "raise ", "throw "]
        has_error_handling = any(p in output for p in error_handling_patterns)
        self.datalog.add_fact("code_has_error_handling", case_id, "true" if has_error_handling else "false")

        # Detect input validation
        validation_patterns = ["validate", "sanitize", "escape", "parameterized", "prepared"]
        has_validation = any(p in output_lower for p in validation_patterns)
        self.datalog.add_fact("code_has_validation", case_id, "true" if has_validation else "false")

    def _extract_cs_facts(self, case_id: str, request: VerificationRequest) -> None:
        """Extract customer service facts."""
        output_lower = request.llm_output.lower()

        # Detect promise-like language
        promise_patterns = [
            "i promise", "i guarantee", "we will definitely", "absolutely will",
            "100% refund", "full refund guaranteed", "i'll make sure",
        ]

        for pattern in promise_patterns:
            if pattern in output_lower:
                self.datalog.add_fact("cs_pattern", case_id, "unauthorized_promise")

        # Detect tone issues
        negative_tone_patterns = [
            "that's not my problem", "you should have", "it's your fault",
            "obviously", "clearly you", "as i already said",
        ]

        for pattern in negative_tone_patterns:
            if pattern in output_lower:
                self.datalog.add_fact("cs_pattern", case_id, "negative_tone")

        # Detect empathy
        empathy_patterns = [
            "i understand", "i'm sorry", "i apologize", "thank you for",
            "appreciate your patience", "happy to help",
        ]
        has_empathy = any(p in output_lower for p in empathy_patterns)
        self.datalog.add_fact("cs_has_empathy", case_id, "true" if has_empathy else "false")

    def _parse_datalog_violations(
        self,
        result: DatalogResult,
        case_id: str,
    ) -> list[tuple[str, str, str, str]]:
        """
        Parse violations from Datalog output.

        Returns:
            List of (domain, violation_type, detail, severity) tuples
        """
        violations = []
        for row in result.get_relation("output_domain_violation"):
            if len(row) >= 5 and row[0] == case_id:
                violations.append((row[1], row[2], row[3], row[4]))  # domain, type, detail, severity
        return violations

    def check(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> LayerResult:
        """
        Check domain best practices against the request.

        Args:
            request: The verification request
            context: Accumulated context from Layer 1 and extractors

        Returns:
            LayerResult with violations and reasoning
        """
        result = LayerResult(layer=self.layer_number)
        case_id = request.request_id

        # Step 1: Detect applicable domains
        detected_domains = self._detect_domains(request, context)
        result.add_reasoning(self.create_reasoning_step(
            step_type="domain_detection",
            description="Detected applicable domains",
            inputs={"request_id": case_id},
            outputs={"domains": detected_domains},
        ))

        if not detected_domains:
            result.add_reasoning(self.create_reasoning_step(
                step_type="skip",
                description="No active domains detected, skipping domain checks",
                inputs={},
                outputs={"skipped": True},
            ))
            return result

        # Step 2: Populate facts
        self._populate_datalog_facts(case_id, request, context, detected_domains)

        # Step 3: Run domain rules
        program = self._get_rules_program(detected_domains)

        if program.strip():
            datalog_result = self.datalog.run_inline(
                program,
                output_relations=["output_domain_violation"]
            )

            if datalog_result.success:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description="Applied domain-specific Datalog rules",
                    inputs={
                        "domains": detected_domains,
                        "program_lines": len(program.split("\n")),
                    },
                    outputs={"success": True},
                ))

                # Parse and add violations
                dl_violations = self._parse_datalog_violations(datalog_result, case_id)
                for domain, vtype, detail, severity in dl_violations:
                    result.add_violation(self.create_violation(
                        violation_type=vtype,
                        message=f"[{domain}] {vtype}: {detail}",
                        evidence={"domain": domain, "detail": detail, "source": "datalog"},
                        severity=severity,
                        rule_id=f"dp_{domain}_{vtype}",
                    ))
            else:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description=f"Datalog execution failed: {datalog_result.error}",
                    inputs={"domains": detected_domains},
                    outputs={"success": False, "error": datalog_result.error},
                ))
        else:
            result.add_reasoning(self.create_reasoning_step(
                step_type="rule_application",
                description="No Datalog rules available for detected domains",
                inputs={"domains": detected_domains},
                outputs={"skipped": True},
            ))

        # Step 4: Apply heuristic checks (fallback when no Datalog rules)
        self._apply_heuristic_checks(case_id, request, detected_domains, result)

        # Store detected domains for downstream layers
        result.facts_extracted = {
            "detected_domains": detected_domains,
        }
        result.metadata["domains_checked"] = detected_domains

        return result

    def _apply_heuristic_checks(
        self,
        case_id: str,
        request: VerificationRequest,
        domains: list[str],
        result: LayerResult,
    ) -> None:
        """Apply heuristic checks as fallback/supplement to Datalog rules."""
        output = request.llm_output

        # Coding domain heuristics
        if RuleDomain.CODING.value in domains:
            # Check for eval with string concatenation (high risk)
            if "eval(" in output and ("+" in output or "format" in output or "f'" in output or 'f"' in output):
                result.add_violation(self.create_violation(
                    violation_type="dangerous_eval",
                    message="Potential code injection: eval() with string concatenation",
                    evidence={
                        "pattern": "eval with string building",
                        "domain": "coding",
                    },
                    severity="error",
                    rule_id="dp_coding_no_dangerous_functions",
                    suggestion="Use ast.literal_eval() for safe evaluation or avoid eval entirely",
                ))

            # Check for SQL injection vulnerability patterns
            sql_injection_patterns = [
                'execute("SELECT', "execute('SELECT",
                'cursor.execute(f"', "cursor.execute(f'",
                '% query', 'format(query',
            ]
            for pattern in sql_injection_patterns:
                if pattern in output:
                    result.add_violation(self.create_violation(
                        violation_type="sql_injection_risk",
                        message="Potential SQL injection vulnerability detected",
                        evidence={
                            "pattern": pattern,
                            "domain": "coding",
                        },
                        severity="error",
                        rule_id="dp_coding_sql_injection",
                        suggestion="Use parameterized queries instead of string formatting",
                    ))
                    break

        # Customer service heuristics
        if RuleDomain.CUSTOMER_SERVICE.value in domains:
            output_lower = output.lower()

            # Check for unprofessional language
            unprofessional = [
                "whatever", "i don't care", "not my job", "figure it out",
            ]
            for phrase in unprofessional:
                if phrase in output_lower:
                    result.add_violation(self.create_violation(
                        violation_type="unprofessional_language",
                        message=f"Unprofessional language detected: '{phrase}'",
                        evidence={
                            "phrase": phrase,
                            "domain": "customer_service",
                        },
                        severity="warning",
                        rule_id="dp_cs_professional_tone",
                    ))
                    break

    def load_rules(self, deployment_id: str) -> list[Rule]:
        """
        Load rules for this layer from storage.

        Args:
            deployment_id: The deployment identifier

        Returns:
            List of active rules
        """
        rules = list(self._dynamic_rules)

        # Load from storage if available
        if self.storage:
            stored_rules = self.storage.get_rules_for_layer(
                layer=2,
                deployment_id=deployment_id,
                enabled_only=True,
            )
            for model in stored_rules:
                rules.append(model.to_rule())

        return rules
