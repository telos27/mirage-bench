"""
Datalog compiler for converting natural language rules to Soufflé.

Provides both:
1. Template-based compilation for common patterns
2. LLM-based compilation for complex rules
"""

import logging
import re
from typing import Any, Optional

from .schemas import (
    NaturalRule,
    RuleSeverity,
    CompiledRule,
    CompilationResult,
)


# Common Datalog patterns for rule compilation
DATALOG_TEMPLATES = {
    # Pattern: If X contains Y, violation
    "contains_pattern": '''
// Rule: {rule_id}
// {description}
.decl {fact_name}(case_id: symbol, value: symbol)
.input {fact_name}

violation(id, "{rule_id}", "{message}") :-
    {fact_name}(id, val),
    contains(val, "{pattern}").
''',

    # Pattern: If X is missing when Y exists, violation
    "missing_pattern": '''
// Rule: {rule_id}
// {description}
.decl {required_fact}(case_id: symbol)
.decl {context_fact}(case_id: symbol)
.input {required_fact}
.input {context_fact}

violation(id, "{rule_id}", "{message}") :-
    {context_fact}(id),
    !{required_fact}(id).
''',

    # Pattern: If X matches regex Y, violation
    "regex_pattern": '''
// Rule: {rule_id}
// {description}
.decl {fact_name}(case_id: symbol, value: symbol)
.input {fact_name}

violation(id, "{rule_id}", "{message}") :-
    {fact_name}(id, val),
    match("{regex}", val).
''',

    # Pattern: If action X references target not in context Y, violation
    "ungrounded_reference": '''
// Rule: {rule_id}
// {description}
.decl action_target(case_id: symbol, target: symbol)
.decl context_element(case_id: symbol, element: symbol)
.input action_target
.input context_element

violation(id, "{rule_id}", "{message}") :-
    action_target(id, target),
    !context_element(id, target).
''',

    # Pattern: If error/warning present and action ignores it, violation
    "ignored_signal": '''
// Rule: {rule_id}
// {description}
.decl {signal_fact}(case_id: symbol, message: symbol)
.decl action_addresses_{signal_type}(case_id: symbol)
.input {signal_fact}
.input action_addresses_{signal_type}

violation(id, "{rule_id}", "{message}") :-
    {signal_fact}(id, _),
    !action_addresses_{signal_type}(id).
''',

    # Pattern: Repeated action detection
    "repeated_action": '''
// Rule: {rule_id}
// {description}
.decl action_type(case_id: symbol, action: symbol)
.decl previous_action(case_id: symbol, action: symbol)
.decl action_failed(case_id: symbol, action: symbol)
.input action_type
.input previous_action
.input action_failed

violation(id, "{rule_id}", "{message}") :-
    action_type(id, action),
    previous_action(id, action),
    action_failed(id, action).
''',

    # Generic violation template
    "generic": '''
// Rule: {rule_id}
// {description}
{input_declarations}

violation(id, "{rule_id}", "{message}") :-
    {body}.
''',
}


# Mapping from natural language patterns to Datalog templates
PATTERN_MAPPINGS = [
    # Security patterns
    (r"hardcoded\s+(secret|password|key|credential)", "contains_pattern", {
        "fact_name": "code_content",
        "pattern_field": "pattern",
    }),
    (r"user\s+input.*eval|eval.*user\s+input", "contains_pattern", {
        "fact_name": "code_content",
        "pattern_field": "pattern",
    }),
    # Missing validation patterns
    (r"missing\s+(validation|sanitization|check)", "missing_pattern", {
        "required_fact": "has_validation",
        "context_fact": "has_user_input",
    }),
    # Reference patterns
    (r"reference.*not\s+(in|present|found|available)", "ungrounded_reference", {}),
    (r"ungrounded|undefined|unknown", "ungrounded_reference", {}),
    # Ignored signal patterns
    (r"ignor(e|ed|ing)\s+error", "ignored_signal", {
        "signal_fact": "error_message",
        "signal_type": "error",
    }),
    (r"ignor(e|ed|ing)\s+warning", "ignored_signal", {
        "signal_fact": "warning_message",
        "signal_type": "warning",
    }),
    # Repeated action patterns
    (r"repeat(ed|ing)\s+.*fail(ed|ure|ing)", "repeated_action", {}),
]


class DatalogCompiler:
    """
    Compiles natural language rules to Soufflé Datalog.

    Uses pattern matching for common rule types and falls back to
    LLM-based compilation for complex rules.

    Example:
        compiler = DatalogCompiler()
        result = compiler.compile_rules(rules)
        print(result.combined_datalog)
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_llm_fallback: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the compiler.

        Args:
            llm_client: Optional LLM client for complex rules
            use_llm_fallback: Whether to use LLM for rules that don't match templates
            logger: Logger instance
        """
        self.llm = llm_client
        self.use_llm_fallback = use_llm_fallback
        self.logger = logger or logging.getLogger(__name__)

    def compile_rules(
        self,
        rules: list[NaturalRule],
    ) -> CompilationResult:
        """
        Compile a list of rules to Datalog.

        Args:
            rules: Rules to compile

        Returns:
            CompilationResult with compiled rules and combined Datalog
        """
        compiled = []
        failed = []

        for rule in rules:
            try:
                compiled_rule = self.compile_rule(rule)
                if compiled_rule.is_valid:
                    compiled.append(compiled_rule)
                else:
                    failed.append((rule, "; ".join(compiled_rule.validation_errors)))
            except Exception as e:
                self.logger.error(f"Failed to compile rule {rule.name}: {e}")
                failed.append((rule, str(e)))

        # Combine all compiled rules
        combined = self._combine_datalog(compiled)

        return CompilationResult(
            compiled_rules=compiled,
            failed_rules=failed,
            combined_datalog=combined,
            statistics={
                "total": len(rules),
                "compiled": len(compiled),
                "failed": len(failed),
                "template_matched": sum(
                    1 for c in compiled
                    if "template:" in c.datalog_code[:100]
                ),
                "llm_generated": sum(
                    1 for c in compiled
                    if "llm_generated" in c.datalog_code[:100]
                ),
            },
        )

    def compile_rule(self, rule: NaturalRule) -> CompiledRule:
        """
        Compile a single rule to Datalog.

        Args:
            rule: Rule to compile

        Returns:
            CompiledRule with Datalog code
        """
        # Try template matching first
        template_match = self._match_template(rule)
        if template_match:
            template_name, template_vars = template_match
            datalog = self._apply_template(template_name, template_vars, rule)
            input_relations = self._extract_input_relations(datalog)

            return CompiledRule(
                rule_id=rule.rule_id,
                natural_rule=rule,
                datalog_code=f"// template:{template_name}\n{datalog}",
                input_relations=input_relations,
                is_valid=True,
            )

        # Try LLM-based compilation
        if self.use_llm_fallback and self.llm:
            datalog = self._compile_with_llm(rule)
            if datalog:
                input_relations = self._extract_input_relations(datalog)
                is_valid, errors = self._validate_datalog(datalog)

                return CompiledRule(
                    rule_id=rule.rule_id,
                    natural_rule=rule,
                    datalog_code=f"// llm_generated\n{datalog}",
                    input_relations=input_relations,
                    is_valid=is_valid,
                    validation_errors=errors,
                )

        # Generate a placeholder/stub rule
        return self._generate_stub_rule(rule)

    def _match_template(
        self,
        rule: NaturalRule,
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Match rule to a template based on natural language patterns."""
        # Combine relevant text for matching
        text = f"{rule.description} {rule.violation_conditions}".lower()

        for pattern, template_name, base_vars in PATTERN_MAPPINGS:
            if re.search(pattern, text, re.IGNORECASE):
                # Build template variables
                vars_copy = base_vars.copy()
                vars_copy["rule_id"] = rule.rule_id
                vars_copy["description"] = rule.description
                vars_copy["message"] = self._generate_message(rule)
                return template_name, vars_copy

        return None

    def _apply_template(
        self,
        template_name: str,
        vars_dict: dict[str, Any],
        rule: NaturalRule,
    ) -> str:
        """Apply a template with variables."""
        template = DATALOG_TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Add any missing common variables
        vars_dict.setdefault("rule_id", rule.rule_id)
        vars_dict.setdefault("description", rule.description)
        vars_dict.setdefault("message", self._generate_message(rule))

        # For contains pattern, extract pattern from violation conditions
        if template_name == "contains_pattern" and "pattern" not in vars_dict:
            vars_dict["pattern"] = self._extract_pattern_from_text(
                rule.violation_conditions
            )

        try:
            return template.format(**vars_dict)
        except KeyError as e:
            self.logger.warning(f"Missing template variable: {e}")
            return ""

    def _compile_with_llm(self, rule: NaturalRule) -> Optional[str]:
        """Use LLM to compile a rule to Datalog."""
        if not self.llm:
            return None

        prompt = f"""Convert this natural language rule to Soufflé Datalog syntax.

Rule: {rule.name}
Description: {rule.description}
Applies when: {rule.conditions}
Violation when: {rule.violation_conditions}

Requirements:
1. Use these conventions:
   - Input facts: .decl fact_name(case_id: symbol, value: symbol)
   - Mark inputs with: .input fact_name
   - Output: violation(case_id, rule_id, message)
2. The rule should be self-contained
3. Use meaningful fact names derived from the rule context
4. Include comments explaining the logic

Output ONLY valid Soufflé Datalog code, no explanations.
"""

        try:
            response, _ = self.llm.complete(
                prompt=prompt,
                system_prompt="You are a Soufflé Datalog expert. Output only valid Datalog code.",
                temperature=0.3,
            )
            return self._extract_datalog(response)
        except Exception as e:
            self.logger.error(f"LLM compilation failed: {e}")
            return None

    def _generate_stub_rule(self, rule: NaturalRule) -> CompiledRule:
        """Generate a placeholder rule when compilation fails."""
        stub = f'''// Rule: {rule.rule_id}
// Description: {rule.description}
// NOTE: This rule could not be automatically compiled.
// Manual implementation required.
//
// Conditions: {rule.conditions}
// Violation: {rule.violation_conditions}

// Placeholder declaration - replace with actual implementation
.decl {rule.rule_id}_input(case_id: symbol, value: symbol)
.input {rule.rule_id}_input

// Placeholder rule - always produces no violations
// violation(id, "{rule.rule_id}", "{self._generate_message(rule)}") :-
//     {rule.rule_id}_input(id, _),
//     false.
'''

        return CompiledRule(
            rule_id=rule.rule_id,
            natural_rule=rule,
            datalog_code=stub,
            input_relations=[f"{rule.rule_id}_input"],
            is_valid=False,
            validation_errors=["Could not compile automatically - manual implementation required"],
        )

    def _generate_message(self, rule: NaturalRule) -> str:
        """Generate a violation message from the rule."""
        # Use description if short enough, otherwise summarize
        if len(rule.description) <= 100:
            return rule.description
        return rule.description[:97] + "..."

    def _extract_pattern_from_text(self, text: str) -> str:
        """Extract a pattern string from natural language text."""
        # Look for quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return quoted[0]

        # Look for code-like patterns
        code_patterns = re.findall(r"`([^`]+)`", text)
        if code_patterns:
            return code_patterns[0]

        # Default to a generic pattern
        return "pattern_not_extracted"

    def _extract_input_relations(self, datalog: str) -> list[str]:
        """Extract input relation names from Datalog code."""
        # Find .input declarations
        inputs = re.findall(r"\.input\s+(\w+)", datalog)
        return list(set(inputs))

    def _extract_datalog(self, text: str) -> str:
        """Extract Datalog code from LLM response."""
        # Look for code blocks
        code_block = re.search(r"```(?:datalog|soufflé|souffle)?\n?(.*?)```", text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # If no code block, try to find Datalog-like content
        lines = text.strip().split("\n")
        datalog_lines = []
        in_code = False

        for line in lines:
            # Heuristic: lines starting with . or // or containing :- are Datalog
            stripped = line.strip()
            if stripped.startswith(".") or stripped.startswith("//") or ":-" in stripped:
                in_code = True
            if in_code:
                datalog_lines.append(line)

        return "\n".join(datalog_lines) if datalog_lines else text

    def _validate_datalog(self, datalog: str) -> tuple[bool, list[str]]:
        """Validate Datalog syntax (basic checks)."""
        errors = []

        # Check for required elements
        if ".decl" not in datalog:
            errors.append("Missing .decl declarations")

        if "violation(" not in datalog and ":-" in datalog:
            errors.append("Missing violation output relation")

        # Check for balanced parentheses
        if datalog.count("(") != datalog.count(")"):
            errors.append("Unbalanced parentheses")

        # Check for balanced quotes
        if datalog.count('"') % 2 != 0:
            errors.append("Unbalanced quotes")

        return len(errors) == 0, errors

    def _combine_datalog(self, compiled_rules: list[CompiledRule]) -> str:
        """Combine multiple compiled rules into one Datalog program."""
        if not compiled_rules:
            return ""

        # Header
        lines = [
            "// Combined Datalog rules",
            f"// Generated from {len(compiled_rules)} rules",
            "",
            "// Common output relation",
            ".decl violation(case_id: symbol, rule_id: symbol, message: symbol)",
            ".output violation",
            "",
        ]

        # Collect all unique input declarations
        seen_decls = set()

        for rule in compiled_rules:
            lines.append(f"// === Rule: {rule.rule_id} ===")
            lines.append(rule.datalog_code)
            lines.append("")

        return "\n".join(lines)


def compile_rules_to_datalog(
    rules: list[NaturalRule],
    llm_client: Optional[Any] = None,
) -> CompilationResult:
    """
    Convenience function to compile rules to Datalog.

    Args:
        rules: Rules to compile
        llm_client: Optional LLM client for complex rules

    Returns:
        CompilationResult with compiled Datalog
    """
    compiler = DatalogCompiler(llm_client=llm_client)
    return compiler.compile_rules(rules)


def compile_rule(
    rule: NaturalRule,
    llm_client: Optional[Any] = None,
) -> CompiledRule:
    """
    Convenience function to compile a single rule.

    Args:
        rule: Rule to compile
        llm_client: Optional LLM client

    Returns:
        CompiledRule with Datalog code
    """
    compiler = DatalogCompiler(llm_client=llm_client)
    return compiler.compile_rule(rule)
