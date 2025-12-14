"""
Rule validation for extracted rules.

Validates natural language rules before compilation to Datalog:
- Completeness (required fields present)
- Consistency (no contradictions)
- Redundancy detection
- Quality checks
"""

import logging
import re
from typing import Optional

from .schemas import (
    NaturalRule,
    RuleSeverity,
    ValidationResult,
)


class RuleValidator:
    """
    Validates extracted rules for completeness and consistency.

    Performs:
    - Field completeness checks
    - Content quality checks
    - Inter-rule consistency checks
    - Redundancy detection

    Example:
        validator = RuleValidator()
        results = validator.validate_rules(rules)
        for result in results:
            if not result.is_valid:
                print(f"Rule {result.rule.name} failed: {result.errors}")
    """

    # Minimum content lengths
    MIN_DESCRIPTION_LENGTH = 10
    MIN_CONDITIONS_LENGTH = 5
    MIN_VIOLATION_LENGTH = 5

    # Words that suggest vague or incomplete rules
    VAGUE_WORDS = [
        "maybe", "perhaps", "sometimes", "might", "could",
        "etc", "and so on", "and more", "things like",
    ]

    # Words that suggest overly broad rules
    OVERLY_BROAD_WORDS = [
        "always", "never", "all", "any", "every", "none",
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the validator."""
        self.logger = logger or logging.getLogger(__name__)

    def validate_rules(
        self,
        rules: list[NaturalRule],
        check_redundancy: bool = True,
    ) -> list[ValidationResult]:
        """
        Validate a list of rules.

        Args:
            rules: Rules to validate
            check_redundancy: Whether to check for redundant rules

        Returns:
            List of validation results
        """
        results = []

        for rule in rules:
            result = self.validate_rule(rule)
            results.append(result)

        # Check for redundancies across rules
        if check_redundancy and len(rules) > 1:
            redundancy_warnings = self._check_redundancy(rules)
            # Add redundancy warnings to relevant rules
            for rule_id, warning in redundancy_warnings:
                for result in results:
                    if result.rule.rule_id == rule_id:
                        result.warnings.append(warning)

        return results

    def validate_rule(self, rule: NaturalRule) -> ValidationResult:
        """
        Validate a single rule.

        Args:
            rule: Rule to validate

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []

        # Check completeness
        completeness_errors = self._check_completeness(rule)
        errors.extend(completeness_errors)

        # Check content quality
        quality_warnings, quality_suggestions = self._check_quality(rule)
        warnings.extend(quality_warnings)
        suggestions.extend(quality_suggestions)

        # Check for vague language
        vague_warnings = self._check_vagueness(rule)
        warnings.extend(vague_warnings)

        # Check for overly broad rules
        broad_warnings = self._check_breadth(rule)
        warnings.extend(broad_warnings)

        # Check rule name format
        name_errors = self._check_name_format(rule)
        errors.extend(name_errors)

        is_valid = len(errors) == 0

        return ValidationResult(
            rule=rule,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

    def _check_completeness(self, rule: NaturalRule) -> list[str]:
        """Check that required fields are present and sufficient."""
        errors = []

        # Required fields
        if not rule.name:
            errors.append("Rule name is required")
        if not rule.description:
            errors.append("Rule description is required")
        elif len(rule.description) < self.MIN_DESCRIPTION_LENGTH:
            errors.append(f"Description too short (min {self.MIN_DESCRIPTION_LENGTH} chars)")

        if not rule.conditions:
            errors.append("Rule conditions are required")
        elif len(rule.conditions) < self.MIN_CONDITIONS_LENGTH:
            errors.append(f"Conditions too short (min {self.MIN_CONDITIONS_LENGTH} chars)")

        if not rule.violation_conditions:
            errors.append("Violation conditions are required")
        elif len(rule.violation_conditions) < self.MIN_VIOLATION_LENGTH:
            errors.append(f"Violation conditions too short (min {self.MIN_VIOLATION_LENGTH} chars)")

        return errors

    def _check_quality(
        self,
        rule: NaturalRule,
    ) -> tuple[list[str], list[str]]:
        """Check content quality and provide suggestions."""
        warnings = []
        suggestions = []

        # Check for missing examples
        if not rule.examples:
            suggestions.append("Add examples to clarify what violations look like")

        # Check for missing rationale
        if not rule.rationale:
            suggestions.append("Add rationale to explain why this rule exists")

        # Check for missing tags
        if not rule.tags:
            suggestions.append("Add tags for better categorization")

        # Check description clarity
        if rule.description and not rule.description.endswith((".","!","?")):
            suggestions.append("Description should end with proper punctuation")

        # Check for actionable conditions
        if rule.conditions:
            action_words = ["when", "if", "while", "during", "upon"]
            has_action_word = any(word in rule.conditions.lower() for word in action_words)
            if not has_action_word:
                suggestions.append(
                    "Conditions should specify when the rule applies "
                    "(use 'when', 'if', etc.)"
                )

        # Check for testable violation conditions
        if rule.violation_conditions:
            testable_words = ["contains", "includes", "matches", "has", "is", "fails", "missing"]
            has_testable = any(word in rule.violation_conditions.lower() for word in testable_words)
            if not has_testable:
                warnings.append(
                    "Violation conditions may be hard to test programmatically"
                )

        return warnings, suggestions

    def _check_vagueness(self, rule: NaturalRule) -> list[str]:
        """Check for vague language that makes rules hard to apply."""
        warnings = []

        text_to_check = f"{rule.description} {rule.conditions} {rule.violation_conditions}"
        text_lower = text_to_check.lower()

        found_vague = [
            word for word in self.VAGUE_WORDS
            if word in text_lower
        ]

        if found_vague:
            warnings.append(
                f"Rule contains vague language: {', '.join(found_vague)}. "
                "Consider making conditions more specific."
            )

        return warnings

    def _check_breadth(self, rule: NaturalRule) -> list[str]:
        """Check for overly broad rules that may cause false positives."""
        warnings = []

        # Check for absolute terms that often indicate overly broad rules
        conditions_lower = rule.conditions.lower()
        violation_lower = rule.violation_conditions.lower()

        # "Never" and "always" in violation conditions are concerning
        concerning_patterns = [
            (r"\bany\b.*\bviolation\b", "Using 'any' may catch too many cases"),
            (r"\ball\b.*\bmust\b", "'All must' rules are often too strict"),
            (r"\bnever\b.*\ballow\b", "'Never allow' may be too restrictive"),
        ]

        for pattern, message in concerning_patterns:
            if re.search(pattern, violation_lower):
                warnings.append(message)

        # Check if conditions are too broad (no specific context)
        if len(rule.conditions.split()) < 5:
            warnings.append(
                "Conditions may be too brief - consider adding more context"
            )

        return warnings

    def _check_name_format(self, rule: NaturalRule) -> list[str]:
        """Check that rule name follows conventions."""
        errors = []

        name = rule.name

        # Should be snake_case
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            errors.append(
                f"Rule name '{name}' should be snake_case "
                "(lowercase with underscores)"
            )

        # Should not be too long
        if len(name) > 50:
            errors.append(f"Rule name too long ({len(name)} chars, max 50)")

        # Should not be too short
        if len(name) < 3:
            errors.append(f"Rule name too short ({len(name)} chars, min 3)")

        return errors

    def _check_redundancy(
        self,
        rules: list[NaturalRule],
    ) -> list[tuple[str, str]]:
        """Check for potentially redundant rules."""
        warnings = []

        # Group rules by domain
        by_domain: dict[str, list[NaturalRule]] = {}
        for rule in rules:
            domain = rule.domain or "general"
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(rule)

        # Check each domain for redundancies
        for domain, domain_rules in by_domain.items():
            if len(domain_rules) < 2:
                continue

            # Simple similarity check based on description overlap
            for i, rule1 in enumerate(domain_rules):
                for rule2 in domain_rules[i + 1:]:
                    similarity = self._text_similarity(
                        rule1.description,
                        rule2.description,
                    )
                    if similarity > 0.7:
                        warnings.append((
                            rule1.rule_id,
                            f"Rule may be redundant with '{rule2.name}' "
                            f"(similarity: {similarity:.0%})"
                        ))

        return warnings

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def filter_valid_rules(
        self,
        rules: list[NaturalRule],
    ) -> tuple[list[NaturalRule], list[tuple[NaturalRule, list[str]]]]:
        """
        Filter rules, returning valid ones and rejected ones with reasons.

        Args:
            rules: Rules to filter

        Returns:
            Tuple of (valid_rules, rejected_rules_with_reasons)
        """
        valid = []
        rejected = []

        for rule in rules:
            result = self.validate_rule(rule)
            if result.is_valid:
                valid.append(rule)
            else:
                rejected.append((rule, result.errors))

        return valid, rejected


def validate_rules(rules: list[NaturalRule]) -> list[ValidationResult]:
    """
    Convenience function to validate a list of rules.

    Args:
        rules: Rules to validate

    Returns:
        List of validation results
    """
    validator = RuleValidator()
    return validator.validate_rules(rules)


def filter_valid_rules(
    rules: list[NaturalRule],
) -> tuple[list[NaturalRule], list[tuple[NaturalRule, list[str]]]]:
    """
    Convenience function to filter valid rules.

    Args:
        rules: Rules to filter

    Returns:
        Tuple of (valid_rules, rejected_rules_with_reasons)
    """
    validator = RuleValidator()
    return validator.filter_valid_rules(rules)
