"""
Schemas for LLM-based rule extraction.

Defines data structures for natural language rules extracted from LLMs
before they are validated and compiled to Datalog.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RuleSeverity(str, Enum):
    """Severity level for rule violations."""
    ERROR = "error"       # Critical violation, must fail
    WARNING = "warning"   # Important but not critical
    INFO = "info"         # Informational, minor issue


class RuleDomain(str, Enum):
    """Predefined domains for rule categorization."""
    CODING = "coding"
    CUSTOMER_SERVICE = "customer_service"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_GENERATION = "content_generation"
    GENERAL = "general"
    CUSTOM = "custom"


@dataclass
class NaturalRule:
    """
    A rule in natural language form, as extracted from an LLM.

    This is the intermediate representation before compilation to Datalog.

    Attributes:
        rule_id: Unique identifier (generated if not provided)
        name: Short, descriptive name (e.g., "no_eval_user_input")
        description: Human-readable description of the rule
        domain: Domain this rule applies to
        conditions: When the rule applies (natural language)
        violation_conditions: What constitutes a violation (natural language)
        severity: How severe a violation is
        examples: Optional examples of violations
        rationale: Why this rule exists
        tags: Optional tags for categorization
        confidence: Extraction confidence score (0-1)
        source: Where this rule was extracted from
    """
    name: str
    description: str
    domain: str
    conditions: str
    violation_conditions: str
    severity: RuleSeverity = RuleSeverity.ERROR
    rule_id: str = ""
    examples: list[str] = field(default_factory=list)
    rationale: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "llm_extraction"

    def __post_init__(self):
        """Generate rule_id if not provided."""
        if not self.rule_id:
            # Generate from name, sanitizing for Datalog
            self.rule_id = self.name.lower().replace(" ", "_").replace("-", "_")
            # Remove non-alphanumeric except underscore
            self.rule_id = "".join(c for c in self.rule_id if c.isalnum() or c == "_")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "conditions": self.conditions,
            "violation_conditions": self.violation_conditions,
            "severity": self.severity.value,
            "examples": self.examples,
            "rationale": self.rationale,
            "tags": self.tags,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NaturalRule":
        """Create from dictionary."""
        data = data.copy()
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = RuleSeverity(data["severity"])
        return cls(**data)


@dataclass
class CompiledRule:
    """
    A rule compiled to Datalog format.

    Attributes:
        rule_id: Original rule ID
        natural_rule: The source natural language rule
        datalog_code: The compiled Datalog code
        input_relations: Required input relations (facts that must be provided)
        output_relation: Output relation name (usually "violation")
        is_valid: Whether the compilation succeeded
        validation_errors: Any errors during validation/compilation
    """
    rule_id: str
    natural_rule: NaturalRule
    datalog_code: str
    input_relations: list[str] = field(default_factory=list)
    output_relation: str = "violation"
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "natural_rule": self.natural_rule.to_dict(),
            "datalog_code": self.datalog_code,
            "input_relations": self.input_relations,
            "output_relation": self.output_relation,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


@dataclass
class ExtractionResult:
    """
    Result of rule extraction from an LLM.

    Attributes:
        domain: The domain for which rules were extracted
        description: The domain description used for extraction
        rules: Extracted rules
        model: LLM model used for extraction
        prompt_tokens: Tokens used in prompt
        completion_tokens: Tokens in completion
        total_cost: Estimated cost
    """
    domain: str
    description: str
    rules: list[NaturalRule]
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "description": self.description,
            "rules": [r.to_dict() for r in self.rules],
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost": self.total_cost,
        }


@dataclass
class ValidationResult:
    """
    Result of rule validation.

    Attributes:
        rule: The rule being validated
        is_valid: Whether the rule passed validation
        errors: Validation errors
        warnings: Validation warnings
        suggestions: Suggestions for improvement
    """
    rule: NaturalRule
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class CompilationResult:
    """
    Result of compiling rules to Datalog.

    Attributes:
        compiled_rules: Successfully compiled rules
        failed_rules: Rules that failed compilation
        combined_datalog: All rules combined into one Datalog program
        statistics: Compilation statistics
    """
    compiled_rules: list[CompiledRule] = field(default_factory=list)
    failed_rules: list[tuple[NaturalRule, str]] = field(default_factory=list)
    combined_datalog: str = ""
    statistics: dict[str, Any] = field(default_factory=dict)
