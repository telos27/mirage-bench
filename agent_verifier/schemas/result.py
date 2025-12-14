"""Verification result schemas."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


class Severity(str, Enum):
    """Severity level of a violation."""
    ERROR = "error"      # Critical violation, must fail
    WARNING = "warning"  # Notable issue, may fail depending on policy
    INFO = "info"        # Informational, does not cause failure


@dataclass
class Violation:
    """
    A single verification violation detected.

    Attributes:
        layer: The layer number where this violation was detected (1-6)
        violation_type: Category of violation (e.g., "ungrounded_reference", "policy_breach")
        severity: How severe this violation is
        message: Human-readable description of the violation
        evidence: Supporting evidence for the violation
        rule_id: Optional identifier of the rule that was violated
        suggestion: Optional suggestion for how to fix the violation
    """
    layer: int
    violation_type: str
    severity: Severity
    message: str
    evidence: dict[str, Any]
    rule_id: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer": self.layer,
            "violation_type": self.violation_type,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence,
            "rule_id": self.rule_id,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Violation":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("severity"), str):
            data["severity"] = Severity(data["severity"])
        return cls(**data)


@dataclass
class ReasoningStep:
    """
    A single step in the verification reasoning chain.

    This provides transparency into how the verification decision was made.

    Attributes:
        layer: The layer number for this reasoning step
        step_type: Type of reasoning (e.g., "fact_extraction", "rule_application", "inference")
        description: Human-readable description of what was done
        inputs: What inputs were used for this step
        outputs: What was produced by this step
        rule_applied: Optional rule identifier if a specific rule was applied
    """
    layer: int
    step_type: str
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    rule_applied: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer": self.layer,
            "step_type": self.step_type,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "rule_applied": self.rule_applied,
        }


@dataclass
class VerificationResult:
    """
    The complete result of a verification request.

    Attributes:
        request_id: The ID of the request this result is for
        verdict: Overall pass/fail decision
        violations: List of all violations detected
        reasoning: Chain of reasoning steps for transparency
        latency_ms: How long verification took in milliseconds
        layers_checked: Which layers were evaluated
        verified_output: Optional corrected/verified output
        metadata: Optional additional metadata
    """
    request_id: str
    verdict: Literal["pass", "fail"]
    violations: list[Violation]
    reasoning: list[ReasoningStep]
    latency_ms: int
    layers_checked: list[int]
    verified_output: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "verdict": self.verdict,
            "violations": [v.to_dict() for v in self.violations],
            "reasoning": [r.to_dict() for r in self.reasoning],
            "latency_ms": self.latency_ms,
            "layers_checked": self.layers_checked,
            "verified_output": self.verified_output,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationResult":
        """Create from dictionary."""
        data = data.copy()
        data["violations"] = [Violation.from_dict(v) for v in data.get("violations", [])]
        data["reasoning"] = [ReasoningStep(**r) for r in data.get("reasoning", [])]
        return cls(**data)

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level violations."""
        return any(v.severity == Severity.ERROR for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level violations."""
        return any(v.severity == Severity.WARNING for v in self.violations)

    def violations_by_layer(self, layer: int) -> list[Violation]:
        """Get violations for a specific layer."""
        return [v for v in self.violations if v.layer == layer]

    def violations_by_type(self, violation_type: str) -> list[Violation]:
        """Get violations of a specific type."""
        return [v for v in self.violations if v.violation_type == violation_type]
