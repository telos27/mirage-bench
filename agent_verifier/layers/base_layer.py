"""Abstract base class for verification layers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..schemas.request import VerificationRequest
from ..schemas.result import Violation, ReasoningStep
from ..schemas.rules import Rule


@dataclass
class LayerResult:
    """
    Result from checking a single layer.

    Attributes:
        layer: The layer number
        violations: Violations detected by this layer
        reasoning: Reasoning steps for transparency
        facts_extracted: Any facts extracted during checking
        metadata: Additional layer-specific metadata
    """
    layer: int
    violations: list[Violation] = field(default_factory=list)
    reasoning: list[ReasoningStep] = field(default_factory=list)
    facts_extracted: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_violations(self) -> bool:
        """Check if any violations were detected."""
        return len(self.violations) > 0

    def add_violation(self, violation: Violation) -> None:
        """Add a violation to this result."""
        self.violations.append(violation)

    def add_reasoning(self, step: ReasoningStep) -> None:
        """Add a reasoning step to this result."""
        self.reasoning.append(step)


class BaseLayer(ABC):
    """
    Abstract base class for verification layers.

    Each layer in the 6-layer verification architecture extends this class.
    Layers are responsible for:
    1. Loading their relevant rules/policies
    2. Checking requests against those rules
    3. Producing violations and reasoning for transparency

    Layer numbers:
        1 - Common Knowledge (universal truths, logic, formats)
        2 - Domain Best Practices (agent-type specific patterns)
        3 - Business Policies (per-deployment rules)
        4 - User Preferences (per-user settings)
        5 - Session Context (conversation history)
        6 - Active Request (current prompt constraints)
    """

    def __init__(self, layer_number: int, layer_name: str):
        """
        Initialize a layer.

        Args:
            layer_number: The layer's position in the hierarchy (1-6)
            layer_name: Human-readable name for the layer
        """
        self.layer_number = layer_number
        self.layer_name = layer_name

    @abstractmethod
    def check(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> LayerResult:
        """
        Check this layer against the request.

        Args:
            request: The verification request to check
            context: Accumulated context from previous layers and extractors

        Returns:
            LayerResult with any violations and reasoning
        """
        pass

    @abstractmethod
    def load_rules(self, deployment_id: str) -> list[Rule]:
        """
        Load rules for this layer.

        Args:
            deployment_id: The deployment to load rules for

        Returns:
            List of rules applicable to this layer for the deployment
        """
        pass

    def create_violation(
        self,
        violation_type: str,
        message: str,
        evidence: dict[str, Any],
        severity: str = "error",
        rule_id: str | None = None,
        suggestion: str | None = None,
    ) -> Violation:
        """
        Helper to create a violation with this layer's number.

        Args:
            violation_type: Category of violation
            message: Human-readable description
            evidence: Supporting evidence
            severity: error, warning, or info
            rule_id: Optional rule identifier
            suggestion: Optional fix suggestion

        Returns:
            A Violation object
        """
        from ..schemas.result import Severity
        severity_enum = Severity(severity)
        return Violation(
            layer=self.layer_number,
            violation_type=violation_type,
            severity=severity_enum,
            message=message,
            evidence=evidence,
            rule_id=rule_id,
            suggestion=suggestion,
        )

    def create_reasoning_step(
        self,
        step_type: str,
        description: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        rule_applied: str | None = None,
    ) -> ReasoningStep:
        """
        Helper to create a reasoning step with this layer's number.

        Args:
            step_type: Type of reasoning (fact_extraction, rule_application, inference)
            description: What was done
            inputs: Input data used
            outputs: Output produced
            rule_applied: Optional rule identifier

        Returns:
            A ReasoningStep object
        """
        return ReasoningStep(
            layer=self.layer_number,
            step_type=step_type,
            description=description,
            inputs=inputs or {},
            outputs=outputs or {},
            rule_applied=rule_applied,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer={self.layer_number}, name='{self.layer_name}')"
