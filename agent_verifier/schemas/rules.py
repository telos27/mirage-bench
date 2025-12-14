"""Rule and policy schemas."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .result import Severity


class RuleType(str, Enum):
    """Type of rule."""
    CONSTRAINT = "constraint"      # Must be satisfied
    REQUIREMENT = "requirement"    # Must be present
    PROHIBITION = "prohibition"    # Must not be present
    PREFERENCE = "preference"      # Preferred but not required


class ConditionOperator(str, Enum):
    """Operators for rule conditions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"        # Regex match
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN = "in"                  # Value in list
    NOT_IN = "not_in"


@dataclass
class RuleCondition:
    """
    A single condition in a rule.

    Attributes:
        field: The field to check (e.g., "output.format", "input.constraints")
        operator: The comparison operator
        value: The value to compare against
        negate: Whether to negate the condition
    """
    field: str
    operator: ConditionOperator
    value: Any
    negate: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "negate": self.negate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleCondition":
        """Create from dictionary."""
        data = data.copy()
        data["operator"] = ConditionOperator(data["operator"])
        return cls(**data)


@dataclass
class Rule:
    """
    A verification rule.

    Attributes:
        rule_id: Unique identifier for this rule
        name: Human-readable name
        description: Detailed description of what this rule checks
        rule_type: Type of rule (constraint, requirement, prohibition, preference)
        layer: Which layer this rule belongs to (1-6)
        conditions: Conditions that trigger this rule
        severity: How severe a violation is
        message_template: Template for violation message
        enabled: Whether this rule is active
        tags: Optional tags for categorization
    """
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    layer: int
    conditions: list[RuleCondition]
    severity: Severity = Severity.ERROR
    message_template: str = ""
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "layer": self.layer,
            "conditions": [c.to_dict() for c in self.conditions],
            "severity": self.severity.value,
            "message_template": self.message_template,
            "enabled": self.enabled,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rule":
        """Create from dictionary."""
        data = data.copy()
        data["rule_type"] = RuleType(data["rule_type"])
        data["severity"] = Severity(data["severity"])
        data["conditions"] = [RuleCondition.from_dict(c) for c in data.get("conditions", [])]
        return cls(**data)


@dataclass
class PolicySpec:
    """
    Specification for a business policy (Layer 3).

    Attributes:
        policy_id: Unique identifier
        deployment_id: Which deployment this applies to
        name: Human-readable name
        description: Description of the policy
        policy_type: Type of policy
        rules: Rules that implement this policy
        priority: Priority for conflict resolution (higher = more important)
        enabled: Whether this policy is active
    """
    policy_id: str
    deployment_id: str
    name: str
    description: str
    policy_type: RuleType
    rules: list[Rule] = field(default_factory=list)
    priority: int = 0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "deployment_id": self.deployment_id,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type.value,
            "rules": [r.to_dict() for r in self.rules],
            "priority": self.priority,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicySpec":
        """Create from dictionary."""
        data = data.copy()
        data["policy_type"] = RuleType(data["policy_type"])
        data["rules"] = [Rule.from_dict(r) for r in data.get("rules", [])]
        return cls(**data)
