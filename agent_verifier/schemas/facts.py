"""Extracted facts schemas for verification."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class InputFacts:
    """
    Facts extracted from the input/prompt.

    Attributes:
        task_goal: The stated goal or task from the prompt
        visible_elements: Elements/entities visible to the agent
        error_messages: Any error messages present
        action_history: Previous actions taken in the session
        state_info: Current state information (URL, page title, etc.)
        constraints: Explicit constraints from the prompt
        format_requirements: Required output format (JSON, code, etc.)
    """
    task_goal: Optional[str] = None
    visible_elements: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    state_info: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    format_requirements: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_goal": self.task_goal,
            "visible_elements": self.visible_elements,
            "error_messages": self.error_messages,
            "action_history": self.action_history,
            "state_info": self.state_info,
            "constraints": self.constraints,
            "format_requirements": self.format_requirements,
        }


@dataclass
class OutputFacts:
    """
    Facts extracted from the LLM output.

    Attributes:
        stated_observations: What the agent claims to observe
        reasoning_steps: The reasoning chain in the output
        action_target: The action/element being targeted
        references: Things referenced in the output
        claims: Factual claims made in the output
        format_used: The format of the output
    """
    stated_observations: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    action_target: Optional[str] = None
    references: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    format_used: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stated_observations": self.stated_observations,
            "reasoning_steps": self.reasoning_steps,
            "action_target": self.action_target,
            "references": self.references,
            "claims": self.claims,
            "format_used": self.format_used,
        }


@dataclass
class ExtractedFacts:
    """
    Container for all extracted facts from a verification request.

    Attributes:
        input_facts: Facts from the input/prompt
        output_facts: Facts from the LLM output
        session_facts: Facts from session context (if available)
        metadata: Additional metadata about extraction
    """
    input_facts: InputFacts = field(default_factory=InputFacts)
    output_facts: OutputFacts = field(default_factory=OutputFacts)
    session_facts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_facts": self.input_facts.to_dict(),
            "output_facts": self.output_facts.to_dict(),
            "session_facts": self.session_facts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedFacts":
        """Create from dictionary."""
        return cls(
            input_facts=InputFacts(**data.get("input_facts", {})),
            output_facts=OutputFacts(**data.get("output_facts", {})),
            session_facts=data.get("session_facts", {}),
            metadata=data.get("metadata", {}),
        )
