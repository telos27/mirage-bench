#!/usr/bin/env python3
"""
Generic fact schema for hallucination detection.

This module defines a domain-agnostic schema for extracting structured facts
from LLM agent inputs and outputs. The schema is designed to support
"common sense" hallucination detection without predefining hallucination types.

Key Principle:
    Hallucinations = Agent behavior that doesn't align with observations + common sense

Two Sources of Truth:
    1. LLM Input (Ground Truth) - What the agent actually observes
    2. Common Sense (LLM Knowledge) - What a reasonable agent should do
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class ViolationType(str, Enum):
    """Types of common sense violations (emergent, not predefined)."""
    UNGROUNDED_REFERENCE = "ungrounded_reference"  # References something not observed
    IGNORED_EVIDENCE = "ignored_evidence"          # Ignores important observation
    REASONING_MISMATCH = "reasoning_mismatch"      # Action contradicts reasoning
    REPEATED_FAILURE = "repeated_failure"          # Repeats action that failed
    STATE_CONFUSION = "state_confusion"            # Misunderstands current state
    GOAL_DEVIATION = "goal_deviation"              # Action doesn't serve goal
    FABRICATION = "fabrication"                    # Makes up information
    OTHER = "other"                                # Other violation types


class Severity(str, Enum):
    """Severity of a violation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceLevel(str, Enum):
    """Confidence in an assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# Input Facts - What the agent observes (Ground Truth)
# =============================================================================

class ObservedFact(BaseModel):
    """A single fact from the agent's observation."""
    content: str = Field(description="The observed fact")
    source: str = Field(description="Where this was observed (screen, history, error, etc.)")
    importance: str = Field(description="Why this might be important: 'critical', 'relevant', 'background'")


class ObservedFacts(BaseModel):
    """Structured representation of what the agent can observe from its input."""

    # Current state observations
    screen_elements: List[str] = Field(
        default_factory=list,
        description="Key UI elements, text, or entities visible in current state"
    )

    # State indicators
    state_indicators: List[str] = Field(
        default_factory=list,
        description="Status messages, errors, warnings, or state information"
    )

    # Historical context
    action_history: List[str] = Field(
        default_factory=list,
        description="Previous actions taken and their outcomes"
    )

    # Task context
    task_goal: str = Field(
        default="",
        description="The goal the agent is trying to achieve"
    )

    # Important facts that should not be ignored
    critical_facts: List[ObservedFact] = Field(
        default_factory=list,
        description="Facts that are critical and should be acknowledged by a reasonable agent"
    )


# =============================================================================
# Agent Response - What the agent outputs
# =============================================================================

class AgentResponse(BaseModel):
    """Structured representation of the agent's output."""

    # Raw content
    thinking: str = Field(description="Agent's reasoning/thinking text")
    action: str = Field(description="Agent's chosen action")

    # Extracted claims
    stated_observations: List[str] = Field(
        default_factory=list,
        description="What the agent claims to observe or believe about the current state"
    )

    stated_reasoning: List[str] = Field(
        default_factory=list,
        description="Agent's explicit reasoning steps"
    )

    stated_intent: str = Field(
        default="",
        description="What the agent says it's trying to accomplish with this action"
    )


# =============================================================================
# Violation Detection - Common sense check results
# =============================================================================

class Violation(BaseModel):
    """A specific common sense violation detected."""

    type: ViolationType = Field(description="Type of violation")
    description: str = Field(description="Human-readable description of the violation")
    evidence_from_input: str = Field(
        default="",
        description="Evidence from input that supports this violation"
    )
    evidence_from_response: str = Field(
        default="",
        description="Evidence from agent response that supports this violation"
    )
    severity: Severity = Field(description="How severe is this violation")

    # For mapping back to traditional hallucination types
    suggested_category: str = Field(
        default="",
        description="Suggested traditional hallucination category (repetitive, misleading, etc.)"
    )


class CommonSenseEvaluation(BaseModel):
    """Result of common sense evaluation."""

    # Overall assessment
    is_reasonable: bool = Field(
        description="Is the agent's response reasonable given the observations?"
    )

    # Specific violations found
    violations: List[Violation] = Field(
        default_factory=list,
        description="List of specific common sense violations"
    )

    # Scoring (for compatibility with existing 0-2 scale)
    score: int = Field(
        description="Score 0-2: 0=hallucination, 1=partial/unclear, 2=reasonable"
    )

    # Explanation
    reasoning: str = Field(
        description="Explanation of why this is or isn't a hallucination"
    )

    # Confidence in this assessment
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence in this evaluation"
    )

    # Emergent type (derived, not predefined)
    emergent_type: str = Field(
        default="",
        description="Emergent hallucination type based on violations found"
    )


# =============================================================================
# Complete Extraction Schema
# =============================================================================

class GenericExtractionResult(BaseModel):
    """
    Complete extraction result for generic hallucination detection.

    This schema captures:
    1. What the agent observed (ground truth from input)
    2. What the agent responded (thinking + action)
    3. Common sense evaluation (does response match observations?)
    """

    # Source 1: Ground truth from input
    observed_facts: ObservedFacts = Field(
        description="Facts extracted from the agent's input"
    )

    # Agent's response
    agent_response: AgentResponse = Field(
        description="Structured representation of agent's response"
    )

    # Source 2 + Comparison: Common sense check
    evaluation: CommonSenseEvaluation = Field(
        description="Common sense evaluation of agent's response"
    )


# =============================================================================
# Utility Functions
# =============================================================================

def violations_to_score(violations: List[Violation]) -> int:
    """
    Convert violations to a 0-2 score for compatibility.

    - 0: High severity violation or multiple medium violations
    - 1: Medium severity or low severity violations
    - 2: No violations
    """
    if not violations:
        return 2

    high_count = sum(1 for v in violations if v.severity == Severity.HIGH)
    medium_count = sum(1 for v in violations if v.severity == Severity.MEDIUM)

    if high_count > 0 or medium_count >= 2:
        return 0
    elif medium_count > 0 or len(violations) > 0:
        return 1
    else:
        return 2


def derive_emergent_type(violations: List[Violation]) -> str:
    """
    Derive an emergent hallucination type from violations.

    Maps violation patterns to traditional hallucination categories.
    """
    if not violations:
        return "none"

    # Count violation types
    type_counts = {}
    for v in violations:
        type_counts[v.type] = type_counts.get(v.type, 0) + 1

    # Map to traditional types
    if ViolationType.REPEATED_FAILURE in type_counts:
        return "repetitive"
    elif ViolationType.UNGROUNDED_REFERENCE in type_counts:
        return "misleading_or_fabrication"
    elif ViolationType.IGNORED_EVIDENCE in type_counts:
        return "unachievable_or_erroneous"
    elif ViolationType.STATE_CONFUSION in type_counts:
        return "misleading"
    elif ViolationType.REASONING_MISMATCH in type_counts:
        return "inconsistent"
    else:
        return "unknown"
