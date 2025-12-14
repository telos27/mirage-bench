#!/usr/bin/env python3
"""
General fact extraction schema for neuro-symbolic verification.

This module defines a domain-general schema for extracting structured facts
from LLM agent outputs. The schema is designed to be:
1. General enough for multiple hallucination types
2. Expressive enough for meaningful reasoning
3. Compatible with Datalog/Soufflé inference

Inspired by: ~/pysem/psychology-knowledge/experiments/triple_extraction.py
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


# =============================================================================
# Core Enums - Domain-General Categories
# =============================================================================

class AwarenessLevel(str, Enum):
    """Level of awareness about a situation or pattern."""
    NONE = "none"           # No awareness detected
    IMPLICIT = "implicit"   # Indirect signs of awareness
    EXPLICIT = "explicit"   # Clear, direct acknowledgment


class ConfidenceLevel(str, Enum):
    """Confidence in an extracted fact."""
    LOW = "low"       # Uncertain extraction
    MEDIUM = "medium" # Reasonably confident
    HIGH = "high"     # Very confident


class SemanticRelation(str, Enum):
    """Semantic relationship between actions/entities.

    General predicates inspired by pysem triple_extraction.py:
    - is_a, causes, leads_to, requires, contrasts_with, etc.
    """
    IDENTICAL = "identical"           # Exact semantic match
    EQUIVALENT = "equivalent"         # Same meaning, different form
    SIMILAR = "similar"               # Related but not equivalent
    DIFFERENT = "different"           # Clearly different
    OPPOSITE = "opposite"             # Contradictory/opposite
    SUBSET_OF = "subset_of"           # More specific version
    SUPERSET_OF = "superset_of"       # More general version
    UNKNOWN = "unknown"               # Cannot determine


class ReasoningQuality(str, Enum):
    """Quality of the agent's reasoning process."""
    POOR = "poor"           # Flawed, missing key considerations
    ADEQUATE = "adequate"   # Basic reasoning, nothing notable
    GOOD = "good"           # Sound reasoning with awareness
    EXCELLENT = "excellent" # Exceptional insight and adaptation


class IntentType(str, Enum):
    """Type of intent behind an action."""
    CONTINUE = "continue"       # Continue current approach
    RETRY = "retry"             # Retry same action
    ADAPT = "adapt"             # Modify approach
    ABANDON = "abandon"         # Give up on current path
    EXPLORE = "explore"         # Try something new
    UNKNOWN = "unknown"


# =============================================================================
# Extracted Facts - What LLM extracts from agent output
# =============================================================================

class ActionSemantics(BaseModel):
    """Semantic understanding of an action."""
    action_type: str = Field(description="High-level action category (e.g., 'click', 'type', 'scroll', 'navigate')")
    target: Optional[str] = Field(default=None, description="Target element or entity")
    parameters: Optional[str] = Field(default=None, description="Key parameters or values")
    intent: IntentType = Field(description="Inferred intent behind the action")
    normalized_form: str = Field(description="Normalized representation for comparison")


class ActionComparison(BaseModel):
    """Comparison between two actions."""
    relation: SemanticRelation = Field(description="Semantic relationship between actions")
    confidence: ConfidenceLevel = Field(description="Confidence in this comparison")
    explanation: str = Field(description="Brief explanation of the relationship")


class PatternAwareness(BaseModel):
    """Agent's awareness of patterns (repetition, loops, history)."""
    awareness_level: AwarenessLevel = Field(description="Level of awareness about patterns")
    recognized_pattern: Optional[str] = Field(default=None, description="What pattern was recognized, if any")
    response_to_pattern: Optional[str] = Field(default=None, description="How agent responded to the pattern")
    evidence: List[str] = Field(default_factory=list, description="Quotes/evidence supporting this assessment")


class ReasoningAssessment(BaseModel):
    """Assessment of the agent's reasoning quality."""
    quality: ReasoningQuality = Field(description="Overall reasoning quality")
    considers_alternatives: bool = Field(description="Whether alternatives were considered")
    considers_history: bool = Field(description="Whether history was considered")
    identifies_issues: bool = Field(description="Whether problems were identified")
    adapts_approach: bool = Field(description="Whether approach was adapted")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from reasoning")
    key_failures: List[str] = Field(default_factory=list, description="Key failures in reasoning")


# =============================================================================
# Main Extraction Schema - What LLM returns
# =============================================================================

class ExtractedFacts(BaseModel):
    """
    Complete set of facts extracted from agent output.

    This is the main schema that the LLM populates.
    Designed to be general enough for multiple verification scenarios.
    """
    # Action understanding
    current_action: ActionSemantics = Field(description="Semantic understanding of the current action")
    reference_action: Optional[ActionSemantics] = Field(
        default=None,
        description="Semantic understanding of reference/comparison action (e.g., repetitive action)"
    )
    action_comparison: Optional[ActionComparison] = Field(
        default=None,
        description="Comparison between current and reference actions"
    )

    # Awareness and reasoning
    pattern_awareness: PatternAwareness = Field(description="Agent's awareness of patterns")
    reasoning_assessment: ReasoningAssessment = Field(description="Assessment of reasoning quality")

    # Meta
    extraction_confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Overall confidence in this extraction"
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Any notes about the extraction process"
    )


# =============================================================================
# Datalog Fact Generation
# =============================================================================

def facts_to_datalog(facts: ExtractedFacts, case_id: str) -> dict[str, list[tuple]]:
    """
    Convert ExtractedFacts to Datalog-compatible fact tuples.

    Returns a dict mapping relation name to list of tuples.
    These can be written to .facts files for Soufflé.
    """
    result = {
        "action_relation": [],
        "awareness_level": [],
        "reasoning_quality": [],
        "considers_alternatives": [],
        "considers_history": [],
        "identifies_issues": [],
        "adapts_approach": [],
        "action_intent": [],
        "extraction_confidence": [],
    }

    # Action comparison
    if facts.action_comparison:
        result["action_relation"].append((
            case_id,
            facts.action_comparison.relation.value,
            facts.action_comparison.confidence.value
        ))

    # Pattern awareness
    result["awareness_level"].append((
        case_id,
        facts.pattern_awareness.awareness_level.value
    ))

    # Reasoning assessment
    ra = facts.reasoning_assessment
    result["reasoning_quality"].append((case_id, ra.quality.value))
    result["considers_alternatives"].append((case_id, "true" if ra.considers_alternatives else "false"))
    result["considers_history"].append((case_id, "true" if ra.considers_history else "false"))
    result["identifies_issues"].append((case_id, "true" if ra.identifies_issues else "false"))
    result["adapts_approach"].append((case_id, "true" if ra.adapts_approach else "false"))

    # Action intent
    result["action_intent"].append((case_id, facts.current_action.intent.value))

    # Meta
    result["extraction_confidence"].append((case_id, facts.extraction_confidence.value))

    return result


def write_datalog_facts(facts_dict: dict[str, list[tuple]], output_dir: str):
    """Write fact tuples to .facts files for Soufflé.

    Note: Soufflé expects tab-separated values WITHOUT quotes for symbol types.
    Quotes in the input are treated as part of the string value.
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for relation_name, tuples in facts_dict.items():
        facts_file = output_path / f"{relation_name}.facts"
        with open(facts_file, "w") as f:
            for tup in tuples:
                # Write values without quotes - Soufflé expects raw tab-separated values
                escaped = []
                for val in tup:
                    if isinstance(val, str):
                        # Escape tabs and newlines that would break the format
                        val = val.replace('\t', ' ').replace('\n', ' ')
                        escaped.append(val)
                    else:
                        escaped.append(str(val))
                f.write("\t".join(escaped) + "\n")
