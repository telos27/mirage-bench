"""
Prompt constraint extraction for Layer 6 (Prompt Constraints).

Extracts explicit instructions, format requirements, and safety constraints
from system prompts and user messages. Used to verify LLM follows instructions.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ConstraintType(Enum):
    """Types of prompt constraints."""
    MUST_DO = "must_do"           # Required action/behavior
    MUST_NOT = "must_not"         # Prohibited action/behavior
    FORMAT = "format"             # Output format requirement
    PERSONA = "persona"           # Role/character to adopt
    SAFETY = "safety"             # Safety/ethical constraint
    BOUNDARY = "boundary"         # Scope/boundary limitation
    STYLE = "style"               # Writing style requirement


@dataclass
class PromptConstraint:
    """A single constraint extracted from a prompt."""
    constraint_type: ConstraintType
    content: str
    source: str = "unknown"       # "system" or "user"
    confidence: float = 1.0       # How confident in extraction
    original_text: str = ""       # Original text that produced this

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.constraint_type.value,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "original_text": self.original_text,
        }


@dataclass
class ExtractedConstraints:
    """All constraints extracted from a prompt."""
    must_do: list[PromptConstraint] = field(default_factory=list)
    must_not: list[PromptConstraint] = field(default_factory=list)
    format_requirements: list[PromptConstraint] = field(default_factory=list)
    persona: Optional[PromptConstraint] = None
    safety_constraints: list[PromptConstraint] = field(default_factory=list)
    boundaries: list[PromptConstraint] = field(default_factory=list)
    style_requirements: list[PromptConstraint] = field(default_factory=list)

    def all_constraints(self) -> list[PromptConstraint]:
        """Get all constraints as a flat list."""
        constraints = (
            self.must_do +
            self.must_not +
            self.format_requirements +
            self.safety_constraints +
            self.boundaries +
            self.style_requirements
        )
        if self.persona:
            constraints.append(self.persona)
        return constraints

    def to_dict(self) -> dict[str, Any]:
        return {
            "must_do": [c.to_dict() for c in self.must_do],
            "must_not": [c.to_dict() for c in self.must_not],
            "format_requirements": [c.to_dict() for c in self.format_requirements],
            "persona": self.persona.to_dict() if self.persona else None,
            "safety_constraints": [c.to_dict() for c in self.safety_constraints],
            "boundaries": [c.to_dict() for c in self.boundaries],
            "style_requirements": [c.to_dict() for c in self.style_requirements],
        }


class PromptConstraintExtractor:
    """
    Extracts constraints from prompts using heuristic patterns.

    Identifies:
    - Required actions (must do X)
    - Prohibited actions (don't do Y)
    - Format requirements (respond in JSON)
    - Personas (you are a helpful assistant)
    - Safety constraints (never provide harmful info)
    - Boundaries (only answer questions about X)
    - Style requirements (be concise, formal)

    Example:
        extractor = PromptConstraintExtractor()
        constraints = extractor.extract(system_prompt, user_message)
    """

    # MUST DO patterns
    MUST_DO_PATTERNS = [
        (r"[Yy]ou\s+must\s+([^.!?\n]+)", 0.95),
        (r"[Aa]lways\s+([^.!?\n]+)", 0.9),
        (r"[Mm]ake\s+sure\s+(?:to\s+)?([^.!?\n]+)", 0.9),
        (r"[Ee]nsure\s+(?:that\s+)?([^.!?\n]+)", 0.9),
        (r"[Rr]equired:\s*([^.!?\n]+)", 0.95),
        (r"[Ii]t\s+is\s+(?:essential|important|critical)\s+(?:to|that)\s+([^.!?\n]+)", 0.9),
        (r"[Pp]lease\s+(?:make\s+sure\s+to\s+)?([^.!?\n]+)", 0.7),
        (r"[Yy]ou\s+should\s+([^.!?\n]+)", 0.8),
        (r"[Yy]ou\s+need\s+to\s+([^.!?\n]+)", 0.85),
    ]

    # MUST NOT patterns
    MUST_NOT_PATTERNS = [
        (r"[Yy]ou\s+must\s+not\s+([^.!?\n]+)", 0.95),
        (r"[Yy]ou\s+must\s+never\s+([^.!?\n]+)", 0.95),
        (r"[Nn]ever\s+([^.!?\n]+)", 0.9),
        (r"[Dd]o\s+not\s+([^.!?\n]+)", 0.9),
        (r"[Dd]on't\s+([^.!?\n]+)", 0.9),
        (r"[Aa]void\s+([^.!?\n]+)", 0.85),
        (r"[Rr]efrain\s+from\s+([^.!?\n]+)", 0.9),
        (r"[Yy]ou\s+should\s+not\s+([^.!?\n]+)", 0.85),
        (r"[Yy]ou\s+shouldn't\s+([^.!?\n]+)", 0.85),
        (r"[Pp]rohibited:\s*([^.!?\n]+)", 0.95),
        (r"[Ff]orbidden:\s*([^.!?\n]+)", 0.95),
        (r"[Uu]nder\s+no\s+circumstances\s+([^.!?\n]+)", 0.95),
    ]

    # FORMAT patterns
    FORMAT_PATTERNS = [
        (r"[Rr]espond\s+(?:only\s+)?(?:in|with|using)\s+([^\n.!?]+(?:format|JSON|XML|markdown|code|list))", 0.95),
        (r"[Oo]utput\s+(?:should\s+be|must\s+be|in)\s+([^\n.!?]+(?:format|JSON|XML))", 0.95),
        (r"[Ff]ormat:\s*([^\n.!?]+)", 0.9),
        (r"[Rr]eturn\s+(?:only\s+)?(?:a\s+)?([^\n.!?]*(?:JSON|XML|list|array|object))", 0.9),
        (r"[Uu]se\s+(?:the\s+following\s+)?([^\n.!?]*format)", 0.85),
        (r"[Ss]tructured\s+as\s+([^\n.!?]+)", 0.85),
    ]

    # PERSONA patterns
    PERSONA_PATTERNS = [
        (r"[Yy]ou\s+are\s+(?:a|an)\s+([^.!?\n]+(?:assistant|expert|helper|advisor|agent|bot))", 0.95),
        (r"[Yy]ou\s+are\s+([^.!?\n]+)", 0.7),
        (r"[Aa]ct\s+as\s+(?:a|an)?\s*([^.!?\n]+)", 0.9),
        (r"[Pp]retend\s+(?:to\s+be|you\s+are)\s+([^.!?\n]+)", 0.85),
        (r"[Yy]our\s+role\s+is\s+([^.!?\n]+)", 0.9),
        (r"[Aa]s\s+(?:a|an)\s+([^.!?\n]+),\s+you", 0.85),
    ]

    # SAFETY patterns
    SAFETY_PATTERNS = [
        (r"[Nn]ever\s+(?:provide|give|share|reveal)\s+([^.!?\n]*(?:harmful|dangerous|illegal|personal|private)[^.!?\n]*)", 0.95),
        (r"[Dd]o\s+not\s+(?:help|assist)\s+with\s+([^.!?\n]*(?:illegal|harmful|dangerous)[^.!?\n]*)", 0.95),
        (r"[Rr]efuse\s+(?:to|any)\s+([^.!?\n]*(?:harmful|illegal|unethical)[^.!?\n]*)", 0.95),
        (r"[Ss]afety:\s*([^.!?\n]+)", 0.95),
        (r"[Pp]rotect\s+(?:user\s+)?([^.!?\n]*(?:privacy|data|information)[^.!?\n]*)", 0.9),
        (r"[Ee]thical\s+(?:guidelines?|rules?):\s*([^.!?\n]+)", 0.95),
    ]

    # BOUNDARY patterns
    BOUNDARY_PATTERNS = [
        (r"[Oo]nly\s+(?:answer|respond\s+to|discuss)\s+([^.!?\n]+)", 0.9),
        (r"[Ll]imit(?:ed)?\s+to\s+([^.!?\n]+)", 0.9),
        (r"[Ss]cope:\s*([^.!?\n]+)", 0.9),
        (r"[Ff]ocus\s+(?:only\s+)?on\s+([^.!?\n]+)", 0.85),
        (r"[Ss]tay\s+within\s+([^.!?\n]+)", 0.9),
        (r"[Oo]utside\s+(?:of\s+)?(?:your\s+)?(?:scope|expertise):\s*([^.!?\n]+)", 0.9),
        (r"[Dd]o\s+not\s+(?:go\s+)?(?:beyond|outside)\s+([^.!?\n]+)", 0.85),
    ]

    # STYLE patterns
    STYLE_PATTERNS = [
        (r"[Bb]e\s+(concise|brief|detailed|verbose|formal|informal|friendly|professional)", 0.9),
        (r"[Uu]se\s+(?:a\s+)?(formal|informal|casual|professional)\s+(?:tone|style|language)", 0.9),
        (r"[Ww]rite\s+in\s+(?:a\s+)?(concise|detailed|formal)\s+(?:manner|way|style)", 0.9),
        (r"[Kk]eep\s+(?:responses?|answers?)\s+(short|brief|concise|detailed)", 0.9),
        (r"[Tt]one:\s*([^.!?\n]+)", 0.9),
        (r"[Ss]tyle:\s*([^.!?\n]+)", 0.9),
        (r"[Mm]aintain\s+(?:a\s+)?([^.!?\n]+)\s+(?:tone|style|voice)", 0.85),
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def _extract_with_patterns(
        self,
        text: str,
        patterns: list[tuple[str, float]],
        constraint_type: ConstraintType,
        source: str,
    ) -> list[PromptConstraint]:
        """Extract constraints using a list of patterns."""
        constraints = []
        seen = set()

        for pattern, confidence in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                content = match.group(1).strip()
                # Skip duplicates and very short content
                if content.lower() not in seen and len(content) > 3:
                    seen.add(content.lower())
                    constraints.append(PromptConstraint(
                        constraint_type=constraint_type,
                        content=content,
                        source=source,
                        confidence=confidence,
                        original_text=match.group(0),
                    ))

        return constraints

    def extract_from_text(
        self,
        text: str,
        source: str = "unknown",
    ) -> ExtractedConstraints:
        """
        Extract constraints from a single text.

        Args:
            text: The text to analyze
            source: Source identifier ("system" or "user")

        Returns:
            ExtractedConstraints with all found constraints
        """
        result = ExtractedConstraints()

        # Extract must-do constraints
        result.must_do = self._extract_with_patterns(
            text, self.MUST_DO_PATTERNS, ConstraintType.MUST_DO, source
        )

        # Extract must-not constraints
        result.must_not = self._extract_with_patterns(
            text, self.MUST_NOT_PATTERNS, ConstraintType.MUST_NOT, source
        )

        # Extract format requirements
        result.format_requirements = self._extract_with_patterns(
            text, self.FORMAT_PATTERNS, ConstraintType.FORMAT, source
        )

        # Extract persona (take first high-confidence match)
        persona_constraints = self._extract_with_patterns(
            text, self.PERSONA_PATTERNS, ConstraintType.PERSONA, source
        )
        if persona_constraints:
            # Sort by confidence and take highest
            persona_constraints.sort(key=lambda c: c.confidence, reverse=True)
            result.persona = persona_constraints[0]

        # Extract safety constraints
        result.safety_constraints = self._extract_with_patterns(
            text, self.SAFETY_PATTERNS, ConstraintType.SAFETY, source
        )

        # Extract boundaries
        result.boundaries = self._extract_with_patterns(
            text, self.BOUNDARY_PATTERNS, ConstraintType.BOUNDARY, source
        )

        # Extract style requirements
        result.style_requirements = self._extract_with_patterns(
            text, self.STYLE_PATTERNS, ConstraintType.STYLE, source
        )

        return result

    def extract(
        self,
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
    ) -> ExtractedConstraints:
        """
        Extract constraints from system prompt and/or user message.

        Args:
            system_prompt: The system prompt (if any)
            user_message: The user's message

        Returns:
            ExtractedConstraints combining constraints from both sources
        """
        result = ExtractedConstraints()

        # Extract from system prompt
        if system_prompt:
            system_constraints = self.extract_from_text(system_prompt, "system")
            result.must_do.extend(system_constraints.must_do)
            result.must_not.extend(system_constraints.must_not)
            result.format_requirements.extend(system_constraints.format_requirements)
            result.safety_constraints.extend(system_constraints.safety_constraints)
            result.boundaries.extend(system_constraints.boundaries)
            result.style_requirements.extend(system_constraints.style_requirements)
            if system_constraints.persona:
                result.persona = system_constraints.persona

        # Extract from user message
        if user_message:
            user_constraints = self.extract_from_text(user_message, "user")
            result.must_do.extend(user_constraints.must_do)
            result.must_not.extend(user_constraints.must_not)
            result.format_requirements.extend(user_constraints.format_requirements)
            result.safety_constraints.extend(user_constraints.safety_constraints)
            result.boundaries.extend(user_constraints.boundaries)
            result.style_requirements.extend(user_constraints.style_requirements)
            # User persona overrides system persona only if system had none
            if user_constraints.persona and not result.persona:
                result.persona = user_constraints.persona

        return result

    def extract_as_rules(
        self,
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Extract constraints and convert to rule format for Datalog.

        Returns rules in format suitable for DatalogEngine.

        Args:
            system_prompt: The system prompt
            user_message: The user message

        Returns:
            List of rule dictionaries
        """
        constraints = self.extract(system_prompt, user_message)
        rules = []

        for constraint in constraints.all_constraints():
            rules.append({
                "rule_type": constraint.constraint_type.value,
                "content": constraint.content,
                "source": constraint.source,
                "confidence": constraint.confidence,
            })

        return rules
