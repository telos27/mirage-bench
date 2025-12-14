"""
Heuristic-based output fact extraction.

Parses LLM output to extract references, actions, and basic reasoning.
For semantic understanding (claims, observations), use LLM-based extraction.
"""

import re
import logging
from typing import Any, Optional

from agent_verifier.schemas import OutputFacts
from agent_verifier.extractors.base import BaseOutputExtractor


class HeuristicOutputExtractor(BaseOutputExtractor):
    """
    Heuristic-based output fact extractor.

    Extracts structural information from LLM output using regex.
    Best for:
    - Action extraction (click, type, scroll, etc.)
    - Reference extraction (quoted strings, element names)
    - Basic reasoning patterns

    For nuanced semantic extraction (claims, observations),
    consider using an LLM-based extractor.

    Example:
        extractor = HeuristicOutputExtractor()
        facts = extractor.extract(llm_output)
    """

    # Action patterns (agent-style outputs)
    ACTION_PATTERNS = [
        # Action tags: <action>click("element")</action> or <action>type("el", "val")</action>
        r"<action>\s*(\w+)\s*\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*['\"][^'\"]*['\"])?\s*\)\s*</action>",
        # Function-style: click("element")
        r"(\w+)\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        # Command style: CLICK element
        r"^(CLICK|TYPE|SCROLL|PRESS|GOTO|WAIT|SUBMIT)\s+(.+)$",
    ]

    # Reference patterns
    REFERENCE_PATTERNS = [
        # Quoted strings
        r"['\"]([^'\"]{3,50})['\"]",
        # "the X button/link/element"
        r"the\s+([a-zA-Z][a-zA-Z\s]{2,25}?)\s+(?:button|link|element|field|tab|page)",
        # Element IDs
        r"(?:element|id|bid)\s*[=:]\s*['\"]?(\w+)['\"]?",
    ]

    # Reasoning indicators
    REASONING_PATTERNS = [
        r"(?:because|since|therefore|thus|so)\s+([^.!?\n]+)",
        r"(?:I\s+(?:will|should|need to|can))\s+([^.!?\n]+)",
        r"(?:First|Next|Then|Finally)[,:]?\s+([^.!?\n]+)",
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def extract_action(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract action type and target from output.

        Returns:
            Tuple of (action_type, action_target) or (None, None)
        """
        for pattern in self.ACTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                action_type = match.group(1).lower()
                action_target = match.group(2).strip() if len(match.groups()) > 1 else None
                return action_type, action_target

        return None, None

    def extract_references(self, text: str) -> list[str]:
        """Extract things referenced in the output."""
        references = []
        seen = set()

        for pattern in self.REFERENCE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                ref = match.strip()
                # Skip very short, common words, or duplicates
                if ref and len(ref) > 2 and ref.lower() not in seen:
                    if ref.lower() not in {"the", "this", "that", "and", "for"}:
                        references.append(ref)
                        seen.add(ref.lower())

        return references[:20]  # Limit

    def extract_reasoning_steps(self, text: str) -> list[str]:
        """Extract reasoning steps from output."""
        steps = []
        seen = set()

        for pattern in self.REASONING_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                step = match.strip()
                if step and len(step) > 10 and step.lower() not in seen:
                    steps.append(step)
                    seen.add(step.lower())

        return steps[:10]  # Limit

    def extract_stated_observations(self, text: str) -> list[str]:
        """
        Extract stated observations (what the agent claims to see/observe).

        Note: This is heuristic-based and may miss nuanced observations.
        For complete extraction, use LLM-based extraction.
        """
        observations = []

        # "I see/notice/observe X"
        see_patterns = [
            r"I\s+(?:see|notice|observe|find|found)\s+([^.!?\n]+)",
            r"(?:there\s+is|there\s+are)\s+([^.!?\n]+)",
            r"(?:shows?|displays?|contains?)\s+([^.!?\n]+)",
        ]

        for pattern in see_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                obs = match.strip()
                if obs and len(obs) > 5:
                    observations.append(obs)

        return observations[:10]

    def extract_claims(self, text: str) -> list[str]:
        """
        Extract factual claims from output.

        Note: This is heuristic-based and limited.
        For complete claim extraction, use LLM-based extraction.
        """
        claims = []

        # Assertion patterns
        claim_patterns = [
            r"(?:is|are|was|were)\s+(?:a|an|the)?\s*([^.!?\n]{5,50})",
            r"(?:has|have|had)\s+([^.!?\n]{5,50})",
        ]

        # This is intentionally conservative - most claims need semantic understanding
        return claims[:5]

    def detect_format(self, text: str) -> Optional[str]:
        """Detect the format of the output."""
        # JSON
        if re.search(r"^\s*\{[\s\S]*\}\s*$", text.strip()):
            return "json"
        # Code block
        if re.search(r"```\w*\n", text):
            return "code"
        # Bullet list
        if re.search(r"^\s*[-*]\s+", text, re.MULTILINE):
            return "list"
        # Numbered list
        if re.search(r"^\s*\d+[.)]\s+", text, re.MULTILINE):
            return "numbered_list"

        return None

    def extract(self, text: str, **kwargs: Any) -> OutputFacts:
        """
        Extract facts from LLM output.

        Args:
            text: The LLM output text
            **kwargs: Additional parameters

        Returns:
            OutputFacts with extracted information
        """
        action_type, action_target = self.extract_action(text)

        return OutputFacts(
            stated_observations=self.extract_stated_observations(text),
            reasoning_steps=self.extract_reasoning_steps(text),
            action_target=action_target,
            references=self.extract_references(text),
            claims=self.extract_claims(text),
            format_used=self.detect_format(text),
        )


class HeuristicCombinedExtractor:
    """
    Combines heuristic input and output extraction.

    Convenience class that wraps both extractors.
    """

    def __init__(
        self,
        input_extractor: "HeuristicInputExtractor",
        output_extractor: Optional[HeuristicOutputExtractor] = None,
        logger: Optional[logging.Logger] = None,
    ):
        from agent_verifier.extractors.heuristic_input import HeuristicInputExtractor
        self.input_extractor = input_extractor
        self.output_extractor = output_extractor or HeuristicOutputExtractor(logger)
        self.logger = logger or logging.getLogger(__name__)

    def extract(self, prompt: str, output: str, **kwargs: Any):
        """Extract facts from both prompt and output."""
        from agent_verifier.schemas import ExtractedFacts

        return ExtractedFacts(
            input_facts=self.input_extractor.extract(prompt, **kwargs),
            output_facts=self.output_extractor.extract(output, **kwargs),
            metadata={"extractor": "heuristic"},
        )
