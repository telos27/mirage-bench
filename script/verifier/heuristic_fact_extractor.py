#!/usr/bin/env python3
"""
Heuristic-based fact extraction for generic hallucination detection.

Parses AXTree and structured input directly instead of using LLM.
This is faster, cheaper (FREE), and more complete than LLM extraction.

Use this for input facts, keep LLM for output facts (semantic understanding).
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class HeuristicInputFacts:
    """Facts extracted from agent input using heuristics."""
    task_goal: str = ""
    visible_elements: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    state_info: List[str] = field(default_factory=list)
    action_history: List[Tuple[str, str]] = field(default_factory=list)  # (action, outcome)
    important_facts: List[str] = field(default_factory=list)


class HeuristicFactExtractor:
    """
    Extracts facts from agent input using heuristics (no LLM needed).

    Parses:
    - AXTree elements (links, buttons, text, tables, etc.)
    - Error messages
    - Action history with outcomes
    - Task goal
    """

    # Patterns for extracting elements from AXTree
    ELEMENT_PATTERNS = [
        # Links: [123] link 'Text', clickable
        r"\[(\d+)\]\s+link\s+'([^']+)'",
        # Buttons: [123] button 'Text', clickable
        r"\[(\d+)\]\s+button\s+'([^']+)'",
        # StaticText: StaticText 'Text'
        r"StaticText\s+'([^']+)'",
        # Tabs: [123] tab 'Text'
        r"\[(\d+)\]\s+tab\s+'([^']+)'",
        # Cells: [123] cell 'Text'
        r"\[(\d+)\]\s+cell\s+'([^']+)'",
        # Textbox: [123] textbox 'placeholder'
        r"\[(\d+)\]\s+textbox\s+'([^']*)'",
        # Headings: [123] heading 'Text'
        r"\[(\d+)\]\s+heading\s+'([^']+)'",
        # Menu items: [123] menuitem 'Text'
        r"\[(\d+)\]\s+menuitem\s+'([^']+)'",
        # List items: [123] listitem
        r"\[(\d+)\]\s+listitem",
        # Images: [123] image 'alt text'
        r"\[(\d+)\]\s+image\s+'([^']+)'",
    ]

    # Patterns for error messages
    ERROR_PATTERNS = [
        r"[Ee]rror[:\s]+([^\n]+)",
        r"[Ff]ailed[:\s]+([^\n]+)",
        r"[Tt]imeout[:\s]+([^\n]+)",
        r"invalid\s+data[^\n]*",
        r"[Pp]lease\s+resolve[^\n]*",
        r"[Cc]annot\s+[^\n]+",
        r"[Uu]nable\s+to\s+[^\n]+",
    ]

    # Patterns for action history
    ACTION_HISTORY_PATTERN = r"##\s*step\s+(\d+)\s*\n.*?<action>\s*([^<]+)\s*</action>(?:.*?(?:Error|error|TimeoutError|failed)[^\n]*)?(?:\n|$)"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def extract_visible_elements(self, text: str) -> List[str]:
        """Extract visible elements from AXTree."""
        elements = []
        seen = set()

        for pattern in self.ELEMENT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Pattern with bid and text
                    if len(match) >= 2:
                        elem_text = match[1].strip()
                    else:
                        elem_text = match[0].strip()
                else:
                    elem_text = match.strip()

                # Skip empty, very short, or icon-only elements
                if elem_text and len(elem_text) > 1 and not elem_text.startswith('\\ue'):
                    # Normalize and dedupe
                    normalized = elem_text.lower().strip()
                    if normalized not in seen and len(normalized) < 200:
                        seen.add(normalized)
                        elements.append(elem_text)

        return elements

    def extract_error_messages(self, text: str) -> List[str]:
        """Extract error messages from text with deduplication."""
        errors = []
        seen_normalized = set()

        def is_duplicate(msg: str) -> bool:
            """Check if message is a duplicate using key phrase matching."""
            msg_lower = msg.lower()
            # Extract key phrases for comparison
            key_phrases = []
            if "invalid data" in msg_lower:
                key_phrases.append("invalid data")
            if "please resolve" in msg_lower:
                key_phrases.append("please resolve")
            if "timeout" in msg_lower:
                key_phrases.append("timeout")
            if "error" in msg_lower:
                key_phrases.append("error")

            # Create a normalized key
            norm_key = " ".join(sorted(key_phrases)) if key_phrases else msg_lower[:50]
            if norm_key in seen_normalized:
                return True
            seen_normalized.add(norm_key)
            return False

        # Look for tab-related errors first (most specific)
        tab_error_matches = re.findall(
            r"tab\s+'([^']*(?:invalid|error|changed)[^']*)'",
            text,
            re.IGNORECASE
        )
        for match in tab_error_matches:
            if match and not is_duplicate(match):
                errors.append(match)
                if len(errors) >= 3:  # Limit to avoid duplicates
                    break

        # If no tab errors found, look for general error patterns
        if not errors:
            for pattern in self.ERROR_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    error_text = match.strip() if isinstance(match, str) else match[0].strip()
                    if error_text and len(error_text) > 5 and not is_duplicate(error_text):
                        errors.append(error_text)
                        if len(errors) >= 5:  # Limit total errors
                            break
                if len(errors) >= 5:
                    break

        return errors

    def extract_action_history(self, text: str) -> List[Tuple[str, str]]:
        """Extract action history with outcomes."""
        history = []

        # Find all step sections
        step_pattern = r"##\s*step\s+(\d+)\s*\n(.*?)(?=##\s*step\s+\d+|#\s*Action\s+space:|$)"
        step_matches = re.findall(step_pattern, text, re.DOTALL | re.IGNORECASE)

        for step_num, step_content in step_matches:
            # Extract action
            action_match = re.search(r"<action>\s*([^<]+)\s*</action>", step_content)
            if action_match:
                action = action_match.group(1).strip()

                # Determine outcome
                outcome = "unknown"
                if re.search(r"TimeoutError|Timeout", step_content):
                    outcome = "timeout"
                elif re.search(r"Error from previous action|error", step_content, re.IGNORECASE):
                    outcome = "failed"
                elif re.search(r"success", step_content, re.IGNORECASE):
                    outcome = "success"

                history.append((action, outcome))

        return history

    def extract_task_goal(self, text: str) -> str:
        """Extract task goal from input."""
        # Look for goal section
        goal_patterns = [
            r"##\s*Goal:\s*\n([^\n#]+)",
            r"Goal:\s*([^\n]+)",
            r"Task Goal:\s*([^\n]+)",
            r"'goal':\s*'([^']+)'",
        ]

        for pattern in goal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def extract_state_info(self, text: str) -> List[str]:
        """Extract current state information."""
        state_info = []

        # Extract tab info
        tab_match = re.search(r"Tab\s+\d+\s*\([^)]*\):\s*\n\s*Title:\s*([^\n]+)", text)
        if tab_match:
            state_info.append(f"Page: {tab_match.group(1).strip()}")

        # Extract URL
        url_match = re.search(r"URL:\s*(http[^\s\n]+)", text)
        if url_match:
            state_info.append(f"URL: {url_match.group(1).strip()}")

        # Extract focused element
        focused_match = re.search(r"Focused element:\s*\n\s*bid='([^']+)'", text)
        if focused_match:
            state_info.append(f"Focused: {focused_match.group(1)}")

        return state_info

    def extract(self, input_text: str) -> HeuristicInputFacts:
        """
        Extract all facts from input text using heuristics.

        Args:
            input_text: Raw input text containing AXTree, history, etc.

        Returns:
            HeuristicInputFacts with extracted information
        """
        return HeuristicInputFacts(
            task_goal=self.extract_task_goal(input_text),
            visible_elements=self.extract_visible_elements(input_text),
            error_messages=self.extract_error_messages(input_text),
            state_info=self.extract_state_info(input_text),
            action_history=self.extract_action_history(input_text),
            important_facts=[],  # Can be extended
        )


def extract_output_facts_heuristic(thinking: str, action: str) -> Dict[str, Any]:
    """
    Extract basic output facts using heuristics.

    For more nuanced extraction, use LLM.
    """
    # Extract action type and target
    action_match = re.match(r"(\w+)\s*\(\s*['\"]?([^'\")\s]+)['\"]?\s*\)", action)
    action_type = action_match.group(1) if action_match else "unknown"
    action_target = action_match.group(2) if action_match else ""

    # Extract references from thinking (quoted strings, element names)
    references = []
    # Quoted strings
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", thinking)
    references.extend([q for q in quoted if len(q) > 2 and len(q) < 50])

    # "the X" patterns
    the_patterns = re.findall(r"the\s+([a-zA-Z][a-zA-Z\s]{2,30}?)(?:\s+(?:button|link|page|section|tab|element|field))?", thinking, re.IGNORECASE)
    references.extend([p.strip() for p in the_patterns if len(p.strip()) > 2])

    return {
        "action_type": action_type,
        "action_target": action_target,
        "references_made": list(set(references))[:20],  # Dedupe and limit
        "stated_observations": [],  # Would need LLM for this
        "reasoning_steps": [],  # Would need LLM for this
    }
