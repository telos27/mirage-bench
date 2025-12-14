"""
Heuristic-based input fact extraction.

Parses structured input directly using regex patterns - no LLM needed.
Supports domain-specific plugins for web, code, chat, and custom domains.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from agent_verifier.schemas import InputFacts
from agent_verifier.extractors.base import BaseInputExtractor


class DomainPlugin(ABC):
    """
    Abstract base for domain-specific extraction plugins.

    Plugins provide regex patterns and extraction logic for specific domains
    like web browsing, code editing, chat, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name for identification."""
        pass

    @abstractmethod
    def extract_elements(self, text: str) -> list[str]:
        """Extract visible elements/entities from text."""
        pass

    @abstractmethod
    def extract_state(self, text: str) -> dict[str, Any]:
        """Extract state information from text."""
        pass

    def extract_errors(self, text: str) -> list[str]:
        """Extract error messages. Override for custom patterns."""
        return []


class WebBrowserPlugin(DomainPlugin):
    """
    Plugin for web browser agent scenarios.

    Parses AXTree elements, URLs, page titles, etc.
    Based on MIRAGE-Bench heuristic extractor.
    """

    # Patterns for AXTree elements
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

    @property
    def name(self) -> str:
        return "web_browser"

    def extract_elements(self, text: str) -> list[str]:
        """Extract visible elements from AXTree."""
        elements = []
        seen = set()

        for pattern in self.ELEMENT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    elem_text = match[1].strip() if len(match) >= 2 else match[0].strip()
                else:
                    elem_text = match.strip()

                # Skip empty, very short, or icon-only elements
                if elem_text and len(elem_text) > 1 and not elem_text.startswith('\\ue'):
                    normalized = elem_text.lower().strip()
                    if normalized not in seen and len(normalized) < 200:
                        seen.add(normalized)
                        elements.append(elem_text)

        return elements

    def extract_state(self, text: str) -> dict[str, Any]:
        """Extract browser state (URL, title, focused element)."""
        state = {}

        # Extract tab/page info
        tab_match = re.search(r"Tab\s+\d+\s*\([^)]*\):\s*\n\s*Title:\s*([^\n]+)", text)
        if tab_match:
            state["page_title"] = tab_match.group(1).strip()

        # Extract URL
        url_match = re.search(r"URL:\s*(http[^\s\n]+)", text)
        if url_match:
            state["url"] = url_match.group(1).strip()

        # Extract focused element
        focused_match = re.search(r"Focused element:\s*\n\s*bid='([^']+)'", text)
        if focused_match:
            state["focused_element"] = focused_match.group(1)

        return state

    def extract_errors(self, text: str) -> list[str]:
        """Extract error messages from browser context."""
        errors = []
        seen_normalized = set()

        # Tab-related errors
        tab_error_matches = re.findall(
            r"tab\s+'([^']*(?:invalid|error|changed)[^']*)'",
            text,
            re.IGNORECASE
        )
        for match in tab_error_matches:
            if match and match.lower() not in seen_normalized:
                errors.append(match)
                seen_normalized.add(match.lower())
                if len(errors) >= 3:
                    break

        return errors


class CodeEditorPlugin(DomainPlugin):
    """
    Plugin for code editor agent scenarios.

    Parses file paths, line numbers, symbols, diagnostics, etc.
    """

    @property
    def name(self) -> str:
        return "code_editor"

    def extract_elements(self, text: str) -> list[str]:
        """Extract code elements (files, functions, symbols)."""
        elements = []
        seen = set()

        # File paths
        file_matches = re.findall(r"(?:file|path):\s*([^\s\n]+\.\w+)", text, re.IGNORECASE)
        for match in file_matches:
            if match.lower() not in seen:
                elements.append(f"file: {match}")
                seen.add(match.lower())

        # Function/class definitions
        def_matches = re.findall(r"(?:def|class|function)\s+(\w+)", text)
        for match in def_matches:
            if match.lower() not in seen:
                elements.append(f"symbol: {match}")
                seen.add(match.lower())

        # Line numbers
        line_matches = re.findall(r"line\s+(\d+)", text, re.IGNORECASE)
        for line in line_matches[:10]:  # Limit
            elements.append(f"line: {line}")

        return elements

    def extract_state(self, text: str) -> dict[str, Any]:
        """Extract editor state (current file, cursor position)."""
        state = {}

        # Current file
        current_file = re.search(r"(?:current|open)\s+file:\s*([^\s\n]+)", text, re.IGNORECASE)
        if current_file:
            state["current_file"] = current_file.group(1)

        # Cursor/line position
        cursor_match = re.search(r"(?:cursor|position).*?line\s+(\d+)", text, re.IGNORECASE)
        if cursor_match:
            state["cursor_line"] = int(cursor_match.group(1))

        return state

    def extract_errors(self, text: str) -> list[str]:
        """Extract code errors/diagnostics."""
        errors = []

        # Compiler/lint errors
        error_patterns = [
            r"error\[?\w*\]?:\s*([^\n]+)",
            r"warning:\s*([^\n]+)",
            r"(?:syntax|type)\s+error:\s*([^\n]+)",
        ]

        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            errors.extend([m.strip() for m in matches[:5]])

        return errors[:10]


class ChatPlugin(DomainPlugin):
    """
    Plugin for chat/conversation scenarios.

    Parses message history, user mentions, etc.
    """

    @property
    def name(self) -> str:
        return "chat"

    def extract_elements(self, text: str) -> list[str]:
        """Extract conversation elements (speakers, topics)."""
        elements = []

        # Speaker/role mentions
        role_matches = re.findall(r"(?:user|assistant|system|human|ai):\s*", text, re.IGNORECASE)
        unique_roles = set(r.lower().strip(": ") for r in role_matches)
        elements.extend([f"role: {r}" for r in unique_roles])

        # Quoted content
        quotes = re.findall(r'"([^"]{5,100})"', text)
        elements.extend([f"quote: {q[:50]}..." if len(q) > 50 else f"quote: {q}" for q in quotes[:5]])

        return elements

    def extract_state(self, text: str) -> dict[str, Any]:
        """Extract conversation state."""
        state = {}

        # Count messages
        message_count = len(re.findall(r"(?:user|assistant|human|ai):", text, re.IGNORECASE))
        if message_count:
            state["message_count"] = message_count

        return state


class HeuristicInputExtractor(BaseInputExtractor):
    """
    Heuristic-based input fact extractor.

    Uses regex patterns to extract facts from input text.
    Supports domain-specific plugins for specialized extraction.

    Example:
        extractor = HeuristicInputExtractor()
        extractor.add_plugin(WebBrowserPlugin())
        facts = extractor.extract(input_text)
    """

    # Common error patterns
    ERROR_PATTERNS = [
        r"[Ee]rror[:\s]+([^\n]+)",
        r"[Ff]ailed[:\s]+([^\n]+)",
        r"[Tt]imeout[:\s]+([^\n]+)",
        r"invalid\s+data[^\n]*",
        r"[Pp]lease\s+resolve[^\n]*",
        r"[Cc]annot\s+[^\n]+",
        r"[Uu]nable\s+to\s+[^\n]+",
    ]

    # Goal extraction patterns
    GOAL_PATTERNS = [
        r"##\s*Goal:\s*\n([^\n#]+)",
        r"Goal:\s*([^\n]+)",
        r"Task Goal:\s*([^\n]+)",
        r"'goal':\s*'([^']+)'",
        r"Objective:\s*([^\n]+)",
        r"Task:\s*([^\n]+)",
    ]

    # Constraint patterns
    CONSTRAINT_PATTERNS = [
        r"[Mm]ust\s+([^\n.]+)",
        r"[Ss]hould\s+not\s+([^\n.]+)",
        r"[Dd]o\s+not\s+([^\n.]+)",
        r"[Rr]equired:\s*([^\n]+)",
        r"[Cc]onstraint:\s*([^\n]+)",
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.plugins: list[DomainPlugin] = []

    def add_plugin(self, plugin: DomainPlugin) -> "HeuristicInputExtractor":
        """Add a domain plugin. Returns self for chaining."""
        self.plugins.append(plugin)
        return self

    def extract_goal(self, text: str) -> Optional[str]:
        """Extract task goal from input."""
        for pattern in self.GOAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def extract_errors(self, text: str) -> list[str]:
        """Extract error messages with deduplication."""
        errors = []
        seen_normalized = set()

        # First try plugin-specific errors
        for plugin in self.plugins:
            plugin_errors = plugin.extract_errors(text)
            for err in plugin_errors:
                norm = err.lower()[:50]
                if norm not in seen_normalized:
                    errors.append(err)
                    seen_normalized.add(norm)

        # Then common patterns
        for pattern in self.ERROR_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                error_text = match.strip() if isinstance(match, str) else match[0].strip()
                if error_text and len(error_text) > 5:
                    norm = error_text.lower()[:50]
                    if norm not in seen_normalized:
                        errors.append(error_text)
                        seen_normalized.add(norm)
                        if len(errors) >= 10:
                            return errors

        return errors

    def extract_action_history(self, text: str) -> list[dict[str, Any]]:
        """Extract action history with outcomes."""
        history = []

        # Step-based format
        step_pattern = r"##\s*step\s+(\d+)\s*\n(.*?)(?=##\s*step\s+\d+|#\s*Action\s+space:|$)"
        step_matches = re.findall(step_pattern, text, re.DOTALL | re.IGNORECASE)

        for step_num, step_content in step_matches:
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

                history.append({
                    "step": int(step_num),
                    "action": action,
                    "outcome": outcome,
                })

        return history

    def extract_constraints(self, text: str) -> list[str]:
        """Extract explicit constraints from input."""
        constraints = []
        seen = set()

        for pattern in self.CONSTRAINT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                constraint = match.strip()
                if constraint and constraint.lower() not in seen:
                    constraints.append(constraint)
                    seen.add(constraint.lower())

        return constraints

    def extract_format_requirements(self, text: str) -> Optional[str]:
        """Extract output format requirements."""
        # JSON format
        if re.search(r"(?:respond|output|return)\s+(?:in\s+)?json", text, re.IGNORECASE):
            return "json"
        # Code format
        if re.search(r"(?:write|provide|generate)\s+(?:the\s+)?code", text, re.IGNORECASE):
            return "code"
        # List format
        if re.search(r"(?:list|enumerate|bullet)", text, re.IGNORECASE):
            return "list"

        return None

    def extract(self, text: str, **kwargs: Any) -> InputFacts:
        """
        Extract all facts from input text.

        Args:
            text: Input text to analyze
            **kwargs: Additional parameters (domain hints, etc.)

        Returns:
            InputFacts with extracted information
        """
        # Collect elements from all plugins
        visible_elements = []
        for plugin in self.plugins:
            visible_elements.extend(plugin.extract_elements(text))

        # Collect state from all plugins
        state_info = {}
        for plugin in self.plugins:
            state_info.update(plugin.extract_state(text))

        return InputFacts(
            task_goal=self.extract_goal(text),
            visible_elements=visible_elements,
            error_messages=self.extract_errors(text),
            action_history=self.extract_action_history(text),
            state_info=state_info,
            constraints=self.extract_constraints(text),
            format_requirements=self.extract_format_requirements(text),
        )


# Pre-configured extractors for common domains
def create_web_extractor(logger: Optional[logging.Logger] = None) -> HeuristicInputExtractor:
    """Create an extractor configured for web browser agents."""
    return HeuristicInputExtractor(logger).add_plugin(WebBrowserPlugin())


def create_code_extractor(logger: Optional[logging.Logger] = None) -> HeuristicInputExtractor:
    """Create an extractor configured for code editor agents."""
    return HeuristicInputExtractor(logger).add_plugin(CodeEditorPlugin())


def create_chat_extractor(logger: Optional[logging.Logger] = None) -> HeuristicInputExtractor:
    """Create an extractor configured for chat agents."""
    return HeuristicInputExtractor(logger).add_plugin(ChatPlugin())
