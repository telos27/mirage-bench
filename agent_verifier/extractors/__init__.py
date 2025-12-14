"""Fact extraction components for agent_verifier."""

from agent_verifier.extractors.base import (
    BaseInputExtractor,
    BaseOutputExtractor,
    BaseCombinedExtractor,
)
from agent_verifier.extractors.heuristic_input import (
    HeuristicInputExtractor,
    DomainPlugin,
    WebBrowserPlugin,
    CodeEditorPlugin,
    ChatPlugin,
    create_web_extractor,
    create_code_extractor,
    create_chat_extractor,
)
from agent_verifier.extractors.heuristic_output import (
    HeuristicOutputExtractor,
    HeuristicCombinedExtractor,
)
from agent_verifier.extractors.prompt_constraints import (
    PromptConstraintExtractor,
    PromptConstraint,
    ExtractedConstraints,
    ConstraintType,
)

__all__ = [
    # Base classes
    "BaseInputExtractor",
    "BaseOutputExtractor",
    "BaseCombinedExtractor",
    # Heuristic extractors
    "HeuristicInputExtractor",
    "HeuristicOutputExtractor",
    "HeuristicCombinedExtractor",
    # Plugins
    "DomainPlugin",
    "WebBrowserPlugin",
    "CodeEditorPlugin",
    "ChatPlugin",
    # Factory functions
    "create_web_extractor",
    "create_code_extractor",
    "create_chat_extractor",
    # Prompt constraint extraction
    "PromptConstraintExtractor",
    "PromptConstraint",
    "ExtractedConstraints",
    "ConstraintType",
]
