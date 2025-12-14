"""Base extractor interface for fact extraction."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from agent_verifier.schemas import InputFacts, OutputFacts, ExtractedFacts


class BaseInputExtractor(ABC):
    """
    Abstract base class for input fact extractors.

    Input extractors analyze the prompt/input to identify:
    - Task goals and requirements
    - Visible elements/context
    - Error messages
    - Action history
    - State information
    - Constraints
    """

    @abstractmethod
    def extract(self, text: str, **kwargs: Any) -> InputFacts:
        """
        Extract facts from input text.

        Args:
            text: The input/prompt text to analyze
            **kwargs: Additional extraction parameters

        Returns:
            InputFacts containing extracted information
        """
        pass


class BaseOutputExtractor(ABC):
    """
    Abstract base class for output fact extractors.

    Output extractors analyze LLM output to identify:
    - Stated observations
    - Reasoning steps
    - Action targets
    - References made
    - Claims
    """

    @abstractmethod
    def extract(self, text: str, **kwargs: Any) -> OutputFacts:
        """
        Extract facts from output text.

        Args:
            text: The LLM output text to analyze
            **kwargs: Additional extraction parameters

        Returns:
            OutputFacts containing extracted information
        """
        pass


class BaseCombinedExtractor(ABC):
    """
    Abstract base class for combined extractors.

    Combined extractors handle both input and output extraction,
    useful when extraction benefits from seeing both together.
    """

    @abstractmethod
    def extract(
        self,
        prompt: str,
        output: str,
        **kwargs: Any
    ) -> ExtractedFacts:
        """
        Extract facts from both prompt and output.

        Args:
            prompt: The input/prompt text
            output: The LLM output text
            **kwargs: Additional extraction parameters

        Returns:
            ExtractedFacts containing input and output facts
        """
        pass
