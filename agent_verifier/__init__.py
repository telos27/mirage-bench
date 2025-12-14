"""
Agent Verifier - A general-purpose AI agent verification system.

This package provides a 6-layer verification architecture for detecting
hallucinations and policy violations in AI agent outputs.
"""

__version__ = "0.1.0"

from .schemas.request import VerificationRequest
from .schemas.result import VerificationResult, Violation, ReasoningStep
from .layers.base_layer import BaseLayer, LayerResult

__all__ = [
    "VerificationRequest",
    "VerificationResult",
    "Violation",
    "ReasoningStep",
    "BaseLayer",
    "LayerResult",
]
