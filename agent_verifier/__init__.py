"""
Agent Verifier - A general-purpose AI agent verification system.

This package provides a 6-layer verification architecture for detecting
hallucinations and policy violations in AI agent outputs.
"""

__version__ = "0.1.0"

from .schemas.request import VerificationRequest
from .schemas.result import VerificationResult, Violation, ReasoningStep, Severity
from .layers.base_layer import BaseLayer, LayerResult
from .layers.layer1_common import CommonKnowledgeLayer, ExtractedFacts
from .engine.verifier import VerificationEngine, EngineConfig
from .reasoning.datalog_engine import DatalogEngine, DatalogResult, check_souffle_installed

__all__ = [
    # Schemas
    "VerificationRequest",
    "VerificationResult",
    "Violation",
    "ReasoningStep",
    "Severity",
    # Layers
    "BaseLayer",
    "LayerResult",
    "CommonKnowledgeLayer",
    "ExtractedFacts",
    # Engine
    "VerificationEngine",
    "EngineConfig",
    # Reasoning
    "DatalogEngine",
    "DatalogResult",
    "check_souffle_installed",
]
