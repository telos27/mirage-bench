"""
Agent Verifier - A general-purpose AI agent verification system.

This package provides a 6-layer verification architecture for detecting
hallucinations and policy violations in AI agent outputs.

Layers:
    1. Common Knowledge - Universal truths and consistency
    2. Domain Best Practices - Domain-specific rules (coding, customer service, etc.)
    3. Business Policies - Organization-level policies and compliance
    4. User Preferences - Per-user personalization settings
    5. Session History - Multi-turn context and consistency
    6. Prompt Constraints - Prompt-specific instruction enforcement

Quick Start:
    from agent_verifier import quick_verify

    result = quick_verify(
        prompt="Write hello world in Python",
        output="print('Hello, World!')",
    )
    print(result.verdict)  # "pass" or "fail"

Full Engine:
    from agent_verifier import create_full_engine, VerificationRequest

    engine = create_full_engine()
    request = VerificationRequest(
        request_id="req_001",
        deployment_id="my-app",
        prompt="Help me with coding",
        llm_output="Here is the code...",
        llm_model="gpt-4",
    )
    result = engine.verify(request)
"""

__version__ = "0.1.0"

# Schemas
from .schemas.request import VerificationRequest
from .schemas.result import VerificationResult, Violation, ReasoningStep, Severity
from .schemas.session import Session, Turn, EstablishedFact

# Layers
from .layers.base_layer import BaseLayer, LayerResult
from .layers.layer1_common import CommonKnowledgeLayer, ExtractedFacts
from .layers.layer2_domain import DomainBestPracticesLayer, DomainConfig
from .layers.layer3_business import BusinessPoliciesLayer, PolicyConfig
from .layers.layer4_preferences import UserPreferencesLayer, UserPreferenceSet
from .layers.layer5_session import SessionHistoryLayer, SessionState
from .layers.layer6_prompt import PromptConstraintsLayer

# Engine
from .engine.verifier import (
    VerificationEngine,
    EngineConfig,
    create_engine,
    create_full_engine,
    create_lightweight_engine,
    create_coding_engine,
    quick_verify,
)

# Reasoning
from .reasoning.datalog_engine import DatalogEngine, DatalogResult, check_souffle_installed

__all__ = [
    # Schemas
    "VerificationRequest",
    "VerificationResult",
    "Violation",
    "ReasoningStep",
    "Severity",
    "Session",
    "Turn",
    "EstablishedFact",
    # Layers
    "BaseLayer",
    "LayerResult",
    "CommonKnowledgeLayer",
    "ExtractedFacts",
    "DomainBestPracticesLayer",
    "DomainConfig",
    "BusinessPoliciesLayer",
    "PolicyConfig",
    "UserPreferencesLayer",
    "UserPreferenceSet",
    "SessionHistoryLayer",
    "SessionState",
    "PromptConstraintsLayer",
    # Engine
    "VerificationEngine",
    "EngineConfig",
    "create_engine",
    "create_full_engine",
    "create_lightweight_engine",
    "create_coding_engine",
    "quick_verify",
    # Reasoning
    "DatalogEngine",
    "DatalogResult",
    "check_souffle_installed",
]
