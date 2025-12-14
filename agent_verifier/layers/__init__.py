"""Verification layer implementations."""

from .base_layer import BaseLayer, LayerResult
from .layer1_common import CommonKnowledgeLayer, ExtractedFacts
from .layer2_domain import DomainBestPracticesLayer, DomainConfig, DOMAIN_CONFIGS

__all__ = [
    # Base
    "BaseLayer",
    "LayerResult",
    # Layer 1: Common Knowledge
    "CommonKnowledgeLayer",
    "ExtractedFacts",
    # Layer 2: Domain Best Practices
    "DomainBestPracticesLayer",
    "DomainConfig",
    "DOMAIN_CONFIGS",
]
