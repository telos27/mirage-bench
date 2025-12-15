"""Verification layer implementations."""

from .base_layer import BaseLayer, LayerResult
from .layer1_common import CommonKnowledgeLayer, ExtractedFacts
from .layer2_domain import DomainBestPracticesLayer, DomainConfig, DOMAIN_CONFIGS
from .layer3_business import (
    BusinessPoliciesLayer,
    PolicyConfig,
    create_content_policy,
    create_privacy_policy,
    create_compliance_policy,
)
from .layer4_preferences import (
    UserPreferencesLayer,
    UserPreferenceSet,
    ResponseStyle,
    ResponseLength,
    ResponseFormat,
    Tone,
    ExpertiseLevel,
    create_developer_preferences,
    create_beginner_preferences,
    create_executive_preferences,
)

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
    # Layer 3: Business Policies
    "BusinessPoliciesLayer",
    "PolicyConfig",
    "create_content_policy",
    "create_privacy_policy",
    "create_compliance_policy",
    # Layer 4: User Preferences
    "UserPreferencesLayer",
    "UserPreferenceSet",
    "ResponseStyle",
    "ResponseLength",
    "ResponseFormat",
    "Tone",
    "ExpertiseLevel",
    "create_developer_preferences",
    "create_beginner_preferences",
    "create_executive_preferences",
]
