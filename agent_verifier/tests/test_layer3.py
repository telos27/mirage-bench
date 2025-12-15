"""Tests for Layer 3: Business Policies."""

import pytest
from ..layers.layer3_business import (
    BusinessPoliciesLayer,
    PolicyConfig,
    create_content_policy,
    create_privacy_policy,
    create_compliance_policy,
)
from ..schemas.request import VerificationRequest


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def layer():
    """Create a basic layer without storage."""
    return BusinessPoliciesLayer()


@pytest.fixture
def content_policy():
    """Create a content policy for testing."""
    return PolicyConfig(
        policy_id="test_content",
        name="Test Content Policy",
        description="Test policy for content restrictions",
        policy_type="constraint",
        forbidden_words=["prohibited", "banned", "secret"],
        required_words=["disclaimer"],
        max_length=1000,
        min_length=10,
    )


@pytest.fixture
def privacy_policy():
    """Create a privacy policy for testing."""
    return PolicyConfig(
        policy_id="test_privacy",
        name="Test Privacy Policy",
        description="Test policy for privacy",
        policy_type="prohibition",
        forbids_pii=True,
        forbids_external_links=True,
    )


@pytest.fixture
def compliance_policy():
    """Create a compliance policy for testing."""
    return PolicyConfig(
        policy_id="test_compliance",
        name="Test Compliance Policy",
        description="Test policy for compliance",
        policy_type="requirement",
        required_disclaimers=["ai_generated"],
        allowed_languages=["english"],
        required_format="text",
    )


def make_request(
    prompt: str = "Test prompt",
    output: str = "Test output",
    deployment_id: str = "test_deployment",
    request_id: str = "test_001",
) -> VerificationRequest:
    """Helper to create test requests."""
    return VerificationRequest(
        request_id=request_id,
        deployment_id=deployment_id,
        prompt=prompt,
        llm_output=output,
        llm_model="test-model",
    )


# ============================================
# PolicyConfig Tests
# ============================================

class TestPolicyConfig:
    """Tests for PolicyConfig dataclass."""

    def test_basic_creation(self):
        """Test basic policy config creation."""
        config = PolicyConfig(
            policy_id="test",
            name="Test Policy",
            description="A test policy",
            policy_type="constraint",
        )
        assert config.policy_id == "test"
        assert config.name == "Test Policy"
        assert config.enabled is True
        assert config.priority == 0

    def test_with_constraints(self, content_policy):
        """Test policy config with constraints."""
        assert "prohibited" in content_policy.forbidden_words
        assert "disclaimer" in content_policy.required_words
        assert content_policy.max_length == 1000
        assert content_policy.min_length == 10

    def test_privacy_flags(self, privacy_policy):
        """Test privacy policy flags."""
        assert privacy_policy.forbids_pii is True
        assert privacy_policy.forbids_external_links is True
        assert privacy_policy.forbids_code_execution is False


# ============================================
# Convenience Function Tests
# ============================================

class TestConvenienceFunctions:
    """Tests for policy creation helper functions."""

    def test_create_content_policy(self):
        """Test create_content_policy helper."""
        policy = create_content_policy(
            policy_id="content_test",
            forbidden_words=["bad", "word"],
            required_words=["good"],
            max_length=500,
        )
        assert policy.policy_id == "content_test"
        assert "bad" in policy.forbidden_words
        assert "good" in policy.required_words
        assert policy.max_length == 500
        assert policy.policy_type == "constraint"

    def test_create_privacy_policy(self):
        """Test create_privacy_policy helper."""
        policy = create_privacy_policy(
            policy_id="privacy_test",
            forbids_pii=True,
            forbids_external_links=True,
        )
        assert policy.policy_id == "privacy_test"
        assert policy.forbids_pii is True
        assert policy.forbids_external_links is True
        assert policy.policy_type == "prohibition"

    def test_create_compliance_policy(self):
        """Test create_compliance_policy helper."""
        policy = create_compliance_policy(
            policy_id="compliance_test",
            required_disclaimers=["ai_generated"],
            allowed_languages=["english", "spanish"],
        )
        assert policy.policy_id == "compliance_test"
        assert "ai_generated" in policy.required_disclaimers
        assert "english" in policy.allowed_languages
        assert policy.policy_type == "requirement"


# ============================================
# Layer Basic Tests
# ============================================

class TestBusinessPoliciesLayerBasic:
    """Basic tests for BusinessPoliciesLayer."""

    def test_layer_creation(self, layer):
        """Test layer initialization."""
        assert layer.layer_number == 3
        assert layer.layer_name == "Business Policies"
        assert layer.datalog is not None

    def test_add_policy(self, layer, content_policy):
        """Test adding a policy."""
        layer.add_policy("test_deployment", content_policy)
        policies = layer.get_policies("test_deployment")
        assert len(policies) == 1
        assert policies[0].policy_id == "test_content"

    def test_add_multiple_policies(self, layer, content_policy, privacy_policy):
        """Test adding multiple policies."""
        layer.add_policy("test_deployment", content_policy)
        layer.add_policy("test_deployment", privacy_policy)
        policies = layer.get_policies("test_deployment")
        assert len(policies) == 2

    def test_remove_policy(self, layer, content_policy):
        """Test removing a policy."""
        layer.add_policy("test_deployment", content_policy)
        result = layer.remove_policy("test_deployment", "test_content")
        assert result is True
        assert len(layer.get_policies("test_deployment")) == 0

    def test_remove_nonexistent_policy(self, layer):
        """Test removing a policy that doesn't exist."""
        result = layer.remove_policy("test_deployment", "nonexistent")
        assert result is False

    def test_get_policies_empty(self, layer):
        """Test getting policies when none exist."""
        policies = layer.get_policies("nonexistent_deployment")
        assert policies == []

    def test_layer_with_initial_policies(self, content_policy, privacy_policy):
        """Test layer creation with initial policies."""
        layer = BusinessPoliciesLayer(policies=[content_policy, privacy_policy])
        policies = layer.get_policies("default")
        assert len(policies) == 2


# ============================================
# Check Method Tests - No Policies
# ============================================

class TestCheckNoPolicies:
    """Tests for check() when no policies are configured."""

    def test_no_policies_skips(self, layer):
        """Test that check skips when no policies exist."""
        request = make_request(output="Any output here")
        result = layer.check(request, {})

        assert len(result.violations) == 0
        # Should have reasoning about skipping
        skip_steps = [r for r in result.reasoning if r.step_type == "skip"]
        assert len(skip_steps) > 0


# ============================================
# Forbidden Word Tests
# ============================================

class TestForbiddenWords:
    """Tests for forbidden word detection."""

    def test_forbidden_word_detected(self, layer, content_policy):
        """Test detection of forbidden words."""
        layer.add_policy("test_deployment", content_policy)
        request = make_request(
            output="This output contains prohibited content",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        forbidden_violations = [
            v for v in result.violations
            if v.violation_type == "forbidden_word"
        ]
        assert len(forbidden_violations) >= 1

    def test_multiple_forbidden_words(self, layer, content_policy):
        """Test detection of multiple forbidden words."""
        layer.add_policy("test_deployment", content_policy)
        request = make_request(
            output="This has prohibited and banned and secret words",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        forbidden_violations = [
            v for v in result.violations
            if v.violation_type == "forbidden_word"
        ]
        assert len(forbidden_violations) >= 3

    def test_no_forbidden_words(self, layer, content_policy):
        """Test clean output with no forbidden words."""
        layer.add_policy("test_deployment", content_policy)
        request = make_request(
            output="This output is clean and contains disclaimer text",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        forbidden_violations = [
            v for v in result.violations
            if v.violation_type == "forbidden_word"
        ]
        assert len(forbidden_violations) == 0

    def test_case_insensitive_detection(self, layer, content_policy):
        """Test that forbidden word detection is case insensitive."""
        layer.add_policy("test_deployment", content_policy)
        request = make_request(
            output="This contains PROHIBITED content",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        forbidden_violations = [
            v for v in result.violations
            if v.violation_type == "forbidden_word"
        ]
        assert len(forbidden_violations) >= 1


# ============================================
# Required Word Tests
# ============================================

class TestRequiredWords:
    """Tests for required word detection."""

    def test_missing_required_word(self, layer, content_policy):
        """Test detection of missing required words."""
        layer.add_policy("test_deployment", content_policy)
        request = make_request(
            output="This output has no required text",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        missing_violations = [
            v for v in result.violations
            if v.violation_type == "missing_required_word"
        ]
        assert len(missing_violations) >= 1

    def test_required_word_present(self, layer, content_policy):
        """Test that present required words don't trigger violations."""
        layer.add_policy("test_deployment", content_policy)
        request = make_request(
            output="This output includes the required disclaimer",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        missing_violations = [
            v for v in result.violations
            if v.violation_type == "missing_required_word"
        ]
        assert len(missing_violations) == 0


# ============================================
# Length Constraint Tests
# ============================================

class TestLengthConstraints:
    """Tests for length constraint checking."""

    def test_max_length_exceeded(self, layer):
        """Test detection of output exceeding max length."""
        policy = create_content_policy("length_test", max_length=50)
        layer.add_policy("test_deployment", policy)

        long_output = "x" * 100  # 100 characters
        request = make_request(output=long_output, deployment_id="test_deployment")
        result = layer.check(request, {})

        length_violations = [
            v for v in result.violations
            if v.violation_type == "max_length_exceeded"
        ]
        assert len(length_violations) >= 1

    def test_min_length_not_met(self, layer):
        """Test detection of output below min length."""
        policy = create_content_policy("length_test", min_length=50)
        layer.add_policy("test_deployment", policy)

        short_output = "short"  # 5 characters
        request = make_request(output=short_output, deployment_id="test_deployment")
        result = layer.check(request, {})

        length_violations = [
            v for v in result.violations
            if v.violation_type == "min_length_not_met"
        ]
        assert len(length_violations) >= 1

    def test_length_within_bounds(self, layer):
        """Test output within length bounds."""
        policy = create_content_policy("length_test", min_length=10, max_length=100)
        layer.add_policy("test_deployment", policy)

        good_output = "x" * 50  # 50 characters
        request = make_request(output=good_output, deployment_id="test_deployment")
        result = layer.check(request, {})

        length_violations = [
            v for v in result.violations
            if "length" in v.violation_type
        ]
        assert len(length_violations) == 0


# ============================================
# PII Detection Tests
# ============================================

class TestPIIDetection:
    """Tests for PII detection."""

    def test_email_detection(self, layer, privacy_policy):
        """Test detection of email addresses."""
        layer.add_policy("test_deployment", privacy_policy)
        request = make_request(
            output="Contact me at john.doe@example.com",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        pii_violations = [
            v for v in result.violations
            if v.violation_type == "pii_forbidden"
        ]
        assert len(pii_violations) >= 1

    def test_phone_detection(self, layer, privacy_policy):
        """Test detection of phone numbers."""
        layer.add_policy("test_deployment", privacy_policy)
        request = make_request(
            output="Call me at 555-123-4567",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        pii_violations = [
            v for v in result.violations
            if v.violation_type == "pii_forbidden"
        ]
        assert len(pii_violations) >= 1

    def test_ssn_detection(self, layer, privacy_policy):
        """Test detection of SSN patterns."""
        layer.add_policy("test_deployment", privacy_policy)
        request = make_request(
            output="My SSN is 123-45-6789",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        pii_violations = [
            v for v in result.violations
            if v.violation_type == "pii_forbidden"
        ]
        assert len(pii_violations) >= 1

    def test_no_pii_clean(self, layer, privacy_policy):
        """Test clean output with no PII."""
        layer.add_policy("test_deployment", privacy_policy)
        request = make_request(
            output="This is a clean output with no personal information",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        pii_violations = [
            v for v in result.violations
            if v.violation_type == "pii_forbidden"
        ]
        assert len(pii_violations) == 0


# ============================================
# External Link Tests
# ============================================

class TestExternalLinks:
    """Tests for external link detection."""

    def test_external_link_detected(self, layer, privacy_policy):
        """Test detection of external links."""
        layer.add_policy("test_deployment", privacy_policy)
        request = make_request(
            output="Check out https://example.com for more info",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        link_violations = [
            v for v in result.violations
            if v.violation_type == "external_link_forbidden"
        ]
        assert len(link_violations) >= 1

    def test_no_external_links(self, layer, privacy_policy):
        """Test output without external links."""
        layer.add_policy("test_deployment", privacy_policy)
        request = make_request(
            output="This output has no links",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        link_violations = [
            v for v in result.violations
            if v.violation_type == "external_link_forbidden"
        ]
        assert len(link_violations) == 0


# ============================================
# Disclaimer Tests
# ============================================

class TestDisclaimers:
    """Tests for disclaimer requirements."""

    def test_missing_disclaimer(self, layer, compliance_policy):
        """Test detection of missing disclaimer."""
        layer.add_policy("test_deployment", compliance_policy)
        request = make_request(
            output="Regular output without any disclaimers",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        disclaimer_violations = [
            v for v in result.violations
            if v.violation_type == "missing_disclaimer"
        ]
        assert len(disclaimer_violations) >= 1

    def test_disclaimer_present(self, layer, compliance_policy):
        """Test that present disclaimer doesn't trigger violation."""
        layer.add_policy("test_deployment", compliance_policy)
        request = make_request(
            output="This response was generated by AI. Here is your answer.",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        disclaimer_violations = [
            v for v in result.violations
            if v.violation_type == "missing_disclaimer"
        ]
        assert len(disclaimer_violations) == 0


# ============================================
# Format Tests
# ============================================

class TestFormatChecking:
    """Tests for output format checking."""

    def test_wrong_format_detected(self, layer, compliance_policy):
        """Test detection of wrong format."""
        layer.add_policy("test_deployment", compliance_policy)
        # compliance_policy requires "text" format
        request = make_request(
            output='{"key": "value"}',  # JSON format
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        format_violations = [
            v for v in result.violations
            if v.violation_type == "wrong_format"
        ]
        assert len(format_violations) >= 1

    def test_correct_format(self, layer, compliance_policy):
        """Test that correct format doesn't trigger violation."""
        layer.add_policy("test_deployment", compliance_policy)
        request = make_request(
            output="This is plain text generated by AI content",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        format_violations = [
            v for v in result.violations
            if v.violation_type == "wrong_format"
        ]
        assert len(format_violations) == 0


# ============================================
# Code Execution Tests
# ============================================

class TestCodeExecutionDetection:
    """Tests for code execution pattern detection."""

    def test_shell_command_detected(self, layer):
        """Test detection of shell command patterns."""
        policy = PolicyConfig(
            policy_id="no_code",
            name="No Code Policy",
            description="Forbids code execution",
            policy_type="prohibition",
            forbids_code_execution=True,
        )
        layer.add_policy("test_deployment", policy)

        request = make_request(
            output="Run this: os.system('rm -rf /')",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        code_violations = [
            v for v in result.violations
            if v.violation_type == "code_execution_forbidden"
        ]
        assert len(code_violations) >= 1

    def test_eval_detected(self, layer):
        """Test detection of eval patterns."""
        policy = PolicyConfig(
            policy_id="no_code",
            name="No Code Policy",
            description="Forbids code execution",
            policy_type="prohibition",
            forbids_code_execution=True,
        )
        layer.add_policy("test_deployment", policy)

        request = make_request(
            output="Use eval(user_input) to run it",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        code_violations = [
            v for v in result.violations
            if v.violation_type == "code_execution_forbidden"
        ]
        assert len(code_violations) >= 1


# ============================================
# Multiple Policy Tests
# ============================================

class TestMultiplePolicies:
    """Tests for multiple policies together."""

    def test_multiple_policies_all_pass(self, layer, content_policy, privacy_policy):
        """Test output that passes all policies."""
        content_policy.forbidden_words = ["badword"]
        content_policy.required_words = ["good"]
        layer.add_policy("test_deployment", content_policy)
        layer.add_policy("test_deployment", privacy_policy)

        request = make_request(
            output="This is a good clean output",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        # Should have minimal violations
        critical_violations = [
            v for v in result.violations
            if v.violation_type in ["forbidden_word", "pii_forbidden", "external_link_forbidden"]
        ]
        assert len(critical_violations) == 0

    def test_multiple_policies_mixed_results(self, layer, content_policy, privacy_policy):
        """Test output that passes some policies but not others."""
        layer.add_policy("test_deployment", content_policy)
        layer.add_policy("test_deployment", privacy_policy)

        request = make_request(
            output="This has prohibited content but no PII and includes disclaimer",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        # Should have forbidden word violation but not PII
        forbidden_violations = [
            v for v in result.violations
            if v.violation_type == "forbidden_word"
        ]
        pii_violations = [
            v for v in result.violations
            if v.violation_type == "pii_forbidden"
        ]

        assert len(forbidden_violations) >= 1
        assert len(pii_violations) == 0


# ============================================
# Default Deployment Fallback Tests
# ============================================

class TestDefaultDeploymentFallback:
    """Tests for default deployment fallback behavior."""

    def test_fallback_to_default(self, layer, content_policy):
        """Test fallback to default deployment when specific one has no policies."""
        layer.add_policy("default", content_policy)

        request = make_request(
            output="This contains prohibited content",
            deployment_id="nonexistent_deployment",
        )
        result = layer.check(request, {})

        # Should use default policies
        forbidden_violations = [
            v for v in result.violations
            if v.violation_type == "forbidden_word"
        ]
        assert len(forbidden_violations) >= 1


# ============================================
# Extracted Rules Tests
# ============================================

class TestExtractedRules:
    """Tests for dynamically extracted rules."""

    def test_add_extracted_rule(self, layer):
        """Test adding an extracted Datalog rule."""
        layer.add_extracted_rule("custom_rule(x) :- some_fact(x).")
        assert len(layer._extracted_rules) == 1

    def test_clear_extracted_rules(self, layer):
        """Test clearing extracted rules."""
        layer.add_extracted_rule("rule1(x) :- fact1(x).")
        layer.add_extracted_rule("rule2(x) :- fact2(x).")
        layer.clear_extracted_rules()
        assert len(layer._extracted_rules) == 0


# ============================================
# Compliance Metadata Tests
# ============================================

class TestComplianceMetadata:
    """Tests for compliance metadata in results."""

    def test_compliant_metadata(self, layer, content_policy):
        """Test compliance metadata for compliant output."""
        content_policy.forbidden_words = []
        content_policy.required_words = []
        content_policy.max_length = None
        content_policy.min_length = None
        layer.add_policy("test_deployment", content_policy)

        request = make_request(
            output="Clean output",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        # Should have compliance info in metadata
        assert "is_compliant" in result.metadata or len(result.violations) == 0

    def test_policies_checked_in_facts(self, layer, content_policy):
        """Test that checked policies are recorded in facts."""
        layer.add_policy("test_deployment", content_policy)

        request = make_request(
            output="Test output",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        assert "policies_checked" in result.facts_extracted
        assert content_policy.policy_id in result.facts_extracted["policies_checked"]


# ============================================
# Reasoning Steps Tests
# ============================================

class TestReasoningSteps:
    """Tests for reasoning step generation."""

    def test_policy_lookup_reasoning(self, layer, content_policy):
        """Test that policy lookup is recorded in reasoning."""
        layer.add_policy("test_deployment", content_policy)

        request = make_request(
            output="Test output",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        lookup_steps = [
            r for r in result.reasoning
            if r.step_type == "policy_lookup"
        ]
        assert len(lookup_steps) >= 1

    def test_rule_application_reasoning(self, layer, content_policy):
        """Test that rule application is recorded in reasoning."""
        layer.add_policy("test_deployment", content_policy)

        request = make_request(
            output="Test output",
            deployment_id="test_deployment",
        )
        result = layer.check(request, {})

        application_steps = [
            r for r in result.reasoning
            if r.step_type == "rule_application"
        ]
        assert len(application_steps) >= 1


# ============================================
# Load Rules Tests
# ============================================

class TestLoadRules:
    """Tests for load_rules method."""

    def test_load_rules_returns_basic_rules(self, layer):
        """Test that load_rules returns basic rules."""
        rules = layer.load_rules("test_deployment")
        assert len(rules) > 0
        rule_ids = [r.rule_id for r in rules]
        assert "bp_content_compliance" in rule_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
