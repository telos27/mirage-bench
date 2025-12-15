"""End-to-end integration tests for the full verification engine."""

import pytest

from agent_verifier import (
    VerificationRequest,
    VerificationResult,
    VerificationEngine,
    EngineConfig,
    create_engine,
    create_full_engine,
    create_lightweight_engine,
    create_coding_engine,
    quick_verify,
    Session,
    Turn,
)
from agent_verifier.layers import (
    CommonKnowledgeLayer,
    DomainBestPracticesLayer,
    BusinessPoliciesLayer,
    UserPreferencesLayer,
    SessionHistoryLayer,
    PromptConstraintsLayer,
    create_content_policy,
    create_developer_preferences,
)


# ============================================
# Factory Function Tests
# ============================================

class TestFactoryFunctions:
    """Tests for engine factory functions."""

    def test_create_engine_default(self):
        """Test creating engine with default config (all layers)."""
        engine = create_engine()

        assert len(engine.active_layers) == 6
        assert engine.get_layer(1) is not None
        assert engine.get_layer(6) is not None

    def test_create_engine_specific_layers(self):
        """Test creating engine with specific layers."""
        engine = create_engine(layers=[1, 3, 6])

        assert len(engine.active_layers) == 3
        assert engine.get_layer(1) is not None
        assert engine.get_layer(2) is None
        assert engine.get_layer(3) is not None

    def test_create_engine_with_domains(self):
        """Test creating engine with domain config."""
        engine = create_engine(layers=[2], domains=["coding", "customer_service"])

        layer2 = engine.get_layer(2)
        assert layer2 is not None
        assert "coding" in layer2.get_active_domains()

    def test_create_full_engine(self):
        """Test creating full engine."""
        engine = create_full_engine()

        assert len(engine.active_layers) == 6

    def test_create_lightweight_engine(self):
        """Test creating lightweight engine."""
        engine = create_lightweight_engine()

        assert len(engine.active_layers) == 2
        assert engine.get_layer(1) is not None
        assert engine.get_layer(6) is not None

    def test_create_coding_engine(self):
        """Test creating coding engine."""
        engine = create_coding_engine()

        assert len(engine.active_layers) == 3
        layer2 = engine.get_layer(2)
        assert "coding" in layer2.get_active_domains()


# ============================================
# Quick Verify Tests
# ============================================

class TestQuickVerify:
    """Tests for quick_verify function."""

    def test_quick_verify_basic(self):
        """Test basic quick verification."""
        result = quick_verify(
            prompt="Hello",
            output="Hi there!",
        )

        assert isinstance(result, VerificationResult)
        assert result.request_id == "quick_verify"
        assert result.verdict in ("pass", "fail")

    def test_quick_verify_with_system_prompt(self):
        """Test quick verification with system prompt."""
        result = quick_verify(
            prompt="Help me",
            output="I'm happy to help!",
            system_prompt="You are a helpful assistant.",
        )

        assert isinstance(result, VerificationResult)

    def test_quick_verify_custom_layers(self):
        """Test quick verification with custom layers."""
        result = quick_verify(
            prompt="Hello",
            output="Hi!",
            layers=[1],
        )

        assert 1 in result.layers_checked
        assert 6 not in result.layers_checked


# ============================================
# Full Engine Verification Tests
# ============================================

class TestFullEngineVerification:
    """Tests for full engine verification."""

    def test_verify_simple_request(self):
        """Test verifying a simple request."""
        engine = create_full_engine()

        request = VerificationRequest(
            request_id="test_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi there! How can I help you?",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert result.request_id == "test_001"
        assert result.verdict in ("pass", "fail")
        assert len(result.layers_checked) >= 1

    def test_verify_with_all_layers(self):
        """Test that all layers are checked."""
        engine = create_full_engine()

        request = VerificationRequest(
            request_id="test_002",
            deployment_id="my-app",
            prompt="Write a Python function",
            llm_output="def hello():\n    print('Hello')",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        # All 6 layers should be checked
        assert len(result.layers_checked) == 6
        assert set(result.layers_checked) == {1, 2, 3, 4, 5, 6}

    def test_verify_with_reasoning(self):
        """Test that reasoning is included."""
        engine = create_full_engine()

        request = VerificationRequest(
            request_id="test_003",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi!",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert len(result.reasoning) > 0
        # Should have at least one reasoning step per layer
        layers_in_reasoning = set(r.layer for r in result.reasoning)
        assert len(layers_in_reasoning) >= 1

    def test_verify_batch(self):
        """Test batch verification."""
        engine = create_lightweight_engine()

        requests = [
            VerificationRequest(
                request_id=f"batch_{i}",
                deployment_id="my-app",
                prompt=f"Prompt {i}",
                llm_output=f"Response {i}",
                llm_model="gpt-4",
            )
            for i in range(5)
        ]

        results = engine.verify_batch(requests)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.request_id == f"batch_{i}"


# ============================================
# Layer Integration Tests
# ============================================

class TestLayerIntegration:
    """Tests for layer integration."""

    def test_layer1_common_knowledge(self):
        """Test Layer 1 integration."""
        engine = create_engine(layers=[1])

        request = VerificationRequest(
            request_id="test_l1",
            deployment_id="my-app",
            prompt="What is Python?",
            llm_output="Python is a programming language.",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert 1 in result.layers_checked
        # Check that Layer 1 produced reasoning
        l1_reasoning = [r for r in result.reasoning if r.layer == 1]
        assert len(l1_reasoning) >= 1

    def test_layer2_domain(self):
        """Test Layer 2 integration."""
        engine = create_engine(layers=[2], domains=["coding"])

        request = VerificationRequest(
            request_id="test_l2",
            deployment_id="my-app",
            prompt="Write code",
            llm_output="def foo(): pass",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert 2 in result.layers_checked

    def test_layer3_business_policy(self):
        """Test Layer 3 with policy."""
        engine = create_engine(layers=[3])

        # Add a policy
        layer3 = engine.get_layer(3)
        layer3.add_policy("my-app", create_content_policy(
            policy_id="test_policy",
            forbidden_words=["competitor"],
        ))

        request = VerificationRequest(
            request_id="test_l3",
            deployment_id="my-app",
            prompt="Tell me about options",
            llm_output="You should try competitor product!",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert 3 in result.layers_checked
        # Should detect forbidden word
        l3_violations = [v for v in result.violations if v.layer == 3]
        assert len(l3_violations) >= 1

    def test_layer4_user_preferences(self):
        """Test Layer 4 with preferences."""
        engine = create_engine(layers=[4])

        # Add preferences
        layer4 = engine.get_layer(4)
        layer4.set_preferences("user_123", create_developer_preferences("user_123"))

        request = VerificationRequest(
            request_id="test_l4",
            deployment_id="my-app",
            prompt="Help me",
            llm_output="Here is some technical code example.",
            llm_model="gpt-4",
            user_id="user_123",
        )

        result = engine.verify(request)

        assert 4 in result.layers_checked

    def test_layer5_session(self):
        """Test Layer 5 with session."""
        engine = create_engine(layers=[5])

        # Create a session
        layer5 = engine.get_layer(5)
        layer5.create_session("sess_001", "user_123", "my-app")
        layer5.add_turn("sess_001", "turn_0", "Hello", "Hi there!")

        request = VerificationRequest(
            request_id="test_l5",
            deployment_id="my-app",
            prompt="Tell me more",
            llm_output="Sure, let me explain further.",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = engine.verify(request)

        assert 5 in result.layers_checked

    def test_layer6_prompt_constraints(self):
        """Test Layer 6 with constraints."""
        engine = create_engine(layers=[6])
        engine.set_system_prompt("You must always respond in JSON format.")

        request = VerificationRequest(
            request_id="test_l6",
            deployment_id="my-app",
            prompt="Give me data",
            llm_output='{"data": "value"}',
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert 6 in result.layers_checked


# ============================================
# Fail Fast Tests
# ============================================

class TestFailFast:
    """Tests for fail-fast behavior."""

    def test_fail_fast_enabled(self):
        """Test fail-fast stops at critical violation."""
        engine = create_engine(layers=[3, 4, 5, 6], fail_fast=True)

        # Add policy that will trigger violation
        layer3 = engine.get_layer(3)
        layer3.add_policy("my-app", create_content_policy(
            policy_id="strict_policy",
            forbidden_words=["badword"],
        ))

        request = VerificationRequest(
            request_id="test_ff",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="This has badword in it.",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        # Should have stopped early
        assert result.verdict == "fail"

    def test_fail_fast_disabled(self):
        """Test all layers checked when fail-fast disabled."""
        engine = create_engine(layers=[1, 6], fail_fast=False)

        request = VerificationRequest(
            request_id="test_noff",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi!",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        # All layers should be checked
        assert len(result.layers_checked) == 2


# ============================================
# Context Flow Tests
# ============================================

class TestContextFlow:
    """Tests for context flowing between layers."""

    def test_facts_flow_between_layers(self):
        """Test that facts extracted by one layer are available to next."""
        engine = create_full_engine()

        request = VerificationRequest(
            request_id="test_ctx",
            deployment_id="my-app",
            prompt="Write Python code",
            llm_output="def hello(): print('Hello')",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        # Check metadata shows layers were checked in order
        assert result.metadata["config"]["enabled_layers"] == [1, 2, 3, 4, 5, 6]


# ============================================
# Edge Cases
# ============================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_output(self):
        """Test handling empty output."""
        engine = create_lightweight_engine()

        request = VerificationRequest(
            request_id="test_empty",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="",
            llm_model="gpt-4",
        )

        result = engine.verify(request)
        assert result is not None

    def test_very_long_output(self):
        """Test handling very long output."""
        engine = create_lightweight_engine()

        long_output = "word " * 10000

        request = VerificationRequest(
            request_id="test_long",
            deployment_id="my-app",
            prompt="Hello",
            llm_output=long_output,
            llm_model="gpt-4",
        )

        result = engine.verify(request)
        assert result is not None

    def test_unicode_content(self):
        """Test handling unicode content."""
        engine = create_lightweight_engine()

        request = VerificationRequest(
            request_id="test_unicode",
            deployment_id="my-app",
            prompt="Hello 你好",
            llm_output="Hi there! 你好 🎉",
            llm_model="gpt-4",
        )

        result = engine.verify(request)
        assert result is not None

    def test_no_layers_enabled(self):
        """Test with no layers enabled."""
        engine = create_engine(layers=[])

        request = VerificationRequest(
            request_id="test_nolayers",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi!",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert result.verdict == "pass"
        assert len(result.layers_checked) == 0


# ============================================
# Real-World Scenario Tests
# ============================================

class TestRealWorldScenarios:
    """Tests for real-world scenarios."""

    def test_coding_assistant_scenario(self):
        """Test coding assistant scenario."""
        engine = create_coding_engine()
        engine.set_system_prompt(
            "You are a helpful coding assistant. Always provide working code examples."
        )

        request = VerificationRequest(
            request_id="coding_001",
            deployment_id="coding-app",
            prompt="Write a function to add two numbers",
            llm_output="""
def add(a, b):
    return a + b

# Example usage
result = add(5, 3)
print(result)  # Output: 8
            """,
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert result.verdict == "pass"

    def test_customer_service_scenario(self):
        """Test customer service scenario."""
        engine = create_engine(layers=[1, 2, 3, 6], domains=["customer_service"])

        # Add customer service policy
        layer3 = engine.get_layer(3)
        layer3.add_policy("support-app", create_content_policy(
            policy_id="support_policy",
            required_words=["help"],
        ))

        engine.set_system_prompt(
            "You are a customer service agent. Be polite and helpful."
        )

        request = VerificationRequest(
            request_id="cs_001",
            deployment_id="support-app",
            prompt="I have a problem with my order",
            llm_output="I'd be happy to help you with your order issue.",
            llm_model="gpt-4",
        )

        result = engine.verify(request)

        assert result.verdict == "pass"

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation scenario."""
        engine = create_engine(layers=[1, 5, 6])

        # Setup session
        layer5 = engine.get_layer(5)
        layer5.create_session("conv_001", "user_123", "chat-app")
        layer5.add_turn("conv_001", "turn_0", "My name is Alice", "Nice to meet you, Alice!")
        layer5.establish_fact("conv_001", "f1", "user_stated", "name", "Alice", "turn_0", 0.95)

        # Second turn
        request = VerificationRequest(
            request_id="turn_1",
            deployment_id="chat-app",
            prompt="What is my name?",
            llm_output="Your name is Alice, as you mentioned earlier.",
            llm_model="gpt-4",
            session_id="conv_001",
        )

        result = engine.verify(request)

        assert 5 in result.layers_checked


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
