"""Tests for Layer 6: Prompt Constraints."""

import pytest

from agent_verifier.layers import (
    PromptConstraintsLayer,
    ConstraintCheckResult,
    check_constraints,
)
from agent_verifier.schemas.request import VerificationRequest
from agent_verifier.extractors.prompt_constraints import (
    PromptConstraintExtractor,
    ExtractedConstraints,
    ConstraintType,
    PromptConstraint,
)


# ============================================
# Output Analysis Tests
# ============================================

class TestOutputAnalysis:
    """Tests for output analysis functionality."""

    def test_detect_format_json(self):
        """Test JSON format detection."""
        layer = PromptConstraintsLayer()

        json_output = '{"key": "value", "number": 42}'
        assert layer._detect_format(json_output) == "json"

        json_array = '[{"id": 1}, {"id": 2}]'
        assert layer._detect_format(json_array) == "json"

    def test_detect_format_xml(self):
        """Test XML format detection."""
        layer = PromptConstraintsLayer()

        xml_output = '<root><item>value</item></root>'
        assert layer._detect_format(xml_output) == "xml"

    def test_detect_format_code_blocks(self):
        """Test code block format detection."""
        layer = PromptConstraintsLayer()

        code_output = "Here is the code:\n```python\nprint('hello')\n```"
        assert layer._detect_format(code_output) == "code_blocks"

    def test_detect_format_bullet_list(self):
        """Test bullet list format detection."""
        layer = PromptConstraintsLayer()

        bullet_output = "Items:\n- First item\n- Second item\n- Third item"
        assert layer._detect_format(bullet_output) == "bullet_list"

    def test_detect_format_numbered_list(self):
        """Test numbered list format detection."""
        layer = PromptConstraintsLayer()

        numbered_output = "Steps:\n1. First step\n2. Second step\n3. Third step"
        assert layer._detect_format(numbered_output) == "numbered_list"

    def test_detect_format_prose(self):
        """Test prose format detection."""
        layer = PromptConstraintsLayer()

        prose_output = "This is a paragraph of text that explains something in detail."
        assert layer._detect_format(prose_output) == "prose"

    def test_detect_style_formal(self):
        """Test formal style detection."""
        layer = PromptConstraintsLayer()

        formal_output = "Therefore, it is consequently important to furthermore consider the implications."
        assert layer._detect_style(formal_output) == "formal"

    def test_detect_style_casual(self):
        """Test casual style detection."""
        layer = PromptConstraintsLayer()

        casual_output = "Hey, that's awesome! Yeah, gonna be cool lol"
        assert layer._detect_style(casual_output) == "casual"

    def test_detect_style_concise(self):
        """Test concise style detection."""
        layer = PromptConstraintsLayer()

        concise_output = "Yes, that works."
        assert layer._detect_style(concise_output) == "concise"

    def test_detect_style_detailed(self):
        """Test detailed style detection."""
        layer = PromptConstraintsLayer()

        detailed_output = " ".join(["word"] * 200)
        assert layer._detect_style(detailed_output) == "detailed"

    def test_extract_mentions(self):
        """Test extracting mentions from output."""
        layer = PromptConstraintsLayer()

        output = 'The user mentioned "Python" and "JavaScript". Also discussed Machine Learning.'
        mentions = layer._extract_mentions(output)

        assert "python" in mentions
        assert "javascript" in mentions

    def test_extract_actions(self):
        """Test extracting actions from output."""
        layer = PromptConstraintsLayer()

        output = "I will help you with that. Let me check the documentation. Here is the code."
        actions = layer._extract_actions(output)

        assert len(actions) >= 1

    def test_extract_topics(self):
        """Test extracting topics from output."""
        layer = PromptConstraintsLayer()

        output = "Here is the Python code with SQL query to access the database."
        topics = layer._extract_topics(output)

        assert "programming" in topics
        assert "database" in topics

    def test_detect_safety_issues(self):
        """Test detecting safety issues."""
        layer = PromptConstraintsLayer()

        # Safe output
        safe_output = "Here is how to create a Python function."
        issues = layer._detect_safety_issues(safe_output)
        assert len(issues) == 0

        # Potentially unsafe
        unsafe_output = "password: abc123"
        issues = layer._detect_safety_issues(unsafe_output)
        assert "credential_exposure" in issues


# ============================================
# Constraint Extraction Tests
# ============================================

class TestConstraintExtraction:
    """Tests for constraint extraction."""

    def test_extract_constraints_from_request(self):
        """Test extracting constraints from a request."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="You must respond in JSON format. Never mention competitors.",
            llm_output="{}",
            llm_model="gpt-4",
        )

        constraints = layer.extract_constraints(request, {})

        assert len(constraints.format_requirements) >= 1 or len(constraints.must_do) >= 1

    def test_extract_constraints_with_system_prompt(self):
        """Test extracting constraints with system prompt in context."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Help me with coding",
            llm_output="Here is some code.",
            llm_model="gpt-4",
        )

        context = {
            "system_prompt": "You are a helpful assistant. You must always be polite. Never provide medical advice."
        }

        constraints = layer.extract_constraints(request, context)

        assert constraints.persona is not None
        assert len(constraints.must_do) >= 1

    def test_extract_constraints_preextracted(self):
        """Test using pre-extracted constraints from context."""
        layer = PromptConstraintsLayer()

        pre_extracted = ExtractedConstraints(
            must_do=[
                PromptConstraint(
                    constraint_type=ConstraintType.MUST_DO,
                    content="be helpful",
                    source="system",
                    confidence=0.9,
                )
            ]
        )

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi!",
            llm_model="gpt-4",
        )

        context = {"extracted_constraints": pre_extracted}

        constraints = layer.extract_constraints(request, context)

        assert len(constraints.must_do) == 1
        assert constraints.must_do[0].content == "be helpful"


# ============================================
# Verification Tests
# ============================================

class TestVerification:
    """Tests for verification functionality."""

    def test_check_no_constraints(self):
        """Test checking when there are no constraints."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi there!",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        assert len(result.reasoning) >= 1
        assert result.metadata.get("constraints_satisfied", True) is True

    def test_check_with_constraints(self):
        """Test checking with constraints."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="You must respond in JSON format.",
            llm_output='{"message": "Hello"}',
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        assert len(result.reasoning) >= 1

    def test_check_format_violation(self):
        """Test detecting format violations."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Respond only in JSON format.",
            llm_output="This is just plain text, not JSON.",
            llm_model="gpt-4",
        )

        context = {
            "system_prompt": "You must always respond in JSON format."
        }

        result = layer.check(request, context)

        assert len(result.reasoning) >= 1

    def test_check_with_system_prompt(self):
        """Test checking with system prompt constraints."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Help me",
            llm_output="Here is how I can help you.",
            llm_model="gpt-4",
        )

        context = {
            "system_prompt": "You are a helpful assistant. Always be concise. Never mention competitors."
        }

        result = layer.check(request, context)

        assert len(result.reasoning) >= 2  # extraction + rule application


# ============================================
# Convenience Function Tests
# ============================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_constraints_simple(self):
        """Test simple constraint checking."""
        result = check_constraints(
            prompt="Hello",
            output="Hi there!",
        )

        assert result.layer == 6
        assert isinstance(result.violations, list)

    def test_check_constraints_with_system_prompt(self):
        """Test constraint checking with system prompt."""
        result = check_constraints(
            prompt="Help me with code",
            output="Here is the code.",
            system_prompt="You are a coding assistant. Always provide examples.",
        )

        assert result.layer == 6
        assert len(result.reasoning) >= 1


# ============================================
# Rule Management Tests
# ============================================

class TestRuleManagement:
    """Tests for rule management."""

    def test_add_extracted_rule(self):
        """Test adding extracted Datalog rules."""
        layer = PromptConstraintsLayer()

        rule = "custom_violation(id) :- some_condition(id)."
        layer.add_extracted_rule(rule)

        program = layer._get_rules_program()
        assert "custom_violation" in program

    def test_clear_extracted_rules(self):
        """Test clearing extracted rules."""
        layer = PromptConstraintsLayer()

        layer.add_extracted_rule("rule1(x) :- fact(x).")
        layer.add_extracted_rule("rule2(x) :- fact(x).")
        layer.clear_extracted_rules()

        program = layer._get_rules_program()
        assert "rule1" not in program

    def test_load_rules(self):
        """Test loading rules for deployment."""
        layer = PromptConstraintsLayer()

        rules = layer.load_rules("my-app")

        assert len(rules) >= 1
        assert any(r.rule_id == "pc_must_do" for r in rules)
        assert any(r.rule_id == "pc_must_not" for r in rules)


# ============================================
# Configuration Tests
# ============================================

class TestConfiguration:
    """Tests for layer configuration."""

    def test_min_confidence_threshold(self):
        """Test minimum confidence threshold."""
        layer = PromptConstraintsLayer(min_confidence=0.9)
        assert layer.min_confidence == 0.9

    def test_strict_safety_mode(self):
        """Test strict safety mode."""
        layer = PromptConstraintsLayer(strict_safety=True)
        assert layer.strict_safety is True

        layer2 = PromptConstraintsLayer(strict_safety=False)
        assert layer2.strict_safety is False


# ============================================
# Edge Cases
# ============================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_output(self):
        """Test handling empty output."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})
        assert result is not None

    def test_very_long_output(self):
        """Test handling very long output."""
        layer = PromptConstraintsLayer()

        long_output = "word " * 10000

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output=long_output,
            llm_model="gpt-4",
        )

        result = layer.check(request, {})
        assert result is not None

    def test_special_characters_in_output(self):
        """Test handling special characters."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output='Text with "quotes" and \t tabs and \n newlines',
            llm_model="gpt-4",
        )

        result = layer.check(request, {})
        assert result is not None

    def test_unicode_output(self):
        """Test handling unicode output."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hello 你好 مرحبا 🎉",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})
        assert result is not None

    def test_many_constraints(self):
        """Test handling many constraints."""
        layer = PromptConstraintsLayer()

        # Create a prompt with many constraints
        constraints = [
            "You must be helpful.",
            "You must be polite.",
            "You must be concise.",
            "Never be rude.",
            "Never provide false information.",
            "Always use proper grammar.",
            "Respond in a professional tone.",
        ]
        prompt = " ".join(constraints)

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt=prompt,
            llm_output="I am happy to help you.",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})
        assert result is not None


# ============================================
# Layer Result Tests
# ============================================

class TestLayerResult:
    """Tests for LayerResult structure."""

    def test_result_structure(self):
        """Test that results have correct structure."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi!",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        assert result.layer == 6
        assert isinstance(result.violations, list)
        assert isinstance(result.reasoning, list)
        assert isinstance(result.facts_extracted, dict)
        assert isinstance(result.metadata, dict)

    def test_result_facts_extracted(self):
        """Test that facts are properly extracted."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="You must be helpful.",
            llm_output="I am happy to help!",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        assert "total_constraints" in result.facts_extracted
        assert "constraints_satisfied" in result.facts_extracted


# ============================================
# Integration Tests
# ============================================

class TestIntegration:
    """Integration tests for Layer 6."""

    def test_must_do_scenario(self):
        """Test must-do constraint scenario."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="You must include a code example in your response.",
            llm_output="Here is a code example:\n```python\nprint('hello')\n```",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        # Should detect the must-do constraint
        assert len(result.reasoning) >= 1
        assert result.facts_extracted["total_constraints"] >= 1

    def test_must_not_scenario(self):
        """Test must-not constraint scenario."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Never mention competitors.",
            llm_output="I recommend our product.",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        assert len(result.reasoning) >= 1

    def test_format_scenario(self):
        """Test format constraint scenario."""
        layer = PromptConstraintsLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Respond in JSON format.",
            llm_output='{"status": "ok", "message": "done"}',
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        assert len(result.reasoning) >= 1

    def test_persona_scenario(self):
        """Test persona constraint scenario."""
        layer = PromptConstraintsLayer()

        context = {
            "system_prompt": "You are a helpful coding assistant. Always be encouraging."
        }

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Help me learn Python",
            llm_output="Great choice! Python is a wonderful language to learn.",
            llm_model="gpt-4",
        )

        constraints = layer.extract_constraints(request, context)

        assert constraints.persona is not None
        assert "assistant" in constraints.persona.content.lower()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
