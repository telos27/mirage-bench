"""Tests for Layer 1: Common Knowledge."""

import pytest

from agent_verifier import (
    VerificationRequest,
    CommonKnowledgeLayer,
    ExtractedFacts,
    Severity,
    check_souffle_installed,
)
from agent_verifier.schemas.rules import Rule, RuleType


def make_request(request_id="test-1", prompt="Test prompt", output="Test output"):
    """Helper to create test requests."""
    return VerificationRequest(
        request_id=request_id,
        deployment_id="test-deployment",
        prompt=prompt,
        llm_output=output,
        llm_model="test-model",
    )


class TestExtractedFacts:
    """Tests for ExtractedFacts dataclass."""

    def test_default_values(self):
        facts = ExtractedFacts()
        assert facts.context_elements == []
        assert facts.context_errors == []
        assert facts.output_references == []
        assert facts.request_task == ""
        assert facts.acknowledges_error is False

    def test_with_values(self):
        facts = ExtractedFacts(
            context_elements=["button", "link"],
            context_errors=["Error: not found"],
            output_references=["button"],
            acknowledges_error=True,
        )
        assert len(facts.context_elements) == 2
        assert len(facts.context_errors) == 1
        assert facts.acknowledges_error is True


class TestCommonKnowledgeLayer:
    """Tests for CommonKnowledgeLayer."""

    def test_create_layer(self):
        layer = CommonKnowledgeLayer()
        assert layer.layer_number == 1
        assert layer.layer_name == "Common Knowledge"

    def test_has_basic_rules(self):
        layer = CommonKnowledgeLayer()
        rules = layer.load_rules("any-deployment")
        assert len(rules) >= 3  # At least the 3 basic rules

    def test_add_rule(self):
        layer = CommonKnowledgeLayer()
        initial_count = len(layer._dynamic_rules)

        layer.add_rule(Rule(
            rule_id="custom_rule",
            name="Custom Rule",
            description="Test rule",
            rule_type=RuleType.CONSTRAINT,
            layer=1,
            conditions=[],  # Empty conditions - checked via Datalog
        ))

        assert len(layer._dynamic_rules) == initial_count + 1

    def test_add_extracted_rule(self):
        layer = CommonKnowledgeLayer()
        assert len(layer.extracted_rules) == 0

        layer.add_extracted_rule("custom_violation(id) :- some_condition(id).")
        assert len(layer.extracted_rules) == 1

    def test_clear_extracted_rules(self):
        layer = CommonKnowledgeLayer()
        layer.add_extracted_rule("rule1")
        layer.add_extracted_rule("rule2")
        layer.clear_extracted_rules()
        assert len(layer.extracted_rules) == 0

    def test_check_returns_layer_result(self):
        layer = CommonKnowledgeLayer()
        request = make_request()
        result = layer.check(request, {})

        assert result.layer == 1
        assert isinstance(result.violations, list)
        assert isinstance(result.reasoning, list)

    def test_basic_extraction_acknowledges_error(self):
        """Test that basic extraction detects error acknowledgment."""
        layer = CommonKnowledgeLayer()

        # Output that acknowledges error
        request = make_request(
            output="I see there was an error. Let me try a different approach."
        )
        facts = layer._extract_facts(request, {})
        assert facts.acknowledges_error is True

        # Output that doesn't mention errors
        request = make_request(output="Everything looks good.")
        facts = layer._extract_facts(request, {})
        assert facts.acknowledges_error is False

    def test_basic_extraction_acknowledges_history(self):
        """Test that basic extraction detects history acknowledgment."""
        layer = CommonKnowledgeLayer()

        # Output that acknowledges previous attempts
        request = make_request(
            output="I already tried this before, let me try something else."
        )
        facts = layer._extract_facts(request, {})
        assert facts.acknowledges_history is True

        # Output without history reference
        request = make_request(output="Let me click the button.")
        facts = layer._extract_facts(request, {})
        assert facts.acknowledges_history is False

    def test_check_uses_provided_context(self):
        """Test that check uses facts from context if provided."""
        layer = CommonKnowledgeLayer()

        request = make_request()
        context = {
            "extracted_facts": {
                "context_elements": ["submit button", "cancel link"],
                "context_errors": [],
                "output_references": ["submit button"],
                "acknowledges_error": False,
            }
        }

        result = layer.check(request, context)

        # Should have extracted facts info in reasoning
        assert any("fact_extraction" in r.step_type for r in result.reasoning)

    def test_reasoning_includes_steps(self):
        """Test that reasoning steps are included."""
        layer = CommonKnowledgeLayer()
        request = make_request()
        result = layer.check(request, {})

        # Should have at least extraction step
        step_types = [r.step_type for r in result.reasoning]
        assert "fact_extraction" in step_types


@pytest.mark.skipif(
    not check_souffle_installed(),
    reason="Soufflé is not installed"
)
class TestCommonKnowledgeLayerWithSouffle:
    """Tests that require Soufflé to be installed."""

    def test_detect_ungrounded_reference(self):
        """Test detection of ungrounded reference via Datalog."""
        layer = CommonKnowledgeLayer()

        request = make_request(request_id="test-ungrounded")
        context = {
            "extracted_facts": {
                "context_elements": ["visible button"],
                "context_errors": [],
                "output_references": ["invisible element"],  # Not in context!
                "output_action": "",
                "output_target": "",
                "acknowledges_error": False,
                "acknowledges_history": False,
                "action_history": [],
            }
        }

        result = layer.check(request, context)

        # Should detect ungrounded reference
        violation_types = [v.violation_type for v in result.violations]
        assert "ungrounded_reference" in violation_types

    def test_detect_ignored_error(self):
        """Test detection of ignored error via Datalog."""
        layer = CommonKnowledgeLayer()

        request = make_request(
            request_id="test-ignored-error",
            output="Let me click the button."  # No error acknowledgment
        )
        context = {
            "extracted_facts": {
                "context_elements": ["button"],
                "context_errors": ["Error: action failed"],  # Has error!
                "output_references": [],
                "output_action": "click",
                "output_target": "button",
                "acknowledges_error": False,  # Not acknowledged
                "acknowledges_history": False,
                "action_history": [],
            }
        }

        result = layer.check(request, context)

        # Should detect ignored error
        violation_types = [v.violation_type for v in result.violations]
        assert "ignored_error" in violation_types

    def test_detect_repeated_failed_action(self):
        """Test detection of repeated failed action via Datalog."""
        layer = CommonKnowledgeLayer()

        request = make_request(
            request_id="test-repeated",
            output="Let me click the button."  # No history acknowledgment
        )
        context = {
            "extracted_facts": {
                "context_elements": ["button"],
                "context_errors": [],
                "output_references": [],
                "output_action": "click button",
                "output_target": "button",
                "acknowledges_error": False,
                "acknowledges_history": False,  # Not acknowledged
                "action_history": [("click button", "failed")],  # Same action failed before!
            }
        }

        result = layer.check(request, context)

        # Should detect repeated failed action
        violation_types = [v.violation_type for v in result.violations]
        assert "repeated_failed_action" in violation_types

    def test_no_violation_when_acknowledged(self):
        """Test that acknowledged errors don't trigger violations."""
        layer = CommonKnowledgeLayer()

        request = make_request(
            request_id="test-acknowledged",
            output="I see the error, let me try differently."
        )
        context = {
            "extracted_facts": {
                "context_elements": ["button"],
                "context_errors": ["Error: something failed"],
                "output_references": [],
                "output_action": "try different",
                "output_target": "",
                "acknowledges_error": True,  # Acknowledged!
                "acknowledges_history": False,
                "action_history": [],
            }
        }

        result = layer.check(request, context)

        # Should NOT detect ignored error since it was acknowledged
        violation_types = [v.violation_type for v in result.violations]
        assert "ignored_error" not in violation_types

    def test_pass_when_consistent(self):
        """Test that consistent output has no violations."""
        layer = CommonKnowledgeLayer()

        request = make_request(request_id="test-consistent")
        context = {
            "extracted_facts": {
                "context_elements": ["submit button", "cancel link"],
                "context_errors": [],
                "output_references": ["submit button"],  # In context
                "output_action": "click",
                "output_target": "submit button",  # In context
                "acknowledges_error": False,
                "acknowledges_history": False,
                "action_history": [],
            }
        }

        result = layer.check(request, context)

        # Should have no violations
        assert len(result.violations) == 0

        # Should have consistency metadata
        assert result.metadata.get("is_consistent", False) is True
