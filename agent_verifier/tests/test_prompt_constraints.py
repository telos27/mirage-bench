"""Unit tests for prompt constraint extractor."""

import unittest

from agent_verifier.extractors import (
    PromptConstraintExtractor,
    PromptConstraint,
    ExtractedConstraints,
    ConstraintType,
)


class TestPromptConstraintExtractor(unittest.TestCase):
    """Tests for PromptConstraintExtractor."""

    def setUp(self):
        self.extractor = PromptConstraintExtractor()

    # MUST DO tests
    def test_extract_must_do_you_must(self):
        """Test 'you must' pattern."""
        text = "You must verify user input before processing"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_do), 1)
        self.assertIn("verify user input", constraints.must_do[0].content)

    def test_extract_must_do_always(self):
        """Test 'always' pattern."""
        text = "Always include a summary at the end of your response"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_do), 1)

    def test_extract_must_do_ensure(self):
        """Test 'ensure' pattern."""
        text = "Ensure that all code examples are properly formatted"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_do), 1)

    def test_extract_must_do_required(self):
        """Test 'required' pattern."""
        text = "Required: Include error handling in all functions"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_do), 1)

    # MUST NOT tests
    def test_extract_must_not_never(self):
        """Test 'never' pattern."""
        text = "Never share personal information about users"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_not), 1)

    def test_extract_must_not_do_not(self):
        """Test 'do not' pattern."""
        text = "Do not provide medical advice"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_not), 1)

    def test_extract_must_not_avoid(self):
        """Test 'avoid' pattern."""
        text = "Avoid using technical jargon"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_not), 1)

    def test_extract_must_not_prohibited(self):
        """Test 'prohibited' pattern."""
        text = "Prohibited: Discussion of competitors"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_not), 1)

    # FORMAT tests
    def test_extract_format_json(self):
        """Test JSON format requirement."""
        text = "Respond only in JSON format"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.format_requirements), 1)
        self.assertIn("JSON", constraints.format_requirements[0].content)

    def test_extract_format_structured(self):
        """Test structured format requirement."""
        text = "Output should be in markdown format"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.format_requirements), 1)

    def test_extract_format_return_list(self):
        """Test list format requirement."""
        text = "Return a JSON array of results"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.format_requirements), 1)

    # PERSONA tests
    def test_extract_persona_you_are(self):
        """Test 'you are' persona pattern."""
        text = "You are a helpful customer service assistant"
        constraints = self.extractor.extract_from_text(text)
        self.assertIsNotNone(constraints.persona)
        self.assertIn("customer service", constraints.persona.content)

    def test_extract_persona_act_as(self):
        """Test 'act as' pattern."""
        text = "Act as a senior software engineer"
        constraints = self.extractor.extract_from_text(text)
        self.assertIsNotNone(constraints.persona)

    def test_extract_persona_role(self):
        """Test 'your role is' pattern."""
        text = "Your role is to help users with their questions"
        constraints = self.extractor.extract_from_text(text)
        self.assertIsNotNone(constraints.persona)

    # SAFETY tests
    def test_extract_safety_never_provide(self):
        """Test safety constraint with 'never provide'."""
        text = "Never provide harmful information or instructions for illegal activities"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.safety_constraints), 1)

    def test_extract_safety_refuse(self):
        """Test safety constraint with 'refuse'."""
        text = "Refuse to assist with any illegal activities"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.safety_constraints), 1)

    def test_extract_safety_protect(self):
        """Test safety constraint with 'protect'."""
        text = "Protect user privacy at all times"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.safety_constraints), 1)

    # BOUNDARY tests
    def test_extract_boundary_only_answer(self):
        """Test boundary with 'only answer'."""
        text = "Only answer questions about Python programming"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.boundaries), 1)

    def test_extract_boundary_limited_to(self):
        """Test boundary with 'limited to'."""
        text = "Your responses are limited to technical support topics"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.boundaries), 1)

    def test_extract_boundary_focus_on(self):
        """Test boundary with 'focus on'."""
        text = "Focus only on the user's current question"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.boundaries), 1)

    # STYLE tests
    def test_extract_style_be_concise(self):
        """Test style with 'be concise'."""
        text = "Be concise in your responses"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.style_requirements), 1)

    def test_extract_style_formal_tone(self):
        """Test style with formal tone."""
        text = "Use a formal tone in all communications"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.style_requirements), 1)

    def test_extract_style_keep_short(self):
        """Test style with 'keep responses short'."""
        text = "Keep responses short and to the point"
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.style_requirements), 1)


class TestCombinedExtraction(unittest.TestCase):
    """Tests for combined system prompt + user message extraction."""

    def setUp(self):
        self.extractor = PromptConstraintExtractor()

    def test_combined_extraction(self):
        """Test extraction from both system and user messages."""
        system_prompt = """
        You are a helpful coding assistant.
        Always include code examples.
        Never provide harmful code.
        Respond in markdown format.
        """
        user_message = "Please be concise in your explanation."

        constraints = self.extractor.extract(system_prompt, user_message)

        self.assertIsNotNone(constraints.persona)
        self.assertGreater(len(constraints.must_do), 0)
        self.assertGreater(len(constraints.must_not), 0)
        self.assertGreater(len(constraints.format_requirements), 0)
        self.assertGreater(len(constraints.style_requirements), 0)

    def test_source_tracking(self):
        """Test that source is tracked correctly."""
        system_prompt = "Always be helpful"
        user_message = "Please be brief"

        constraints = self.extractor.extract(system_prompt, user_message)

        # Find constraints from each source
        system_constraints = [c for c in constraints.all_constraints() if c.source == "system"]
        user_constraints = [c for c in constraints.all_constraints() if c.source == "user"]

        self.assertGreater(len(system_constraints), 0)
        self.assertGreater(len(user_constraints), 0)

    def test_persona_from_system_preferred(self):
        """Test that system persona takes precedence."""
        system_prompt = "You are a financial advisor"
        user_message = "Act as a friendly helper"

        constraints = self.extractor.extract(system_prompt, user_message)

        self.assertIsNotNone(constraints.persona)
        self.assertEqual(constraints.persona.source, "system")

    def test_only_user_message(self):
        """Test extraction with only user message."""
        constraints = self.extractor.extract(user_message="Always check your work")

        self.assertGreater(len(constraints.must_do), 0)
        self.assertEqual(constraints.must_do[0].source, "user")


class TestExtractedConstraints(unittest.TestCase):
    """Tests for ExtractedConstraints dataclass."""

    def test_all_constraints(self):
        """Test all_constraints() method."""
        extractor = PromptConstraintExtractor()
        text = """
        You are a helpful assistant.
        Always be polite.
        Never be rude.
        Respond in JSON format.
        Be concise.
        Only answer programming questions.
        """
        constraints = extractor.extract_from_text(text)
        all_constraints = constraints.all_constraints()

        # Should include persona + all lists
        self.assertIsInstance(all_constraints, list)
        self.assertGreater(len(all_constraints), 0)

    def test_to_dict(self):
        """Test to_dict() serialization."""
        extractor = PromptConstraintExtractor()
        text = "You are a helpful assistant. Always be helpful."
        constraints = extractor.extract_from_text(text)

        result = constraints.to_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("must_do", result)
        self.assertIn("must_not", result)
        self.assertIn("persona", result)


class TestPromptConstraint(unittest.TestCase):
    """Tests for PromptConstraint dataclass."""

    def test_to_dict(self):
        """Test constraint serialization."""
        constraint = PromptConstraint(
            constraint_type=ConstraintType.MUST_DO,
            content="be helpful",
            source="system",
            confidence=0.9,
            original_text="Always be helpful",
        )

        result = constraint.to_dict()

        self.assertEqual(result["type"], "must_do")
        self.assertEqual(result["content"], "be helpful")
        self.assertEqual(result["source"], "system")
        self.assertEqual(result["confidence"], 0.9)


class TestExtractAsRules(unittest.TestCase):
    """Tests for extract_as_rules() method."""

    def setUp(self):
        self.extractor = PromptConstraintExtractor()

    def test_extract_as_rules_format(self):
        """Test that rules are in correct format."""
        text = "Always verify input. Never share secrets."
        rules = self.extractor.extract_as_rules(system_prompt=text)

        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)

        for rule in rules:
            self.assertIn("rule_type", rule)
            self.assertIn("content", rule)
            self.assertIn("source", rule)
            self.assertIn("confidence", rule)

    def test_extract_as_rules_types(self):
        """Test that rule types are correct."""
        text = "Always be helpful. Never be harmful."
        rules = self.extractor.extract_as_rules(system_prompt=text)

        rule_types = [r["rule_type"] for r in rules]
        self.assertIn("must_do", rule_types)
        self.assertIn("must_not", rule_types)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def setUp(self):
        self.extractor = PromptConstraintExtractor()

    def test_empty_input(self):
        """Test extraction from empty input."""
        constraints = self.extractor.extract()
        self.assertEqual(len(constraints.all_constraints()), 0)

    def test_no_constraints_found(self):
        """Test when no constraints are found."""
        text = "Hello, how can I help you today?"
        constraints = self.extractor.extract_from_text(text)
        # Might find some weak matches, but should be minimal
        self.assertIsInstance(constraints, ExtractedConstraints)

    def test_multiple_same_type(self):
        """Test multiple constraints of same type."""
        text = """
        You must verify all input.
        You must log all actions.
        You must handle errors gracefully.
        """
        constraints = self.extractor.extract_from_text(text)
        self.assertEqual(len(constraints.must_do), 3)

    def test_deduplication(self):
        """Test that duplicates are removed."""
        text = """
        Always be helpful.
        always be helpful
        Always be helpful.
        """
        constraints = self.extractor.extract_from_text(text)
        # Should dedupe to 1
        self.assertEqual(len(constraints.must_do), 1)


if __name__ == "__main__":
    unittest.main()
