#!/usr/bin/env python3
"""
Tests for the Generic Hallucination Verifier.

Tests the common sense-based approach to hallucination detection.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import logging

from verifier.generic_schema import (
    ViolationType,
    Severity,
    ConfidenceLevel,
    Violation,
    CommonSenseEvaluation,
    violations_to_score,
    derive_emergent_type,
)
from verifier.generic_verifier import GenericHallucinationVerifier


class TestGenericSchema(unittest.TestCase):
    """Test the generic schema utilities."""

    def test_violations_to_score_no_violations(self):
        """No violations should return score 2."""
        self.assertEqual(violations_to_score([]), 2)

    def test_violations_to_score_high_severity(self):
        """High severity violation should return score 0."""
        violations = [
            Violation(
                type=ViolationType.UNGROUNDED_REFERENCE,
                description="Test",
                severity=Severity.HIGH,
            )
        ]
        self.assertEqual(violations_to_score(violations), 0)

    def test_violations_to_score_multiple_medium(self):
        """Multiple medium severity violations should return score 0."""
        violations = [
            Violation(
                type=ViolationType.IGNORED_EVIDENCE,
                description="Test 1",
                severity=Severity.MEDIUM,
            ),
            Violation(
                type=ViolationType.STATE_CONFUSION,
                description="Test 2",
                severity=Severity.MEDIUM,
            ),
        ]
        self.assertEqual(violations_to_score(violations), 0)

    def test_violations_to_score_single_medium(self):
        """Single medium severity violation should return score 1."""
        violations = [
            Violation(
                type=ViolationType.REASONING_MISMATCH,
                description="Test",
                severity=Severity.MEDIUM,
            )
        ]
        self.assertEqual(violations_to_score(violations), 1)

    def test_violations_to_score_low_severity(self):
        """Low severity violations should return score 1."""
        violations = [
            Violation(
                type=ViolationType.OTHER,
                description="Test",
                severity=Severity.LOW,
            )
        ]
        self.assertEqual(violations_to_score(violations), 1)

    def test_derive_emergent_type_repeated_failure(self):
        """Repeated failure violations should map to repetitive type."""
        violations = [
            Violation(
                type=ViolationType.REPEATED_FAILURE,
                description="Agent repeats failed action",
                severity=Severity.HIGH,
            )
        ]
        self.assertEqual(derive_emergent_type(violations), "repetitive")

    def test_derive_emergent_type_ungrounded(self):
        """Ungrounded reference should map to misleading/fabrication."""
        violations = [
            Violation(
                type=ViolationType.UNGROUNDED_REFERENCE,
                description="Agent references non-existent element",
                severity=Severity.HIGH,
            )
        ]
        self.assertEqual(derive_emergent_type(violations), "misleading_or_fabrication")

    def test_derive_emergent_type_ignored_evidence(self):
        """Ignored evidence should map to unachievable/erroneous."""
        violations = [
            Violation(
                type=ViolationType.IGNORED_EVIDENCE,
                description="Agent ignores error message",
                severity=Severity.HIGH,
            )
        ]
        self.assertEqual(derive_emergent_type(violations), "unachievable_or_erroneous")

    def test_derive_emergent_type_none(self):
        """No violations should return 'none'."""
        self.assertEqual(derive_emergent_type([]), "none")


class TestGenericVerifier(unittest.TestCase):
    """Test the GenericHallucinationVerifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_verifier_initialization(self):
        """Test verifier initializes correctly."""
        with patch('verifier.generic_verifier.OpenAI'):
            verifier = GenericHallucinationVerifier(logger=self.logger)
            self.assertEqual(verifier.model_name, "gpt-4o-mini")
            self.assertEqual(verifier.model_temperature, 0.0)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_extract_input_context(self):
        """Test context extraction from result data."""
        with patch('verifier.generic_verifier.OpenAI'):
            verifier = GenericHallucinationVerifier(logger=self.logger)

            result_data = {
                "input": [
                    {
                        "content": [
                            {"type": "text", "text": "Task goal: Click the submit button"}
                        ]
                    }
                ],
                "task_goal": "Submit the form",
                "repetitive_action": "click(submit)",
            }

            context = verifier._extract_input_context(result_data)
            self.assertIn("Submit the form", context["task_goal"])
            self.assertIn("click(submit)", context["additional_context"])

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_evaluate_with_common_sense_reasonable(self):
        """Test evaluation returns reasonable for valid response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_reasonable": True,
            "violations": [],
            "score": 2,
            "reasoning": "The agent correctly identifies the submit button and clicks it.",
            "confidence": "high",
            "emergent_type": "none"
        })

        with patch('verifier.generic_verifier.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            verifier = GenericHallucinationVerifier(logger=self.logger)

            context = {
                "task_goal": "Submit the form",
                "current_state": "Form with submit button visible",
                "action_history": "",
                "additional_context": "",
            }

            evaluation = verifier._evaluate_with_common_sense(
                context=context,
                thinking="I see the submit button, I should click it.",
                action="click(submit_button)",
            )

            self.assertTrue(evaluation.is_reasonable)
            self.assertEqual(evaluation.score, 2)
            self.assertEqual(len(evaluation.violations), 0)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_evaluate_with_common_sense_hallucination(self):
        """Test evaluation detects hallucination."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_reasonable": False,
            "violations": [
                {
                    "type": "repeated_failure",
                    "description": "Agent repeats the same click action that failed 3 times",
                    "evidence_from_input": "Action history shows 3 failed attempts",
                    "evidence_from_response": "Agent says 'I will click the button'",
                    "severity": "high",
                    "suggested_category": "repetitive"
                }
            ],
            "score": 0,
            "reasoning": "The agent is repeating an action that has failed multiple times.",
            "confidence": "high",
            "emergent_type": "repetitive"
        })

        with patch('verifier.generic_verifier.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            verifier = GenericHallucinationVerifier(logger=self.logger)

            context = {
                "task_goal": "Submit the form",
                "current_state": "Form with submit button",
                "action_history": "click(submit) - failed, click(submit) - failed, click(submit) - failed",
                "additional_context": "",
            }

            evaluation = verifier._evaluate_with_common_sense(
                context=context,
                thinking="I will click the submit button.",
                action="click(submit)",
            )

            self.assertFalse(evaluation.is_reasonable)
            self.assertEqual(evaluation.score, 0)
            self.assertEqual(len(evaluation.violations), 1)
            self.assertEqual(evaluation.violations[0].type, ViolationType.REPEATED_FAILURE)
            self.assertEqual(evaluation.emergent_type, "repetitive")


class TestCommonSenseScenarios(unittest.TestCase):
    """Test common sense detection on various hallucination scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_scenario_ignored_error(self):
        """Test detection of ignored error message."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_reasonable": False,
            "violations": [
                {
                    "type": "ignored_evidence",
                    "description": "Agent ignores 'Permission denied' error",
                    "evidence_from_input": "Error: Permission denied",
                    "evidence_from_response": "Proceeding to delete the file",
                    "severity": "high",
                    "suggested_category": "erroneous"
                }
            ],
            "score": 0,
            "reasoning": "Agent ignores clear error message.",
            "confidence": "high",
            "emergent_type": "unachievable_or_erroneous"
        })

        with patch('verifier.generic_verifier.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            verifier = GenericHallucinationVerifier(logger=self.logger)

            context = {
                "task_goal": "Delete the file",
                "current_state": "Error: Permission denied",
                "action_history": "",
                "additional_context": "",
            }

            evaluation = verifier._evaluate_with_common_sense(
                context=context,
                thinking="I will delete the file now.",
                action="delete(file.txt)",
            )

            self.assertFalse(evaluation.is_reasonable)
            self.assertEqual(evaluation.violations[0].type, ViolationType.IGNORED_EVIDENCE)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_scenario_ungrounded_reference(self):
        """Test detection of reference to non-existent element."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_reasonable": False,
            "violations": [
                {
                    "type": "ungrounded_reference",
                    "description": "Agent clicks 'Checkout' button that doesn't exist",
                    "evidence_from_input": "Screen shows: Home, Products, Cart",
                    "evidence_from_response": "Clicking the Checkout button",
                    "severity": "high",
                    "suggested_category": "misleading"
                }
            ],
            "score": 0,
            "reasoning": "Agent references element not present.",
            "confidence": "high",
            "emergent_type": "misleading_or_fabrication"
        })

        with patch('verifier.generic_verifier.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            verifier = GenericHallucinationVerifier(logger=self.logger)

            context = {
                "task_goal": "Complete purchase",
                "current_state": "Navigation: Home, Products, Cart",
                "action_history": "",
                "additional_context": "",
            }

            evaluation = verifier._evaluate_with_common_sense(
                context=context,
                thinking="I see the Checkout button, clicking it.",
                action="click(Checkout)",
            )

            self.assertFalse(evaluation.is_reasonable)
            self.assertEqual(evaluation.violations[0].type, ViolationType.UNGROUNDED_REFERENCE)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_scenario_reasoning_mismatch(self):
        """Test detection of action contradicting reasoning."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_reasonable": False,
            "violations": [
                {
                    "type": "reasoning_mismatch",
                    "description": "Agent says product doesn't match but adds to cart anyway",
                    "evidence_from_input": "",
                    "evidence_from_response": "This isn't the right product... click(Add to Cart)",
                    "severity": "high",
                    "suggested_category": "inconsistent"
                }
            ],
            "score": 0,
            "reasoning": "Action contradicts stated reasoning.",
            "confidence": "high",
            "emergent_type": "inconsistent"
        })

        with patch('verifier.generic_verifier.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            verifier = GenericHallucinationVerifier(logger=self.logger)

            context = {
                "task_goal": "Buy blue shoes size 10",
                "current_state": "Product: Red shoes size 8",
                "action_history": "",
                "additional_context": "",
            }

            evaluation = verifier._evaluate_with_common_sense(
                context=context,
                thinking="This product doesn't match - it's red size 8, not blue size 10.",
                action="click(Add to Cart)",
            )

            self.assertFalse(evaluation.is_reasonable)
            self.assertEqual(evaluation.violations[0].type, ViolationType.REASONING_MISMATCH)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
