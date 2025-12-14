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
from verifier.generic_fact_extractor import (
    GenericFactExtractor,
    ExtractedInputFacts,
    ExtractedOutputFacts,
    ExtractionResult,
    ConsistencyResult,
    ConsistencyViolation,
    check_consistency,
)


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


class TestFactExtractor(unittest.TestCase):
    """Test the GenericFactExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    def test_extracted_input_facts_model(self):
        """Test ExtractedInputFacts model."""
        facts = ExtractedInputFacts(
            task_goal="Submit the form",
            visible_elements=["Submit button", "Cancel button"],
            error_messages=["Error: Invalid email"],
            state_info=["Form page"],
            action_history=["Typed email"],
            important_facts=["Email validation failed"],
        )
        self.assertEqual(facts.task_goal, "Submit the form")
        self.assertEqual(len(facts.visible_elements), 2)
        self.assertEqual(len(facts.error_messages), 1)

    def test_extracted_output_facts_model(self):
        """Test ExtractedOutputFacts model."""
        facts = ExtractedOutputFacts(
            stated_observations=["I see the submit button"],
            reasoning_steps=["The form is ready", "I should submit"],
            stated_intent="Submit the form",
            action_target="submit_button",
            action_type="click",
            references_made=["submit button", "form"],
        )
        self.assertEqual(facts.action_type, "click")
        self.assertEqual(facts.action_target, "submit_button")
        self.assertEqual(len(facts.reasoning_steps), 2)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_fact_extractor_initialization(self):
        """Test fact extractor initializes correctly."""
        with patch('verifier.generic_fact_extractor.OpenAI'):
            extractor = GenericFactExtractor(logger=self.logger)
            self.assertEqual(extractor.model, "gpt-4o-mini")
            self.assertEqual(extractor.temperature, 0.0)


class TestConsistencyCheck(unittest.TestCase):
    """Test the consistency check functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    def test_consistency_result_model(self):
        """Test ConsistencyResult model."""
        result = ConsistencyResult(
            is_consistent=False,
            violations=[
                ConsistencyViolation(
                    violation_type="ungrounded_reference",
                    description="Agent references checkout button not visible",
                    input_evidence="Visible: Home, Cart",
                    output_evidence="Clicking checkout button",
                    severity="high",
                )
            ],
            score=0,
            reasoning="Agent references non-existent element",
        )
        self.assertFalse(result.is_consistent)
        self.assertEqual(result.score, 0)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].violation_type, "ungrounded_reference")

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_check_consistency_consistent(self):
        """Test consistency check with consistent input/output."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.parsed = ConsistencyResult(
            is_consistent=True,
            violations=[],
            score=2,
            reasoning="Agent references match visible elements",
        )

        with patch('verifier.generic_fact_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.return_value = mock_response
            mock_openai.return_value = mock_client

            input_facts = ExtractedInputFacts(
                task_goal="Click submit",
                visible_elements=["Submit button", "Cancel button"],
            )
            output_facts = ExtractedOutputFacts(
                stated_observations=["I see the submit button"],
                action_target="Submit button",
                action_type="click",
            )

            result = check_consistency(
                input_facts, output_facts, mock_client, logger=self.logger
            )

            self.assertTrue(result.is_consistent)
            self.assertEqual(result.score, 2)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_check_consistency_inconsistent(self):
        """Test consistency check with inconsistent input/output."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.parsed = ConsistencyResult(
            is_consistent=False,
            violations=[
                ConsistencyViolation(
                    violation_type="ungrounded_reference",
                    description="Agent clicks Checkout but it's not visible",
                    input_evidence="Visible: Home, Cart",
                    output_evidence="click(Checkout)",
                    severity="high",
                )
            ],
            score=0,
            reasoning="Agent references non-existent element",
        )

        with patch('verifier.generic_fact_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.return_value = mock_response
            mock_openai.return_value = mock_client

            input_facts = ExtractedInputFacts(
                task_goal="Go to checkout",
                visible_elements=["Home button", "Cart button"],
            )
            output_facts = ExtractedOutputFacts(
                stated_observations=["I see the checkout button"],
                action_target="Checkout",
                action_type="click",
                references_made=["Checkout button"],
            )

            result = check_consistency(
                input_facts, output_facts, mock_client, logger=self.logger
            )

            self.assertFalse(result.is_consistent)
            self.assertEqual(result.score, 0)
            self.assertEqual(len(result.violations), 1)


class TestTwoStepVerification(unittest.TestCase):
    """Test the two-step verification process."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_two_step_verification_detects_hallucination(self):
        """Test two-step verification detects hallucination."""
        # Mock extraction result
        mock_extraction = ExtractionResult(
            input_facts=ExtractedInputFacts(
                task_goal="Submit form",
                visible_elements=["Home", "Cart"],
                error_messages=[],
                action_history=["click(submit) - failed", "click(submit) - failed"],
            ),
            output_facts=ExtractedOutputFacts(
                stated_observations=["I see the submit button"],
                action_target="submit",
                action_type="click",
            ),
        )

        # Mock consistency check result
        mock_consistency = ConsistencyResult(
            is_consistent=False,
            violations=[
                ConsistencyViolation(
                    violation_type="ignored_history",
                    description="Agent ignores that action failed twice",
                    severity="high",
                )
            ],
            score=0,
            reasoning="Agent repeats failed action",
        )

        # Mock common sense check result
        mock_common_sense_response = Mock()
        mock_common_sense_response.choices = [Mock()]
        mock_common_sense_response.choices[0].message.content = json.dumps({
            "is_reasonable": False,
            "violations": [
                {
                    "type": "repeated_failure",
                    "description": "Agent repeats action that failed twice",
                    "severity": "high",
                    "suggested_category": "repetitive"
                }
            ],
            "score": 0,
            "reasoning": "Repeating failed action is not reasonable",
            "confidence": "high",
            "emergent_type": "repetitive"
        })

        with patch('verifier.generic_verifier.OpenAI') as mock_openai:
            with patch('verifier.generic_fact_extractor.OpenAI'):
                mock_client = Mock()
                mock_openai.return_value = mock_client

                # Mock fact extraction
                mock_extract_response = Mock()
                mock_extract_response.choices = [Mock()]
                mock_extract_response.choices[0].message.parsed = mock_extraction.input_facts

                # Mock consistency
                mock_consistency_response = Mock()
                mock_consistency_response.choices = [Mock()]
                mock_consistency_response.choices[0].message.parsed = mock_consistency

                mock_client.beta.chat.completions.parse.side_effect = [
                    mock_extract_response,  # input extraction
                    Mock(choices=[Mock(message=Mock(parsed=mock_extraction.output_facts))]),  # output extraction
                    mock_consistency_response,  # consistency check
                ]

                mock_client.chat.completions.create.return_value = mock_common_sense_response

                verifier = GenericHallucinationVerifier(logger=self.logger)

                # Manually test the _verify_two_step method
                # We'll patch the extractor's extract method
                verifier.fact_extractor.extract = Mock(return_value=mock_extraction)

                # Patch check_consistency
                with patch('verifier.generic_verifier.check_consistency', return_value=mock_consistency):
                    result = verifier._verify_two_step(
                        input_text="Task: Submit form. Visible: Home, Cart",
                        thinking="I will click submit",
                        action="click(submit)",
                    )

                self.assertEqual(result["thinking_eval"], 0)
                self.assertTrue(result["is_hallucination"])
                self.assertEqual(result["verifier_type"], "generic_two_step")
                self.assertFalse(result["consistency_check"]["is_consistent"])
                self.assertFalse(result["common_sense_check"]["is_reasonable"])


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
