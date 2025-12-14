#!/usr/bin/env python3
"""
Tests for the Neuro-Symbolic Verifier.

Tests the fact extraction schema, Datalog rules, and integration.
Most tests do NOT require LLM API calls (mocked or use pre-computed facts).
"""

import os
import sys
import json
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from verifier.fact_schema import (
    ExtractedFacts,
    ActionSemantics,
    ActionComparison,
    PatternAwareness,
    ReasoningAssessment,
    AwarenessLevel,
    ConfidenceLevel,
    SemanticRelation,
    ReasoningQuality,
    IntentType,
    facts_to_datalog,
    write_datalog_facts,
)


class TestFactSchema(unittest.TestCase):
    """Test the fact extraction schema."""

    def test_extracted_facts_creation(self):
        """Test creating ExtractedFacts with all fields."""
        facts = ExtractedFacts(
            current_action=ActionSemantics(
                action_type="click",
                target="button#submit",
                parameters=None,
                intent=IntentType.CONTINUE,
                normalized_form="click(button#submit)"
            ),
            reference_action=ActionSemantics(
                action_type="click",
                target="button#submit",
                parameters=None,
                intent=IntentType.RETRY,
                normalized_form="click(button#submit)"
            ),
            action_comparison=ActionComparison(
                relation=SemanticRelation.IDENTICAL,
                confidence=ConfidenceLevel.HIGH,
                explanation="Same button click action"
            ),
            pattern_awareness=PatternAwareness(
                awareness_level=AwarenessLevel.NONE,
                evidence=[]
            ),
            reasoning_assessment=ReasoningAssessment(
                quality=ReasoningQuality.POOR,
                considers_alternatives=False,
                considers_history=False,
                identifies_issues=False,
                adapts_approach=False,
                key_insights=[],
                key_failures=["No awareness of repetition"]
            ),
            extraction_confidence=ConfidenceLevel.HIGH
        )

        self.assertEqual(facts.current_action.action_type, "click")
        self.assertEqual(facts.action_comparison.relation, SemanticRelation.IDENTICAL)
        self.assertEqual(facts.pattern_awareness.awareness_level, AwarenessLevel.NONE)

    def test_facts_to_datalog(self):
        """Test converting ExtractedFacts to Datalog format."""
        facts = ExtractedFacts(
            current_action=ActionSemantics(
                action_type="click",
                target="button",
                parameters=None,
                intent=IntentType.ADAPT,
                normalized_form="click(button)"
            ),
            action_comparison=ActionComparison(
                relation=SemanticRelation.DIFFERENT,
                confidence=ConfidenceLevel.HIGH,
                explanation="Different action"
            ),
            pattern_awareness=PatternAwareness(
                awareness_level=AwarenessLevel.EXPLICIT,
                recognized_pattern="repetitive clicking",
                response_to_pattern="trying different approach",
                evidence=["I've tried this before"]
            ),
            reasoning_assessment=ReasoningAssessment(
                quality=ReasoningQuality.GOOD,
                considers_alternatives=True,
                considers_history=True,
                identifies_issues=True,
                adapts_approach=True,
                key_insights=["Recognized the loop"],
                key_failures=[]
            ),
            extraction_confidence=ConfidenceLevel.HIGH
        )

        datalog_facts = facts_to_datalog(facts, "test_case_1")

        self.assertIn("action_relation", datalog_facts)
        self.assertIn("awareness_level", datalog_facts)
        self.assertEqual(datalog_facts["action_relation"][0][1], "different")
        self.assertEqual(datalog_facts["awareness_level"][0][1], "explicit")
        self.assertEqual(datalog_facts["considers_alternatives"][0][1], "true")


class TestDatalogRules(unittest.TestCase):
    """Test the Datalog rules with pre-computed facts."""

    def setUp(self):
        """Check if Soufflé is available."""
        try:
            result = subprocess.run(
                ["souffle", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.souffle_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.souffle_available = False

    def _run_souffle_test(self, facts_dict: dict, case_id: str) -> dict:
        """Run Soufflé with test facts and return results."""
        if not self.souffle_available:
            self.skipTest("Soufflé not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            facts_dir = tmpdir / "facts"
            output_dir = tmpdir / "output"
            facts_dir.mkdir()
            output_dir.mkdir()

            # Write facts
            write_datalog_facts(facts_dict, str(facts_dir))

            # Run Soufflé
            dl_file = Path(__file__).parent / "verifier" / "hybrid_repetitive.dl"
            result = subprocess.run(
                ["souffle", "-F", str(facts_dir), "-D", str(output_dir), str(dl_file)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                self.fail(f"Soufflé failed: {result.stderr}")

            # Parse output
            scores = {}
            scores_file = output_dir / "output_scores.csv"
            if scores_file.exists():
                with open(scores_file) as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3 and parts[0].strip('"') == case_id:
                            scores["thinking_score"] = int(parts[1])
                            scores["action_score"] = int(parts[2])

            return scores

    def test_identical_action_no_awareness_scores_0(self):
        """Test: identical action + no awareness = score 0."""
        facts_dict = {
            "action_relation": [("case1", "identical", "high")],
            "awareness_level": [("case1", "none")],
            "reasoning_quality": [("case1", "poor")],
            "considers_alternatives": [("case1", "false")],
            "considers_history": [("case1", "false")],
            "identifies_issues": [("case1", "false")],
            "adapts_approach": [("case1", "false")],
            "action_intent": [("case1", "retry")],
            "extraction_confidence": [("case1", "high")],
        }

        scores = self._run_souffle_test(facts_dict, "case1")
        self.assertEqual(scores.get("thinking_score"), 0)
        self.assertEqual(scores.get("action_score"), 0)

    def test_different_action_explicit_awareness_scores_2(self):
        """Test: different action + explicit awareness = score 2."""
        facts_dict = {
            "action_relation": [("case1", "different", "high")],
            "awareness_level": [("case1", "explicit")],
            "reasoning_quality": [("case1", "good")],
            "considers_alternatives": [("case1", "true")],
            "considers_history": [("case1", "true")],
            "identifies_issues": [("case1", "true")],
            "adapts_approach": [("case1", "true")],
            "action_intent": [("case1", "adapt")],
            "extraction_confidence": [("case1", "high")],
        }

        scores = self._run_souffle_test(facts_dict, "case1")
        self.assertEqual(scores.get("thinking_score"), 2)
        self.assertEqual(scores.get("action_score"), 1)

    def test_different_action_no_awareness_scores_1(self):
        """Test: different action + no explicit awareness = score 1."""
        facts_dict = {
            "action_relation": [("case1", "different", "high")],
            "awareness_level": [("case1", "none")],
            "reasoning_quality": [("case1", "adequate")],
            "considers_alternatives": [("case1", "false")],
            "considers_history": [("case1", "false")],
            "identifies_issues": [("case1", "false")],
            "adapts_approach": [("case1", "false")],
            "action_intent": [("case1", "explore")],
            "extraction_confidence": [("case1", "medium")],
        }

        scores = self._run_souffle_test(facts_dict, "case1")
        self.assertEqual(scores.get("thinking_score"), 1)
        self.assertEqual(scores.get("action_score"), 1)

    def test_equivalent_action_with_awareness_scores_1_0(self):
        """Test: equivalent (same semantic) action + awareness = thinking 1, action 0."""
        facts_dict = {
            "action_relation": [("case1", "equivalent", "high")],
            "awareness_level": [("case1", "implicit")],
            "reasoning_quality": [("case1", "adequate")],
            "considers_alternatives": [("case1", "true")],
            "considers_history": [("case1", "true")],
            "identifies_issues": [("case1", "false")],
            "adapts_approach": [("case1", "false")],
            "action_intent": [("case1", "retry")],
            "extraction_confidence": [("case1", "medium")],
        }

        scores = self._run_souffle_test(facts_dict, "case1")
        self.assertEqual(scores.get("thinking_score"), 1)  # Has awareness but still repeating
        self.assertEqual(scores.get("action_score"), 0)   # Equivalent = repetitive


class TestFactExtractorMocked(unittest.TestCase):
    """Test fact extractor with mocked LLM responses."""

    def test_fallback_extraction(self):
        """Test fallback extraction when LLM fails."""
        from verifier.fact_extractor import FactExtractor

        extractor = FactExtractor()

        # Test the fallback directly
        facts = extractor._create_fallback_extraction(
            thinking="Let me try clicking the button",
            action="click(button#submit)",
            reference_action="click(button#submit)"
        )

        self.assertEqual(facts.extraction_confidence, ConfidenceLevel.LOW)
        self.assertIn("Fallback", facts.extraction_notes)
        # Same action strings should be detected as identical
        self.assertEqual(facts.action_comparison.relation, SemanticRelation.IDENTICAL)

    def test_fallback_different_actions(self):
        """Test fallback detects different actions."""
        from verifier.fact_extractor import FactExtractor

        extractor = FactExtractor()

        facts = extractor._create_fallback_extraction(
            thinking="Let me try a different approach",
            action="scroll(down)",
            reference_action="click(button#submit)"
        )

        self.assertEqual(facts.action_comparison.relation, SemanticRelation.DIFFERENT)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def setUp(self):
        """Check dependencies."""
        try:
            result = subprocess.run(
                ["souffle", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.souffle_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.souffle_available = False

    def test_full_pipeline_with_mock_extraction(self):
        """Test full pipeline with pre-extracted facts."""
        if not self.souffle_available:
            self.skipTest("Soufflé not available")

        # Create mock ExtractedFacts
        facts = ExtractedFacts(
            current_action=ActionSemantics(
                action_type="type",
                target="input#search",
                parameters="new query",
                intent=IntentType.ADAPT,
                normalized_form="type(input#search, 'new query')"
            ),
            reference_action=ActionSemantics(
                action_type="type",
                target="input#search",
                parameters="old query",
                intent=IntentType.RETRY,
                normalized_form="type(input#search, 'old query')"
            ),
            action_comparison=ActionComparison(
                relation=SemanticRelation.SIMILAR,
                confidence=ConfidenceLevel.MEDIUM,
                explanation="Same input field, different query"
            ),
            pattern_awareness=PatternAwareness(
                awareness_level=AwarenessLevel.IMPLICIT,
                recognized_pattern="repeated searches",
                response_to_pattern="varying the query",
                evidence=["trying a different search"]
            ),
            reasoning_assessment=ReasoningAssessment(
                quality=ReasoningQuality.ADEQUATE,
                considers_alternatives=True,
                considers_history=False,
                identifies_issues=False,
                adapts_approach=True,
                key_insights=[],
                key_failures=[]
            ),
            extraction_confidence=ConfidenceLevel.MEDIUM
        )

        # Convert to Datalog
        datalog_facts = facts_to_datalog(facts, "integration_test")

        # Run through Soufflé
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            facts_dir = tmpdir / "facts"
            output_dir = tmpdir / "output"
            facts_dir.mkdir()
            output_dir.mkdir()

            write_datalog_facts(datalog_facts, str(facts_dir))

            dl_file = Path(__file__).parent / "verifier" / "hybrid_repetitive.dl"
            result = subprocess.run(
                ["souffle", "-F", str(facts_dir), "-D", str(output_dir), str(dl_file)],
                capture_output=True,
                text=True,
                timeout=30
            )

            self.assertEqual(result.returncode, 0, f"Soufflé failed: {result.stderr}")

            # Check output
            scores_file = output_dir / "output_scores.csv"
            self.assertTrue(scores_file.exists(), "Output scores file not created")


def main():
    """Run tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Neuro-Symbolic Verifier")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFactSchema))
    suite.addTests(loader.loadTestsFromTestCase(TestDatalogRules))
    suite.addTests(loader.loadTestsFromTestCase(TestFactExtractorMocked))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
