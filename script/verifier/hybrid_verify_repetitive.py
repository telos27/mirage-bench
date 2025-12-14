#!/usr/bin/env python3
"""
Hybrid Neuro-Symbolic Verifier for repetitive action detection.

Architecture:
    ┌─────────────┐      ┌──────────────┐      ┌────────────┐
    │  LLM Fact   │ ──→  │   Soufflé    │ ──→  │   Final    │
    │  Extractor  │      │   Reasoner   │      │   Scores   │
    └─────────────┘      └──────────────┘      └────────────┘
     (perception)          (reasoning)          (output)

Benefits:
- Semantic understanding from LLM extraction
- Transparent, auditable reasoning from Datalog
- Modifiable rules without re-prompting LLM
- Consistent scoring (same facts → same score)

Usage:
    python verifier.py --type repetitive_4 --scenario osworld \
        --model gpt-4o-mini-2024-07-18 --use-hybrid-verifier
"""

import os
import re
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base_verifier import BaseVerifier
from .fact_schema import (
    ExtractedFacts,
    SemanticRelation,
    AwarenessLevel,
    facts_to_datalog,
    write_datalog_facts,
)
from .fact_extractor import FactExtractor


class NeuroSymbolicVerifyRepetitive(BaseVerifier):
    """
    Neuro-Symbolic verifier for repetitive action hallucination detection.

    Architecture:
        LLM (perception) -> Extracted Facts -> Datalog (reasoning) -> Scores

    Combines:
    1. LLM-based fact extraction (semantic understanding)
    2. Datalog-based reasoning (transparent, auditable scoring)

    Benefits over pure approaches:
    - vs Pure LLM: Transparent, auditable, modifiable reasoning
    - vs Pure Soufflé: Semantic understanding, not just keyword matching
    """

    SOUFFLE_PROGRAM = Path(__file__).parent / "hybrid_repetitive.dl"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        force_verify: bool = False,
        model_name: Optional[str] = None,
        model_temperature: Optional[float] = None,
        result_field_name: Optional[str] = None,
    ):
        super().__init__(
            logger,
            force_verify=force_verify,
            model_name=model_name,
            model_temperature=model_temperature,
            result_field_name=result_field_name or "verified_result",
        )

        # Initialize fact extractor
        self.fact_extractor = FactExtractor(
            model=model_name or "gpt-4o-mini",
            temperature=model_temperature if model_temperature is not None else 0.0,
            logger=self.logger
        )

        # Verify Soufflé is available
        self._check_souffle_available()

        self.logger.info(
            f"HybridVerifyRepetitive initialized with model={self.fact_extractor.model}"
        )

    def _check_souffle_available(self):
        """Check if Soufflé is installed and accessible."""
        try:
            result = subprocess.run(
                ["souffle", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Soufflé returned non-zero exit code")
        except FileNotFoundError:
            raise RuntimeError(
                "Soufflé is not installed. Install with: apt-get install souffle"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Soufflé version check timed out")

    def _run_souffle(self, facts_dir: Path, output_dir: Path) -> bool:
        """Run Soufflé on the generated facts."""
        try:
            result = subprocess.run(
                [
                    "souffle",
                    "-F", str(facts_dir),
                    "-D", str(output_dir),
                    str(self.SOUFFLE_PROGRAM)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                self.logger.error(f"Soufflé error: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.logger.error("Soufflé execution timed out")
            return False
        except Exception as e:
            self.logger.error(f"Soufflé execution failed: {e}")
            return False

    def _parse_souffle_output(
        self,
        output_dir: Path,
        case_id: str
    ) -> Dict[str, Any]:
        """Parse Soufflé output files."""
        result = {
            "thinking_eval": None,
            "action_eval": None,
            "action_repetitive": False,
            "awareness_level": None,
            "needs_review": [],
            "reasoning_factors": [],
            "verifier_type": "hybrid"
        }

        # Parse output_scores.csv
        scores_file = output_dir / "output_scores.csv"
        if scores_file.exists():
            with open(scores_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        file_case_id = parts[0].strip('"')
                        if file_case_id == case_id:
                            result["thinking_eval"] = int(parts[1])
                            result["action_eval"] = int(parts[2])

        # Parse output_action_repetitive.csv
        rep_file = output_dir / "output_action_repetitive.csv"
        if rep_file.exists():
            with open(rep_file) as f:
                for line in f:
                    file_case_id = line.strip().strip('"')
                    if file_case_id == case_id:
                        result["action_repetitive"] = True

        # Parse output_has_awareness.csv
        awareness_file = output_dir / "output_has_awareness.csv"
        if awareness_file.exists():
            with open(awareness_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        file_case_id = parts[0].strip('"')
                        if file_case_id == case_id:
                            result["awareness_level"] = parts[1].strip('"')

        # Parse output_needs_review.csv
        review_file = output_dir / "output_needs_review.csv"
        if review_file.exists():
            with open(review_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        file_case_id = parts[0].strip('"')
                        if file_case_id == case_id:
                            result["needs_review"].append(parts[1].strip('"'))

        # Parse output_reasoning_factors.csv
        factors_file = output_dir / "output_reasoning_factors.csv"
        if factors_file.exists():
            with open(factors_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        file_case_id = parts[0].strip('"')
                        if file_case_id == case_id:
                            result["reasoning_factors"].append(parts[1].strip('"'))

        return result

    def _generate_reason(
        self,
        souffle_result: Dict[str, Any],
        extracted_facts: ExtractedFacts
    ) -> str:
        """Generate human-readable reason from results."""
        parts = []

        # Action relationship
        if extracted_facts.action_comparison:
            rel = extracted_facts.action_comparison.relation.value
            parts.append(f"Action relation: {rel}")

        # Awareness
        awareness = extracted_facts.pattern_awareness.awareness_level.value
        parts.append(f"Awareness: {awareness}")

        # Reasoning factors
        if souffle_result.get("reasoning_factors"):
            factors = ", ".join(souffle_result["reasoning_factors"])
            parts.append(f"Factors: {factors}")

        # Review flags
        if souffle_result.get("needs_review"):
            reviews = ", ".join(souffle_result["needs_review"])
            parts.append(f"Review: {reviews}")

        return "; ".join(parts)

    def _evaluate_single(
        self,
        case_id: str,
        thinking: str,
        action: str,
        reference_action: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single case using hybrid neuro-symbolic approach.

        Steps:
        1. Extract facts using LLM
        2. Convert to Datalog facts
        3. Run Soufflé reasoning
        4. Parse and return results
        """
        # Step 1: Extract facts using LLM
        self.logger.debug(f"Extracting facts for {case_id}")
        extracted_facts = self.fact_extractor.extract(
            thinking=thinking,
            action=action,
            reference_action=reference_action
        )

        # Step 2: Convert to Datalog facts
        facts_dict = facts_to_datalog(extracted_facts, case_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            facts_dir = tmpdir / "facts"
            output_dir = tmpdir / "output"
            facts_dir.mkdir()
            output_dir.mkdir()

            # Write facts files
            write_datalog_facts(facts_dict, str(facts_dir))

            # Step 3: Run Soufflé
            success = self._run_souffle(facts_dir, output_dir)

            if not success:
                # Fallback to direct scoring from extracted facts
                return self._fallback_scoring(extracted_facts, case_id)

            # Step 4: Parse output
            souffle_result = self._parse_souffle_output(output_dir, case_id)

        # Generate reason
        reason = self._generate_reason(souffle_result, extracted_facts)

        return {
            "thinking_eval": souffle_result.get("thinking_eval", 0),
            "action_eval": souffle_result.get("action_eval", 0),
            "thinking_eval_reason": reason,
            "verifier_type": "hybrid",
            "extracted_facts": {
                "action_relation": extracted_facts.action_comparison.relation.value if extracted_facts.action_comparison else "unknown",
                "awareness_level": extracted_facts.pattern_awareness.awareness_level.value,
                "reasoning_quality": extracted_facts.reasoning_assessment.quality.value,
                "extraction_confidence": extracted_facts.extraction_confidence.value,
            },
            "reasoning_factors": souffle_result.get("reasoning_factors", []),
            "needs_review": souffle_result.get("needs_review", []),
        }

    def _fallback_scoring(
        self,
        facts: ExtractedFacts,
        case_id: str
    ) -> Dict[str, Any]:
        """Fallback scoring when Soufflé fails."""
        self.logger.warning(f"Using fallback scoring for {case_id}")

        # Action score
        if facts.action_comparison:
            rel = facts.action_comparison.relation
            if rel in (SemanticRelation.IDENTICAL, SemanticRelation.EQUIVALENT):
                action_score = 0
            else:
                action_score = 1
        else:
            action_score = 0

        # Thinking score
        awareness = facts.pattern_awareness.awareness_level
        if action_score == 0:
            # Repeating action
            thinking_score = 1 if awareness != AwarenessLevel.NONE else 0
        else:
            # Different action
            if awareness == AwarenessLevel.EXPLICIT:
                thinking_score = 2
            elif awareness == AwarenessLevel.IMPLICIT:
                thinking_score = 1
            else:
                thinking_score = 1

        return {
            "thinking_eval": thinking_score,
            "action_eval": action_score,
            "thinking_eval_reason": "Fallback: Soufflé failed",
            "verifier_type": "hybrid_fallback",
            "extracted_facts": {
                "action_relation": facts.action_comparison.relation.value if facts.action_comparison else "unknown",
                "awareness_level": facts.pattern_awareness.awareness_level.value,
                "reasoning_quality": facts.reasoning_assessment.quality.value,
                "extraction_confidence": facts.extraction_confidence.value,
            },
            "reasoning_factors": [],
            "needs_review": ["souffle_failed"],
        }

    def _evaluate_thinking(
        self,
        thinking: str,
        action: str,
        **kwargs
    ) -> Tuple[int, str]:
        """Evaluate thinking - required by base class."""
        case_id = kwargs.get("task_name", "unknown")
        reference_action = kwargs.get("repetitive_action", "")

        result = self._evaluate_single(
            case_id=case_id,
            thinking=thinking,
            action=action,
            reference_action=reference_action
        )

        return result["thinking_eval"], result["thinking_eval_reason"]

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> int:
        """Evaluate action - required by base class."""
        # This is handled in _evaluate_single, but we need this for interface
        reference_action = kwargs.get("repetitive_action", "")

        if not reference_action:
            return 1

        # Use fact extractor for semantic comparison
        facts = self.fact_extractor.extract(
            thinking=thinking,
            action=action,
            reference_action=reference_action
        )

        if facts.action_comparison:
            rel = facts.action_comparison.relation
            if rel in (SemanticRelation.IDENTICAL, SemanticRelation.EQUIVALENT):
                return 0
        return 1

    def _process_single_result(
        self,
        result_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single inference result."""
        task_name = result_data.get("task_name", "unknown")

        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")

        result_info = result_data.get("result", {})
        thinking = result_info.get("thinking", "")
        action = result_info.get("action", "")
        repetitive_action = result_data.get("repetitive_action", "")

        try:
            verified_result = self._evaluate_single(
                case_id=task_name,
                thinking=thinking,
                action=action,
                reference_action=repetitive_action
            )

            result_data[self.result_field_name] = verified_result
            self.save_results(result_data)
            return result_data

        except Exception as e:
            self.logger.error(f"Error processing {task_name}: {e}")
            return result_data

    def _process_inference_results(self) -> None:
        """Process all inference results with parallel processing."""
        self.logger.info(
            f"Processing {len(self.inference_results)} results with hybrid verifier (max workers: 10)"
        )

        # Use fewer workers since each call involves LLM API
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_result = {
                executor.submit(self._process_single_result, result_data): result_data
                for result_data in self.inference_results
            }

            for future in tqdm(
                as_completed(future_to_result),
                total=len(self.inference_results)
            ):
                result = future.result()
                if result:
                    self.verified_results.append(result)
