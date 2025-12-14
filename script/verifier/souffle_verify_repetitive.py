#!/usr/bin/env python3
"""
Soufflé-based verifier for repetitive action detection.

Uses Soufflé Datalog engine for declarative, logic-based verification.
This provides an alternative to the imperative Python-based logic verifier.
"""

import os
import re
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .base_verifier import BaseVerifier


class SouffleVerifyRepetitive(BaseVerifier):
    """
    Soufflé Datalog-based verifier for repetitive action hallucination detection.

    This verifier uses Soufflé to declaratively compute:
    - Whether the model's action matches the repetitive action
    - Whether awareness keywords are present in the thinking
    - Final scores based on logical rules
    """

    # Path to the Soufflé program
    SOUFFLE_PROGRAM = Path(__file__).parent / "souffle_repetitive.dl"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        force_verify: bool = False,
        model_name: Optional[str] = None,
        model_temperature: Optional[float] = None,
        result_field_name: Optional[str] = None,
    ):
        # Override defaults - Soufflé verifier doesn't use LLM
        super().__init__(
            logger,
            force_verify=force_verify,
            model_name=model_name,
            model_temperature=model_temperature,
            result_field_name=result_field_name or "verified_result",
        )

        # Verify Soufflé is available
        self._check_souffle_available()

        self.logger = logger or logging.getLogger(__name__)

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

    def _normalize_action(self, action: str) -> str:
        """Normalize action string for comparison."""
        if not action:
            return ""
        # Remove extra whitespace and normalize
        action = re.sub(r'\s+', ' ', action.strip())
        return action

    def _tokenize_thinking(self, thinking: str) -> List[str]:
        """Tokenize thinking text into lowercase words."""
        if not thinking:
            return []
        # Convert to lowercase and extract word tokens
        thinking = thinking.lower()
        tokens = re.findall(r'\b[a-z]+\b', thinking)
        return list(set(tokens))  # Unique tokens

    def _escape_souffle_string(self, s: str) -> str:
        """Escape a string for Soufflé fact format."""
        # Escape backslashes and quotes
        s = s.replace('\\', '\\\\')
        s = s.replace('"', '\\"')
        s = s.replace('\n', '\\n')
        s = s.replace('\t', '\\t')
        return s

    def _generate_facts(
        self,
        case_id: str,
        action: str,
        repetitive_action: str,
        thinking: str,
        facts_dir: Path
    ):
        """Generate Soufflé input fact files."""

        # Normalize inputs
        norm_action = self._normalize_action(action)
        norm_rep_action = self._normalize_action(repetitive_action)
        tokens = self._tokenize_thinking(thinking)

        escaped_id = self._escape_souffle_string(case_id)
        escaped_action = self._escape_souffle_string(norm_action)
        escaped_rep_action = self._escape_souffle_string(norm_rep_action)

        # Write input_action.facts
        with open(facts_dir / "input_action.facts", "w") as f:
            f.write(f'"{escaped_id}"\t"{escaped_action}"\n')

        # Write input_repetitive_action.facts
        with open(facts_dir / "input_repetitive_action.facts", "w") as f:
            f.write(f'"{escaped_id}"\t"{escaped_rep_action}"\n')

        # Write thinking_token.facts
        with open(facts_dir / "thinking_token.facts", "w") as f:
            for token in tokens:
                escaped_token = self._escape_souffle_string(token)
                f.write(f'"{escaped_id}"\t"{escaped_token}"\n')

    def _run_souffle(self, facts_dir: Path, output_dir: Path) -> bool:
        """Run Soufflé on the generated facts."""
        try:
            result = subprocess.run(
                [
                    "souffle",
                    "-F", str(facts_dir),  # Input facts directory
                    "-D", str(output_dir),  # Output directory
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

    def _parse_output(self, output_dir: Path, case_id: str) -> Dict[str, Any]:
        """Parse Soufflé output files."""
        result = {
            "thinking_eval": None,
            "action_eval": None,
            "action_matches": False,
            "awareness_keywords": [],
            "verifier_type": "souffle"
        }

        # Parse output_score.csv
        score_file = output_dir / "output_score.csv"
        if score_file.exists():
            with open(score_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        # Remove quotes from case_id
                        file_case_id = parts[0].strip('"')
                        if file_case_id == case_id:
                            result["thinking_eval"] = int(parts[1])
                            result["action_eval"] = int(parts[2])

        # Parse output_action_matches.csv
        matches_file = output_dir / "output_action_matches.csv"
        if matches_file.exists():
            with open(matches_file) as f:
                for line in f:
                    file_case_id = line.strip().strip('"')
                    if file_case_id == case_id:
                        result["action_matches"] = True

        # Parse output_awareness_keywords.csv
        keywords_file = output_dir / "output_awareness_keywords.csv"
        if keywords_file.exists():
            with open(keywords_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        file_case_id = parts[0].strip('"')
                        if file_case_id == case_id:
                            keyword = parts[1].strip('"')
                            result["awareness_keywords"].append(keyword)

        # Generate reason
        if result["action_matches"]:
            if result["awareness_keywords"]:
                result["thinking_eval_reason"] = (
                    f"Action matches repetitive action; "
                    f"Awareness keywords found: {result['awareness_keywords']}"
                )
            else:
                result["thinking_eval_reason"] = "Action matches repetitive action"
        else:
            if result["awareness_keywords"]:
                result["thinking_eval_reason"] = (
                    f"Action differs; Awareness keywords: {result['awareness_keywords']}"
                )
            else:
                result["thinking_eval_reason"] = "Action differs from repetitive action"

        return result

    def _evaluate_thinking(
        self, thinking: str, action: str, **kwargs
    ) -> Tuple[int, str]:
        """
        Evaluate the model's thinking using Soufflé.

        Returns:
            Tuple of (score, reason)
        """
        case_id = kwargs.get("task_name", "unknown")
        repetitive_action = kwargs.get("repetitive_action", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            facts_dir = tmpdir / "facts"
            output_dir = tmpdir / "output"
            facts_dir.mkdir()
            output_dir.mkdir()

            # Generate input facts
            self._generate_facts(
                case_id=case_id,
                action=action,
                repetitive_action=repetitive_action,
                thinking=thinking,
                facts_dir=facts_dir
            )

            # Run Soufflé
            success = self._run_souffle(facts_dir, output_dir)

            if not success:
                # Fallback to simple comparison
                self.logger.warning("Soufflé failed, using fallback logic")
                norm_action = self._normalize_action(action)
                norm_rep = self._normalize_action(repetitive_action)
                if norm_action == norm_rep:
                    return 0, "Soufflé failed; Fallback: action matches"
                else:
                    return 1, "Soufflé failed; Fallback: action differs"

            # Parse output
            result = self._parse_output(output_dir, case_id)

            return result

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> int:
        """
        Evaluate the model's action (fallback for base class compatibility).
        """
        repetitive_action = kwargs.get("repetitive_action", "")
        norm_action = self._normalize_action(action)
        norm_rep = self._normalize_action(repetitive_action)
        return 0 if norm_action == norm_rep else 1

    def _process_single_result(
        self, result_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single inference result using Soufflé.
        Required by BaseVerifier abstract interface.
        """
        task_name = result_data.get("task_name", "unknown")

        # Check if we should verify this result
        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")

        # Extract fields
        result_info = result_data.get("result", {})
        thinking = result_info.get("thinking", "")
        action = result_info.get("action", "")
        repetitive_action = result_data.get("repetitive_action", "")

        try:
            # Run Soufflé-based evaluation - returns full result dict
            souffle_result = self._evaluate_thinking(
                thinking=thinking,
                action=action,
                task_name=task_name,
                repetitive_action=repetitive_action
            )

            # Build verification result from Soufflé output
            verified_result = {
                "thinking_eval": souffle_result.get("thinking_eval", 0),
                "action_eval": souffle_result.get("action_eval", 0),
                "thinking_eval_reason": souffle_result.get("thinking_eval_reason", "Unknown"),
                "verifier_type": "souffle",
                "awareness_keywords": souffle_result.get("awareness_keywords", []),
                "action_matches": souffle_result.get("action_matches", False),
            }

            # Add to original result and save
            result_data[self.result_field_name] = verified_result
            self.save_results(result_data)
            return result_data

        except Exception as e:
            self.logger.error(f"Error processing {task_name}: {e}")
            return result_data

    def _process_inference_results(self) -> None:
        """
        Process all inference results using Soufflé verifier.
        Overrides base class to use the same pattern.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        self.logger.info(
            f"Processing {len(self.inference_results)} inference results with parallel processing (max workers: 20)"
        )

        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_result = {
                executor.submit(self._process_single_result, result_data): result_data
                for result_data in self.inference_results
            }

            for future in tqdm(
                as_completed(future_to_result), total=len(self.inference_results)
            ):
                result = future.result()
                if result:
                    self.verified_results.append(result)
