#!/usr/bin/env python3
"""
Hybrid Generic Verifier: Soufflé (Step 1) + LLM (Step 2).

Step 1 (Consistency): Uses Soufflé Datalog for deterministic, free checking
Step 2 (Common Sense): Uses LLM for nuanced reasoning

Architecture:
    Input/Output -> LLM Fact Extraction -> Soufflé (consistency) -> LLM (common sense) -> Result
"""

import os
import re
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_verifier import BaseVerifier
from .generic_fact_extractor import (
    GenericFactExtractor,
    ExtractedInputFacts,
    ExtractedOutputFacts,
)
from .heuristic_fact_extractor import (
    HeuristicFactExtractor,
    HeuristicInputFacts,
    extract_output_facts_heuristic,
)
from .generic_schema import (
    CommonSenseEvaluation,
    Violation,
    ViolationType,
    Severity,
    ConfidenceLevel,
    derive_emergent_type,
)


class SouffleGenericVerifier(BaseVerifier):
    """
    Hybrid generic verifier using Soufflé for consistency + LLM for common sense.

    Step 1 (Soufflé - FREE):
        - Ungrounded references
        - Ignored errors
        - Repeated failed actions
        - Target not visible

    Step 2 (LLM):
        - Does the behavior make sense?
        - Goal alignment
        - Reasoning quality
    """

    SOUFFLE_PROGRAM = Path(__file__).parent / "generic_consistency.dl"

    COMMON_SENSE_PROMPT = """Given these facts and consistency check results, evaluate if the agent's behavior makes common sense.

## Input Facts
- Task Goal: {task_goal}
- Visible Elements: {visible_elements}
- Error Messages: {error_messages}
- Action History: {action_history}

## Output Facts
- Stated Observations: {stated_observations}
- Action: {action_type} on {action_target}
- References Made: {references_made}

## Consistency Check Result (from rule-based system)
- Is Consistent: {is_consistent}
- Score: {consistency_score}
- Violations Found: {violations}

## Common Sense Questions
1. Given the goal, does this action make progress?
2. Is the agent's reasoning sound?
3. Does the action contradict the stated reasoning?
4. Is this what a reasonable agent would do?

Respond with JSON:
{{
    "is_reasonable": boolean,
    "violations": [
        {{
            "type": "reasoning_mismatch|goal_deviation|state_confusion|other",
            "description": "string",
            "severity": "low|medium|high"
        }}
    ],
    "score": 0-2,
    "reasoning": "explanation",
    "confidence": "low|medium|high"
}}"""

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
            model_name=model_name or "gpt-4o-mini",
            model_temperature=model_temperature if model_temperature is not None else 0.0,
            result_field_name=result_field_name or "generic_verified_result",
        )

        self._check_souffle_available()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        # LLM-based extractor for output facts (semantic understanding)
        self.llm_fact_extractor = GenericFactExtractor(
            model=self.model_name,
            temperature=self.model_temperature,
            logger=self.logger,
        )

        # Heuristic extractor for input facts (FREE, complete)
        self.heuristic_extractor = HeuristicFactExtractor(logger=self.logger)

        self.logger.info(
            f"SouffleGenericVerifier initialized (Heuristic + Soufflé + {self.model_name})"
        )

    def _check_souffle_available(self):
        """Check if Soufflé is installed."""
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

    def _escape_souffle_string(self, s: str) -> str:
        """Escape string for Soufflé fact format."""
        if not s:
            return ""
        s = s.replace('\\', '\\\\')
        s = s.replace('"', '\\"')
        s = s.replace('\n', ' ')
        s = s.replace('\t', ' ')
        return s[:200]  # Limit length

    def _normalize_for_matching(self, s: str) -> str:
        """Normalize string for matching (lowercase, simplified)."""
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s

    def _extract_key_terms(self, s: str) -> List[str]:
        """Extract key terms from a string for fuzzy matching."""
        if not s:
            return []
        s = s.lower().strip()
        # Remove common suffixes
        s = re.sub(r'\s+(link|button|page|section|tab|element|field|input|text|area|box|table|menu|panel|item|icon)s?$', '', s)
        s = re.sub(r'^(the|a|an)\s+', '', s)
        # Get normalized form
        s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return [s] if s else []

    def _stem_word(self, word: str) -> str:
        """Simple stemming - handle common English plurals and suffixes."""
        word = word.lower()
        # Plural rules (order matters)
        if word.endswith('ies') and len(word) > 3:
            return word[:-3] + 'y'  # quantities -> quantity
        if word.endswith('es') and len(word) > 2:
            return word[:-2]  # boxes -> box
        if word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
            return word[:-1]  # products -> product
        if word.endswith('ing') and len(word) > 4:
            return word[:-3]  # selling -> sell
        if word.endswith('ed') and len(word) > 3:
            return word[:-2]  # formatted -> format
        return word

    def _get_word_stems(self, s: str) -> set:
        """Get stemmed versions of all words in a string."""
        if not s:
            return set()
        words = re.findall(r'[a-z]+', s.lower())
        return {self._stem_word(w) for w in words if len(w) > 2}

    def _reference_matches_visible(self, ref: str, visible_elements: List[str]) -> bool:
        """Check if a reference matches any visible element using fuzzy matching."""
        ref_normalized = self._normalize_for_matching(ref)
        ref_terms = self._extract_key_terms(ref)
        ref_stems = self._get_word_stems(ref)

        for elem in visible_elements:
            elem_normalized = self._normalize_for_matching(elem)
            elem_stems = self._get_word_stems(elem)

            # Exact match
            if ref_normalized == elem_normalized:
                return True

            # Reference is contained in element or vice versa
            if ref_normalized and elem_normalized:
                if ref_normalized in elem_normalized or elem_normalized in ref_normalized:
                    return True

            # Key term match
            for term in ref_terms:
                if term and len(term) > 2 and term in elem_normalized:
                    return True

            # Word stem overlap - if any significant stems match
            if ref_stems and elem_stems:
                common_stems = ref_stems & elem_stems
                # If there's a matching stem that's significant (not a stopword)
                significant_stems = {s for s in common_stems if len(s) > 3}
                if significant_stems:
                    return True

        return False

    def _generate_facts_hybrid(
        self,
        case_id: str,
        input_facts: HeuristicInputFacts,
        output_facts: Dict[str, Any],
        current_action: str,
        facts_dir: Path
    ):
        """Generate Soufflé input fact files from heuristic + LLM extracted facts."""
        esc_id = self._escape_souffle_string(case_id)

        # visible_element.facts
        with open(facts_dir / "visible_element.facts", "w") as f:
            for elem in input_facts.visible_elements:
                norm_elem = self._normalize_for_matching(elem)
                if norm_elem:
                    f.write(f'"{esc_id}"\t"{self._escape_souffle_string(norm_elem)}"\n')

        # error_message.facts
        with open(facts_dir / "error_message.facts", "w") as f:
            for msg in input_facts.error_messages:
                if msg:
                    f.write(f'"{esc_id}"\t"{self._escape_souffle_string(msg)}"\n')

        # action_history.facts
        with open(facts_dir / "action_history.facts", "w") as f:
            for hist in input_facts.action_history:
                # Handle both tuple (action, outcome) and string formats
                if isinstance(hist, tuple):
                    action_str, outcome = hist
                    norm_hist = self._normalize_for_matching(action_str)
                else:
                    norm_hist = self._normalize_for_matching(hist)
                    # Try to detect outcome from history text
                    outcome = "unknown"
                    hist_lower = hist.lower()
                    if "failed" in hist_lower or "error" in hist_lower:
                        outcome = "failed"
                    elif "timeout" in hist_lower:
                        outcome = "timeout"
                    elif "success" in hist_lower:
                        outcome = "success"
                if norm_hist:
                    f.write(f'"{esc_id}"\t"{self._escape_souffle_string(norm_hist)}"\t"{outcome}"\n')

        # task_goal.facts
        with open(facts_dir / "task_goal.facts", "w") as f:
            goal = self._escape_souffle_string(input_facts.task_goal or "unknown")
            f.write(f'"{esc_id}"\t"{goal}"\n')

        # state_info.facts
        with open(facts_dir / "state_info.facts", "w") as f:
            for info in input_facts.state_info:
                if info:
                    f.write(f'"{esc_id}"\t"{self._escape_souffle_string(info)}"\n')

        # reference_made.facts - only include references that DON'T match visible elements
        # This is done here because Datalog exact matching is too strict
        with open(facts_dir / "reference_made.facts", "w") as f:
            for ref in output_facts.get("references_made", []):
                # Check if this reference matches any visible element using fuzzy matching
                if not self._reference_matches_visible(ref, input_facts.visible_elements):
                    norm_ref = self._normalize_for_matching(ref)
                    if norm_ref:
                        f.write(f'"{esc_id}"\t"{self._escape_souffle_string(norm_ref)}"\n')

        # stated_observation.facts
        with open(facts_dir / "stated_observation.facts", "w") as f:
            for obs in output_facts.get("stated_observations", []):
                if obs:
                    f.write(f'"{esc_id}"\t"{self._escape_souffle_string(obs)}"\n')

        # action_target.facts
        with open(facts_dir / "action_target.facts", "w") as f:
            target = self._normalize_for_matching(output_facts.get("action_target", ""))
            f.write(f'"{esc_id}"\t"{self._escape_souffle_string(target)}"\n')

        # action_type.facts
        with open(facts_dir / "action_type.facts", "w") as f:
            atype = self._escape_souffle_string(output_facts.get("action_type") or "unknown")
            f.write(f'"{esc_id}"\t"{atype}"\n')

        # current_action.facts
        with open(facts_dir / "current_action.facts", "w") as f:
            norm_action = self._normalize_for_matching(current_action)
            f.write(f'"{esc_id}"\t"{self._escape_souffle_string(norm_action)}"\n')

        # acknowledges_error.facts - check if agent mentions errors
        with open(facts_dir / "acknowledges_error.facts", "w") as f:
            all_output_text = " ".join(
                output_facts.get("stated_observations", []) +
                output_facts.get("reasoning_steps", [])
            )
            if any(kw in all_output_text.lower() for kw in ["error", "failed", "failure", "issue", "problem"]):
                f.write(f'"{esc_id}"\n')

        # acknowledges_history.facts - check if agent mentions history/previous
        with open(facts_dir / "acknowledges_history.facts", "w") as f:
            all_output_text = " ".join(
                output_facts.get("stated_observations", []) +
                output_facts.get("reasoning_steps", [])
            )
            if any(kw in all_output_text.lower() for kw in ["previous", "before", "already", "tried", "again", "repeat"]):
                f.write(f'"{esc_id}"\n')

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
        except Exception as e:
            self.logger.error(f"Soufflé failed: {e}")
            return False

    def _parse_souffle_output(self, output_dir: Path, case_id: str) -> Dict[str, Any]:
        """Parse Soufflé output files."""
        result = {
            "is_consistent": True,
            "score": 2,
            "violations": [],
        }

        # Parse output_consistency.csv
        consistency_file = output_dir / "output_consistency.csv"
        if consistency_file.exists():
            with open(consistency_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        file_id = parts[0].strip('"')
                        if file_id == case_id:
                            result["is_consistent"] = parts[1].strip('"') == "true"
                            result["score"] = int(parts[2])

        # Parse output_violations.csv
        violations_file = output_dir / "output_violations.csv"
        if violations_file.exists():
            with open(violations_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        file_id = parts[0].strip('"')
                        if file_id == case_id:
                            result["violations"].append({
                                "type": parts[1].strip('"'),
                                "detail": parts[2].strip('"'),
                            })

        return result

    def _check_common_sense(
        self,
        input_facts: HeuristicInputFacts,
        output_facts: Dict[str, Any],
        consistency_result: Dict[str, Any],
    ) -> CommonSenseEvaluation:
        """Step 2: LLM common sense check."""
        # Format action history (handle tuple format)
        history_strs = []
        for h in input_facts.action_history[:5]:
            if isinstance(h, tuple):
                history_strs.append(f"{h[0]} -> {h[1]}")
            else:
                history_strs.append(str(h))

        prompt = self.COMMON_SENSE_PROMPT.format(
            task_goal=input_facts.task_goal or "Not specified",
            visible_elements=", ".join(input_facts.visible_elements[:20]) or "None",
            error_messages=", ".join(input_facts.error_messages) or "None",
            action_history=", ".join(history_strs) or "None",
            stated_observations=", ".join(output_facts.get("stated_observations", [])) or "None",
            action_type=output_facts.get("action_type") or "unknown",
            action_target=output_facts.get("action_target") or "unknown",
            references_made=", ".join(output_facts.get("references_made", [])) or "None",
            is_consistent=consistency_result.get("is_consistent", True),
            consistency_score=consistency_result.get("score", 2),
            violations="; ".join([f"{v['type']}: {v['detail']}" for v in consistency_result.get("violations", [])]) or "None",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You evaluate if agent behavior makes common sense. Respond with JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.model_temperature,
                response_format={"type": "json_object"},
            )

            result_json = json.loads(response.choices[0].message.content)

            violations = []
            for v in result_json.get("violations", []):
                try:
                    violations.append(Violation(
                        type=ViolationType(v.get("type", "other")),
                        description=v.get("description", ""),
                        severity=Severity(v.get("severity", "medium")),
                    ))
                except (ValueError, KeyError):
                    violations.append(Violation(
                        type=ViolationType.OTHER,
                        description=v.get("description", str(v)),
                        severity=Severity.MEDIUM,
                    ))

            return CommonSenseEvaluation(
                is_reasonable=result_json.get("is_reasonable", True),
                violations=violations,
                score=result_json.get("score", 2),
                reasoning=result_json.get("reasoning", ""),
                confidence=ConfidenceLevel(result_json.get("confidence", "medium")),
                emergent_type=derive_emergent_type(violations),
            )

        except Exception as e:
            self.logger.error(f"Common sense check failed: {e}")
            return CommonSenseEvaluation(
                is_reasonable=True,
                violations=[],
                score=1,
                reasoning=f"Check failed: {e}",
                confidence=ConfidenceLevel.LOW,
                emergent_type="error",
            )

    def _verify_hybrid(
        self,
        case_id: str,
        input_text: str,
        thinking: str,
        action: str,
    ) -> Dict[str, Any]:
        """
        Two-step hybrid verification:
        1. Extract facts (Heuristic for input - FREE, LLM for output)
        2. Check consistency (Soufflé - FREE)
        3. Check common sense (LLM)
        """
        # Step 0a: Extract input facts using heuristics (FREE)
        self.logger.debug("Extracting input facts (heuristic - FREE)...")
        input_facts = self.heuristic_extractor.extract(input_text)
        self.logger.debug(f"  Found {len(input_facts.visible_elements)} visible elements")
        self.logger.debug(f"  Found {len(input_facts.error_messages)} error messages")
        self.logger.debug(f"  Found {len(input_facts.action_history)} history entries")

        # Step 0b: Extract output facts using LLM (for semantic understanding)
        self.logger.debug("Extracting output facts (LLM)...")
        output_facts = self.llm_fact_extractor.extract_output_facts(thinking, action)
        # Convert to dict for consistency
        output_facts_dict = {
            "stated_observations": output_facts.stated_observations,
            "reasoning_steps": output_facts.reasoning_steps,
            "stated_intent": output_facts.stated_intent,
            "action_target": output_facts.action_target,
            "action_type": output_facts.action_type,
            "references_made": output_facts.references_made,
        }

        # Step 1: Soufflé consistency check (FREE)
        self.logger.debug("Running Soufflé consistency check...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            facts_dir = tmpdir / "facts"
            output_dir = tmpdir / "output"
            facts_dir.mkdir()
            output_dir.mkdir()

            self._generate_facts_hybrid(
                case_id=case_id,
                input_facts=input_facts,
                output_facts=output_facts_dict,
                current_action=action,
                facts_dir=facts_dir,
            )

            souffle_success = self._run_souffle(facts_dir, output_dir)

            if souffle_success:
                consistency = self._parse_souffle_output(output_dir, case_id)
            else:
                # Fallback if Soufflé fails
                consistency = {"is_consistent": True, "score": 1, "violations": []}

        # Step 2: LLM common sense check
        self.logger.debug("Running LLM common sense check...")
        common_sense = self._check_common_sense(
            input_facts,
            output_facts_dict,
            consistency,
        )

        # Combine results
        all_violations = []
        for v in consistency.get("violations", []):
            all_violations.append({
                "type": v["type"],
                "description": v.get("detail", ""),
                "severity": "high" if v["type"] in ["repeated_failed_action", "ignored_error"] else "medium",
                "source": "souffle_consistency",
            })
        for v in common_sense.violations:
            all_violations.append({
                "type": v.type.value,
                "description": v.description,
                "severity": v.severity.value,
                "source": "llm_common_sense",
            })

        final_score = min(consistency.get("score", 2), common_sense.score)
        is_hallucination = final_score == 0

        return {
            "thinking_eval": final_score,
            "action_eval": final_score,
            "is_hallucination": is_hallucination,
            "verifier_type": "souffle_generic_hybrid",

            "consistency_check": {
                "source": "souffle",
                "is_consistent": consistency.get("is_consistent", True),
                "score": consistency.get("score", 2),
                "violations": consistency.get("violations", []),
            },

            "common_sense_check": {
                "source": "llm",
                "is_reasonable": common_sense.is_reasonable,
                "score": common_sense.score,
                "violations": [
                    {"type": v.type.value, "description": v.description, "severity": v.severity.value}
                    for v in common_sense.violations
                ],
                "reasoning": common_sense.reasoning,
                "confidence": common_sense.confidence.value,
            },

            "all_violations": all_violations,
            "emergent_type": common_sense.emergent_type,
            "thinking_eval_reason": f"Soufflé: {len(consistency.get('violations', []))} violations. LLM: {common_sense.reasoning}",

            "extracted_facts": {
                "input": {
                    "source": "heuristic (FREE)",
                    "task_goal": input_facts.task_goal,
                    "visible_elements_count": len(input_facts.visible_elements),
                    "visible_elements": input_facts.visible_elements[:30],  # Show more now
                    "error_messages": input_facts.error_messages,
                    "action_history": [
                        f"{h[0]} -> {h[1]}" if isinstance(h, tuple) else str(h)
                        for h in input_facts.action_history[:5]
                    ],
                },
                "output": {
                    "source": "llm",
                    "stated_observations": output_facts_dict.get("stated_observations", []),
                    "action_target": output_facts_dict.get("action_target"),
                    "action_type": output_facts_dict.get("action_type"),
                    "references_made": output_facts_dict.get("references_made", []),
                },
            },
        }

    def _get_input_text(self, result_data: Dict[str, Any]) -> str:
        """Extract raw input text from result data."""
        input_data = result_data.get("input", [])
        text_parts = []
        for msg in input_data:
            if isinstance(msg, dict):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                elif isinstance(content, str):
                    text_parts.append(content)
        if "task_goal" in result_data:
            text_parts.insert(0, f"Task Goal: {result_data['task_goal']}")
        if "repetitive_action" in result_data:
            text_parts.append(f"Repetitive Action Pattern: {result_data['repetitive_action']}")
        return "\n\n".join(text_parts)[:10000]

    def _evaluate_thinking(self, thinking: str, action: str, **kwargs) -> Tuple[int, str]:
        """Required by base class."""
        return 0, "Use _verify_hybrid instead"

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> int:
        """Required by base class."""
        return 0

    def _process_single_result(self, result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single inference result."""
        task_name = result_data.get("task_name", "unknown")

        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")

        result_info = result_data.get("result", {})
        thinking = result_info.get("thinking", "")
        action = result_info.get("action", "")

        if not thinking and not action:
            self.logger.warning(f"No thinking or action for {task_name}")
            return result_data

        try:
            input_text = self._get_input_text(result_data)
            verified_result = self._verify_hybrid(
                case_id=task_name,
                input_text=input_text,
                thinking=thinking,
                action=action,
            )
            result_data[self.result_field_name] = verified_result
            self.save_results(result_data)
            return result_data

        except Exception as e:
            self.logger.error(f"Error processing {task_name}: {e}")
            return result_data

    def _process_inference_results(self) -> None:
        """Process all inference results."""
        self.logger.info(
            f"Processing {len(self.inference_results)} results with hybrid verifier"
        )

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_result = {
                executor.submit(self._process_single_result, result_data): result_data
                for result_data in self.inference_results
            }
            for future in tqdm(as_completed(future_to_result), total=len(self.inference_results)):
                result = future.result()
                if result:
                    self.verified_results.append(result)
