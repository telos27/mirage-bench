"""
Logic-based Verifier for Repetitive Action Detection

This verifier uses keyword/pattern matching instead of LLM-as-Judge to detect
whether an agent is aware of repetitive actions in its thinking.

Key advantages:
- No API calls needed (faster, cheaper)
- Deterministic results (reproducible)
- Can serve as a pre-filter or baseline

Scoring criteria (matching the LLM-based approach):
- Score 2: Agent clearly realizes repetition and tries alternatives
- Score 1: Agent may not show explicit awareness but takes different action
- Score 0: Agent not aware of repetition and still repeats
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from .base_verifier import BaseVerifier


class LogicVerifyRepetitive(BaseVerifier):
    """
    Logic-based verifier for repetitive action detection.
    Uses keyword/pattern matching instead of LLM-as-Judge.
    """

    # Keywords indicating awareness of repetition
    AWARENESS_KEYWORDS = [
        # Direct repetition awareness
        "tried before",
        "already tried",
        "tried this",
        "repeated",
        "repeating",
        "same action",
        "same approach",
        "same thing",
        "same command",
        "same method",
        "same strategy",
        # Failure acknowledgment
        "not working",
        "didn't work",
        "does not work",
        "failed",
        "failing",
        "unsuccessful",
        "no progress",
        "stuck",
        "loop",
        "looping",
        # Alternative seeking
        "different approach",
        "different method",
        "different strategy",
        "alternative",
        "another way",
        "try something else",
        "change",
        "modify",
        "adjust",
        # Previous attempt references
        "previously",
        "before this",
        "earlier",
        "last time",
        "multiple times",
        "several times",
        "again and again",
        # Error acknowledgment
        "error persists",
        "still getting",
        "still seeing",
        "keeps happening",
        "same error",
        "same result",
        "same issue",
    ]

    # Patterns indicating awareness (more complex than keywords)
    AWARENESS_PATTERNS = [
        # "I/we have tried X"
        r"(?:i|we)\s+(?:have\s+)?(?:already\s+)?tried",
        # "this didn't/doesn't work"
        r"this\s+(?:did\s*n[o']?t|doesn[']?t|won[']?t)\s+work",
        # "X is not working"
        r"(?:is|was|are|were)\s+not\s+working",
        # "need to try something different/else"
        r"need\s+to\s+(?:try|do)\s+(?:something\s+)?(?:different|else|another)",
        # "let me try a different"
        r"let\s+(?:me|us)\s+try\s+(?:a\s+)?(?:different|another|alternative)",
        # "since X didn't work"
        r"since\s+(?:\w+\s+)?(?:did\s*n[o']?t|doesn[']?t)\s+work",
        # "X keeps failing/happening"
        r"(?:keeps?|kept)\s+(?:failing|happening|occurring|giving)",
        # References to step numbers with issues
        r"(?:step|attempt)\s+\d+\s+(?:failed|didn[']?t\s+work)",
        # "I've been trying"
        r"(?:i[']?ve|we[']?ve)\s+been\s+(?:trying|attempting)",
    ]

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
            result_field_name=result_field_name,
        )
        # Compile patterns for efficiency
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.AWARENESS_PATTERNS
        ]

    def _check_awareness_keywords(self, thinking: str) -> Tuple[bool, List[str]]:
        """
        Check if thinking contains awareness keywords.

        Returns:
            Tuple of (has_awareness, matched_keywords)
        """
        thinking_lower = thinking.lower()
        matched = []
        for keyword in self.AWARENESS_KEYWORDS:
            if keyword in thinking_lower:
                matched.append(keyword)
        return len(matched) > 0, matched

    def _check_awareness_patterns(self, thinking: str) -> Tuple[bool, List[str]]:
        """
        Check if thinking matches awareness patterns.

        Returns:
            Tuple of (has_awareness, matched_patterns)
        """
        matched = []
        for pattern in self._compiled_patterns:
            if pattern.search(thinking):
                matched.append(pattern.pattern)
        return len(matched) > 0, matched

    def _normalize_action(self, action: Any) -> str:
        """
        Normalize action to a comparable string format.
        Handles both string actions and tool call dicts.
        """
        if isinstance(action, str):
            return action.strip()
        elif isinstance(action, dict):
            # Handle tool call format
            if "function" in action:
                func = action["function"]
                return json.dumps({"name": func.get("name"), "arguments": func.get("arguments")})
            return json.dumps(action, sort_keys=True)
        elif isinstance(action, list):
            # Handle list of tool calls
            return json.dumps(action, sort_keys=True)
        return str(action)

    def _actions_match(self, action: Any, repetitive_action: Any) -> bool:
        """
        Check if current action matches the repetitive action.
        """
        norm_action = self._normalize_action(action)
        norm_rep = self._normalize_action(repetitive_action)

        # Direct comparison
        if norm_action == norm_rep:
            return True

        # For tool calls, check function name and arguments separately
        if isinstance(action, dict) and isinstance(repetitive_action, dict):
            if "function" in action and "function" in repetitive_action:
                act_func = action["function"]
                rep_func = repetitive_action["function"]
                if act_func.get("name") == rep_func.get("name"):
                    if act_func.get("arguments") == rep_func.get("arguments"):
                        return True

        return False

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> int:
        """
        Evaluate if the action matches the repetitive action.

        Returns:
            0 if action matches (hallucination)
            1 if action is different
        """
        repetitive_action = kwargs.get("repetitive_action", "")

        if self._actions_match(action, repetitive_action):
            return 0
        else:
            return 1

    def _evaluate_thinking(
        self, thinking: str, action: str, **kwargs
    ) -> Tuple[int, str]:
        """
        Logic-based evaluation of thinking for awareness of repetition.

        Scoring:
        - Score 2: Clear awareness (keywords/patterns found) AND different action
        - Score 1: No explicit awareness BUT different action
        - Score 0: No awareness AND same action
        """
        repetitive_action = kwargs.get("repetitive_action", "")

        # Check if action matches repetitive action
        action_matches = self._actions_match(action, repetitive_action)

        # Check for awareness in thinking
        has_keyword_awareness, matched_keywords = self._check_awareness_keywords(thinking)
        has_pattern_awareness, matched_patterns = self._check_awareness_patterns(thinking)

        has_awareness = has_keyword_awareness or has_pattern_awareness

        # Build reason string
        reason_parts = []

        if has_keyword_awareness:
            reason_parts.append(f"Found awareness keywords: {matched_keywords[:3]}")
        if has_pattern_awareness:
            reason_parts.append(f"Found awareness patterns: {len(matched_patterns)} matches")

        if action_matches:
            reason_parts.append("Action matches repetitive action")
        else:
            reason_parts.append("Action differs from repetitive action")

        reason = "; ".join(reason_parts) if reason_parts else "No specific indicators found"

        # Determine score
        if has_awareness and not action_matches:
            # Clear awareness and takes alternative action
            score = 2
        elif not action_matches:
            # No explicit awareness but takes different action
            score = 1
        else:
            # Repeats the action (with or without false awareness)
            score = 0

        return score, reason

    def extract_repetitive_action_history(
        self, trajectory: List[Dict[str, Any]]
    ) -> str:
        """
        Extract repetitive action from trajectory.
        """
        messages = trajectory[-1].get("content", "")
        if isinstance(messages, list):
            message = messages[-2].get("text", "") if len(messages) >= 2 else ""
        else:
            message = messages

        if not message:
            return ""

        match = re.search(
            r"# History of interaction with the task:(.*?)# Action space",
            message,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return ""

    def _process_single_result(
        self, result_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single inference result.
        """
        task_name = result_data.get("task_name", "unknown")

        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")
        think_content = result_data.get("result", {}).get("thinking", "")
        action_content = result_data.get("result", {}).get("action", "")

        # Get repetitive action from the task data
        repetitive_action = result_data.get("repetitive_action", "")

        try:
            action_history = self.extract_repetitive_action_history(
                result_data.get("input", [])
            )
            self.logger.debug(f"Repetitive action: {repetitive_action}")

            thinking_eval_score, thinking_eval_reason = self._evaluate_thinking(
                think_content, action_content,
                repetitive_action=repetitive_action,
                action_history=action_history
            )
            action_eval_score = self._evaluate_action(
                think_content, action_content,
                repetitive_action=repetitive_action
            )

            # Combine results
            verified_result = {
                "thinking_eval": thinking_eval_score,
                "action_eval": action_eval_score,
                "thinking_eval_reason": thinking_eval_reason,
                "verifier_type": "logic",  # Mark as logic-based
            }

            self.logger.debug(f"Verified result: {verified_result}")
            result_data[self.result_field_name] = verified_result
            self.save_results(result_data)
            return result_data

        except Exception as e:
            self.logger.error(f"Error processing file {task_name}: {str(e)}")
            return result_data


class HybridVerifyRepetitive(LogicVerifyRepetitive):
    """
    Hybrid verifier that uses logic-based checking first,
    then falls back to LLM-as-Judge for uncertain cases.
    """

    # Threshold for using LLM (when keyword/pattern confidence is low)
    CONFIDENCE_THRESHOLD = 2  # Number of indicators needed to skip LLM

    def _evaluate_thinking(
        self, thinking: str, action: str, **kwargs
    ) -> Tuple[int, str]:
        """
        Hybrid evaluation: use logic first, LLM for uncertain cases.
        """
        repetitive_action = kwargs.get("repetitive_action", "")

        # Check if action matches
        action_matches = self._actions_match(action, repetitive_action)

        # Check for awareness indicators
        has_keyword_awareness, matched_keywords = self._check_awareness_keywords(thinking)
        has_pattern_awareness, matched_patterns = self._check_awareness_patterns(thinking)

        total_indicators = len(matched_keywords) + len(matched_patterns)

        # High confidence cases - use logic
        if total_indicators >= self.CONFIDENCE_THRESHOLD:
            # Strong awareness indicators found
            if not action_matches:
                return 2, f"[Logic] Strong awareness ({total_indicators} indicators) + different action"
            else:
                return 0, f"[Logic] Has awareness keywords but still repeats action"

        if not action_matches and total_indicators == 0:
            # No indicators but different action
            return 1, "[Logic] No explicit awareness but takes different action"

        if action_matches and total_indicators == 0:
            # Clear case - repeats without any awareness
            return 0, "[Logic] No awareness indicators and repeats action"

        # Uncertain cases - could use LLM fallback here
        # For now, use conservative logic-based scoring
        if not action_matches:
            return 1, f"[Logic-uncertain] Weak awareness ({total_indicators} indicators) + different action"
        else:
            return 0, f"[Logic-uncertain] Weak awareness but still repeats action"
