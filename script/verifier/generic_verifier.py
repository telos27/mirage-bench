#!/usr/bin/env python3
"""
Generic Hallucination Verifier using Common Sense.

This verifier detects hallucinations without predefined hallucination types.
It uses LLM's embedded common sense to check if agent behavior aligns with
what a reasonable agent should do given the observations.

Architecture:
    ┌─────────────────────┐         ┌─────────────────────┐
    │     LLM Input       │         │    Common Sense     │
    │  (what agent sees)  │         │  (LLM world model)  │
    └─────────────────────┘         └─────────────────────┘
             │                               │
             │      ┌───────────────┐        │
             └─────→│   Compare     │←───────┘
                    │ Agent behavior│
                    │ vs Expected   │
                    └───────────────┘
                            │
                            ▼
                     Hallucination?

Usage:
    python verifier.py --type <any_type> --scenario <scenario> \
        --model <model> --use-generic-verifier
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_verifier import BaseVerifier
from .generic_schema import (
    GenericExtractionResult,
    CommonSenseEvaluation,
    Violation,
    ViolationType,
    Severity,
    ConfidenceLevel,
    violations_to_score,
    derive_emergent_type,
)
from .generic_fact_extractor import (
    GenericFactExtractor,
    ExtractedInputFacts,
    ExtractedOutputFacts,
    ExtractionResult,
    ConsistencyResult,
    ConsistencyViolation,
    check_consistency,
)


class GenericHallucinationVerifier(BaseVerifier):
    """
    Generic hallucination verifier using common sense checking.

    Instead of checking against predefined hallucination types, this verifier
    asks: "Does the agent's response make sense given what it observed?"

    Two sources of truth:
    1. LLM Input - Ground truth of what agent can observe
    2. Common Sense - What a reasonable agent should do (LLM knowledge)
    """

    SYSTEM_PROMPT = """You are an expert evaluator assessing whether an AI agent's behavior is reasonable given what it observes.

Your task is to detect "hallucinations" - cases where the agent's behavior doesn't align with reality or common sense.

A hallucination occurs when the agent:
1. References or acts on something NOT present in its observations
2. Ignores important information that IS present (errors, mismatches, failures)
3. Takes an action that contradicts its own reasoning
4. Repeats an action that has already failed
5. Confuses the current state with a different state
6. Makes up information not supported by observations

You will receive:
- The agent's input (what it can observe: screen state, history, task goal)
- The agent's output (its thinking/reasoning and chosen action)

Evaluate whether the agent's response is reasonable given common sense.

You must respond with a JSON object matching this schema:
{
    "is_reasonable": boolean,
    "violations": [
        {
            "type": "ungrounded_reference|ignored_evidence|reasoning_mismatch|repeated_failure|state_confusion|goal_deviation|fabrication|other",
            "description": "string describing the violation",
            "evidence_from_input": "quote from input showing the ground truth",
            "evidence_from_response": "quote from response showing the problem",
            "severity": "low|medium|high",
            "suggested_category": "optional: repetitive|misleading|unachievable|erroneous|etc"
        }
    ],
    "score": 0-2 (0=clear hallucination, 1=partial/unclear, 2=reasonable),
    "reasoning": "explanation of your assessment",
    "confidence": "low|medium|high",
    "emergent_type": "derived hallucination type based on violations"
}"""

    USER_PROMPT_TEMPLATE = """## Agent's Input (What it observes)

### Task Goal
{task_goal}

### Current State / Screen
{current_state}

### Action History
{action_history}

### Additional Context
{additional_context}

---

## Agent's Response

### Thinking / Reasoning
{thinking}

### Action Taken
{action}

---

## Your Evaluation

Based on common sense, evaluate whether this response is reasonable:
1. Does the agent's action make sense given what it observes?
2. Is the agent ignoring any important information?
3. Is the agent referencing anything not present in its observations?
4. Does the action align with the reasoning?

Respond with your evaluation as JSON."""

    # Step 2: Common Sense Check Prompt
    COMMON_SENSE_PROMPT = """Given these extracted facts, evaluate if the agent's behavior makes common sense.

## Input Facts (What the agent observes)
- Task Goal: {task_goal}
- Visible Elements: {visible_elements}
- Error Messages: {error_messages}
- State Info: {state_info}
- Action History: {action_history}
- Important Facts: {important_facts}

## Output Facts (What the agent does)
- Stated Observations: {stated_observations}
- Reasoning: {reasoning_steps}
- Intent: {stated_intent}
- Action: {action_type} on {action_target}

## Consistency Check Result
{consistency_result}

## Common Sense Questions
1. Given the task goal, does this action make progress?
2. Is the agent repeating an action that already failed?
3. Is the agent ignoring obvious errors or blockers?
4. Does the action contradict the stated reasoning?
5. Is this what a reasonable agent would do?

Respond with JSON:
{{
    "is_reasonable": boolean,
    "violations": [
        {{
            "type": "repeated_failure|ignored_evidence|reasoning_mismatch|goal_deviation|state_confusion|other",
            "description": "string",
            "evidence_from_input": "quote",
            "evidence_from_response": "quote",
            "severity": "low|medium|high",
            "suggested_category": "repetitive|misleading|unachievable|erroneous|etc"
        }}
    ],
    "score": 0-2,
    "reasoning": "explanation",
    "confidence": "low|medium|high",
    "emergent_type": "derived hallucination type"
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

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        # Initialize fact extractor for two-step verification
        self.fact_extractor = GenericFactExtractor(
            model=self.model_name,
            temperature=self.model_temperature,
            logger=self.logger,
        )

        self.logger.info(
            f"GenericHallucinationVerifier initialized with model={self.model_name}"
        )

    def _extract_input_context(self, result_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract structured context from the agent's input.

        This extracts the "ground truth" - what the agent actually observes.
        """
        context = {
            "task_goal": "",
            "current_state": "",
            "action_history": "",
            "additional_context": "",
        }

        try:
            input_data = result_data.get("input", [])

            # Extract from input messages
            for msg in input_data:
                if isinstance(msg, dict):
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")

                                # Try to extract task goal
                                if "task" in text.lower() or "goal" in text.lower():
                                    if not context["task_goal"]:
                                        context["task_goal"] = text[:2000]

                                # Extract history if present
                                if "history" in text.lower():
                                    context["action_history"] = text[:3000]

                                # Current state / observation
                                if "observation" in text.lower() or "screen" in text.lower():
                                    context["current_state"] = text[:3000]

            # Also check for specific fields
            if "task_goal" in result_data:
                context["task_goal"] = str(result_data["task_goal"])[:2000]

            if "repetitive_action" in result_data:
                context["additional_context"] += f"\nRepetitive action pattern: {result_data['repetitive_action']}"

            # If we couldn't extract structured context, use raw input
            if not context["current_state"] and input_data:
                context["current_state"] = str(input_data)[:4000]

        except Exception as e:
            self.logger.warning(f"Error extracting input context: {e}")
            context["current_state"] = str(result_data.get("input", ""))[:4000]

        return context

    def _evaluate_with_common_sense(
        self,
        context: Dict[str, str],
        thinking: str,
        action: str,
    ) -> CommonSenseEvaluation:
        """
        Use LLM to evaluate agent's response against common sense.
        """
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            task_goal=context.get("task_goal", "Not specified"),
            current_state=context.get("current_state", "Not available"),
            action_history=context.get("action_history", "No history"),
            additional_context=context.get("additional_context", "None"),
            thinking=thinking,
            action=action,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.model_temperature,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)

            # Parse violations
            violations = []
            for v in result_json.get("violations", []):
                try:
                    violations.append(Violation(
                        type=ViolationType(v.get("type", "other")),
                        description=v.get("description", ""),
                        evidence_from_input=v.get("evidence_from_input", ""),
                        evidence_from_response=v.get("evidence_from_response", ""),
                        severity=Severity(v.get("severity", "medium")),
                        suggested_category=v.get("suggested_category", ""),
                    ))
                except (ValueError, KeyError) as e:
                    self.logger.debug(f"Error parsing violation: {e}")
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
                emergent_type=result_json.get("emergent_type", derive_emergent_type(violations)),
            )

        except Exception as e:
            self.logger.error(f"Error in common sense evaluation: {e}")
            # Return a default evaluation indicating uncertainty
            return CommonSenseEvaluation(
                is_reasonable=True,
                violations=[],
                score=1,
                reasoning=f"Evaluation error: {str(e)}",
                confidence=ConfidenceLevel.LOW,
                emergent_type="error",
            )

    def _check_common_sense(
        self,
        input_facts: ExtractedInputFacts,
        output_facts: ExtractedOutputFacts,
        consistency_result: ConsistencyResult,
    ) -> CommonSenseEvaluation:
        """
        Step 2: Check if the agent's behavior makes common sense.

        Uses extracted facts (not raw text) for focused evaluation.
        """
        prompt = self.COMMON_SENSE_PROMPT.format(
            task_goal=input_facts.task_goal or "Not specified",
            visible_elements=", ".join(input_facts.visible_elements) or "None",
            error_messages=", ".join(input_facts.error_messages) or "None",
            state_info=", ".join(input_facts.state_info) or "None",
            action_history=", ".join(input_facts.action_history) or "None",
            important_facts=", ".join(input_facts.important_facts) or "None",
            stated_observations=", ".join(output_facts.stated_observations) or "None",
            reasoning_steps=", ".join(output_facts.reasoning_steps) or "None",
            stated_intent=output_facts.stated_intent or "None",
            action_type=output_facts.action_type or "unknown",
            action_target=output_facts.action_target or "unknown",
            consistency_result=f"Consistent: {consistency_result.is_consistent}, "
                              f"Violations: {len(consistency_result.violations)}, "
                              f"Reasoning: {consistency_result.reasoning}",
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

            # Parse violations
            violations = []
            for v in result_json.get("violations", []):
                try:
                    violations.append(Violation(
                        type=ViolationType(v.get("type", "other")),
                        description=v.get("description", ""),
                        evidence_from_input=v.get("evidence_from_input", ""),
                        evidence_from_response=v.get("evidence_from_response", ""),
                        severity=Severity(v.get("severity", "medium")),
                        suggested_category=v.get("suggested_category", ""),
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
                emergent_type=result_json.get("emergent_type", derive_emergent_type(violations)),
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

    def _verify_two_step(
        self,
        input_text: str,
        thinking: str,
        action: str,
    ) -> Dict[str, Any]:
        """
        Two-step verification:
        1. Extract facts from input and output
        2. Check consistency (input vs output)
        3. Check common sense (is behavior rational?)

        Returns combined verification result.
        """
        # Step 0: Extract facts
        self.logger.debug("Extracting facts from input and output...")
        extraction = self.fact_extractor.extract(input_text, thinking, action)

        # Step 1: Consistency check
        self.logger.debug("Checking input-output consistency...")
        consistency = check_consistency(
            extraction.input_facts,
            extraction.output_facts,
            self.client,
            self.model_name,
            self.logger,
        )

        # Step 2: Common sense check
        self.logger.debug("Checking common sense...")
        common_sense = self._check_common_sense(
            extraction.input_facts,
            extraction.output_facts,
            consistency,
        )

        # Combine results
        all_violations = []

        # Add consistency violations
        for cv in consistency.violations:
            all_violations.append({
                "type": cv.violation_type,
                "description": cv.description,
                "severity": cv.severity,
                "source": "consistency_check",
                "input_evidence": cv.input_evidence,
                "output_evidence": cv.output_evidence,
            })

        # Add common sense violations
        for v in common_sense.violations:
            all_violations.append({
                "type": v.type.value,
                "description": v.description,
                "severity": v.severity.value,
                "source": "common_sense_check",
                "suggested_category": v.suggested_category,
            })

        # Compute final score (min of both checks)
        final_score = min(consistency.score, common_sense.score)
        is_hallucination = final_score == 0

        return {
            "thinking_eval": final_score,
            "action_eval": final_score,
            "is_hallucination": is_hallucination,
            "verifier_type": "generic_two_step",

            # Step 1 results
            "consistency_check": {
                "is_consistent": consistency.is_consistent,
                "score": consistency.score,
                "violations": [
                    {"type": v.violation_type, "description": v.description, "severity": v.severity}
                    for v in consistency.violations
                ],
                "reasoning": consistency.reasoning,
            },

            # Step 2 results
            "common_sense_check": {
                "is_reasonable": common_sense.is_reasonable,
                "score": common_sense.score,
                "violations": [
                    {"type": v.type.value, "description": v.description, "severity": v.severity.value}
                    for v in common_sense.violations
                ],
                "reasoning": common_sense.reasoning,
                "confidence": common_sense.confidence.value,
            },

            # Combined
            "all_violations": all_violations,
            "emergent_type": common_sense.emergent_type,
            "thinking_eval_reason": f"Consistency: {consistency.reasoning}. Common sense: {common_sense.reasoning}",

            # Extracted facts (for debugging/analysis)
            "extracted_facts": {
                "input": {
                    "task_goal": extraction.input_facts.task_goal,
                    "visible_elements": extraction.input_facts.visible_elements,
                    "error_messages": extraction.input_facts.error_messages,
                    "action_history": extraction.input_facts.action_history,
                },
                "output": {
                    "stated_observations": extraction.output_facts.stated_observations,
                    "reasoning_steps": extraction.output_facts.reasoning_steps,
                    "action_target": extraction.output_facts.action_target,
                    "action_type": extraction.output_facts.action_type,
                },
            },
        }

    def _evaluate_thinking(
        self, thinking: str, action: str, **kwargs
    ) -> Tuple[int, str]:
        """Evaluate thinking - required by base class."""
        context = kwargs.get("context", {})
        evaluation = self._evaluate_with_common_sense(context, thinking, action)
        return evaluation.score, evaluation.reasoning

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> int:
        """Evaluate action - required by base class."""
        # For generic verifier, action score is same as thinking score
        context = kwargs.get("context", {})
        evaluation = self._evaluate_with_common_sense(context, thinking, action)
        return evaluation.score

    def _process_single_result(
        self, result_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single inference result using two-step verification."""
        task_name = result_data.get("task_name", "unknown")

        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")

        # Extract agent's response
        result_info = result_data.get("result", {})
        thinking = result_info.get("thinking", "")
        action = result_info.get("action", "")

        if not thinking and not action:
            self.logger.warning(f"No thinking or action for {task_name}")
            return result_data

        try:
            # Get raw input text
            input_text = self._get_input_text(result_data)

            # Run two-step verification
            verified_result = self._verify_two_step(
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

        # Add task goal and repetitive action if present
        if "task_goal" in result_data:
            text_parts.insert(0, f"Task Goal: {result_data['task_goal']}")
        if "repetitive_action" in result_data:
            text_parts.append(f"Repetitive Action Pattern: {result_data['repetitive_action']}")

        return "\n\n".join(text_parts)[:10000]  # Limit size

    def _process_inference_results(self) -> None:
        """Process all inference results with parallel processing."""
        self.logger.info(
            f"Processing {len(self.inference_results)} results with generic verifier (max workers: 10)"
        )

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
