#!/usr/bin/env python3
"""
Fact extractor for generic hallucination detection.

Extracts structured facts from LLM input (observations) and output (response)
to enable two-step verification:
    1. Consistency check: Does output match input?
    2. Common sense check: Is the behavior rational?

Architecture:
    Input (raw) ──→ Extract ──→ ObservedFacts
    Output (raw) ──→ Extract ──→ AgentClaims

    Then: Compare(ObservedFacts, AgentClaims) ──→ Violations
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI

from .generic_schema import (
    ObservedFacts,
    ObservedFact,
    AgentResponse,
    Severity,
)


# =============================================================================
# Extraction Schemas
# =============================================================================

class ExtractedInputFacts(BaseModel):
    """Facts extracted from the agent's input (what it observes)."""

    task_goal: str = Field(
        default="",
        description="The goal/task the agent is trying to accomplish"
    )

    visible_elements: List[str] = Field(
        default_factory=list,
        description="UI elements, buttons, links, text visible on screen"
    )

    error_messages: List[str] = Field(
        default_factory=list,
        description="Any error messages, warnings, or failure indicators"
    )

    state_info: List[str] = Field(
        default_factory=list,
        description="Current state information (logged in, page title, etc.)"
    )

    action_history: List[str] = Field(
        default_factory=list,
        description="Previous actions and their outcomes"
    )

    important_facts: List[str] = Field(
        default_factory=list,
        description="Critical facts that should not be ignored"
    )


class ExtractedOutputFacts(BaseModel):
    """Facts extracted from the agent's output (what it claims/does)."""

    stated_observations: List[str] = Field(
        default_factory=list,
        description="What the agent claims to see or observe"
    )

    reasoning_steps: List[str] = Field(
        default_factory=list,
        description="The agent's reasoning/thinking steps"
    )

    stated_intent: str = Field(
        default="",
        description="What the agent says it's trying to do"
    )

    action_target: str = Field(
        default="",
        description="What element/entity the action targets"
    )

    action_type: str = Field(
        default="",
        description="Type of action (click, type, scroll, etc.)"
    )

    references_made: List[str] = Field(
        default_factory=list,
        description="Specific elements/entities the agent references"
    )


class ExtractionResult(BaseModel):
    """Complete extraction result for consistency checking."""

    input_facts: ExtractedInputFacts
    output_facts: ExtractedOutputFacts
    extraction_confidence: str = Field(default="medium")
    extraction_notes: str = Field(default="")


# =============================================================================
# Prompts
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting structured facts from LLM agent interactions.

Your task is to extract facts from:
1. The agent's INPUT (what it can observe - screen state, history, errors, goal)
2. The agent's OUTPUT (what it claims, reasons, and does)

Be precise and objective. Extract only what is explicitly present."""


INPUT_EXTRACTION_PROMPT = """Extract facts from this agent INPUT (what the agent observes).

## Agent's Input
{input_text}

## Instructions
Extract:
1. task_goal: What is the agent trying to accomplish?
2. visible_elements: What UI elements, text, buttons, links are visible?
3. error_messages: Any errors, warnings, or failure messages?
4. state_info: Current state (page title, login status, etc.)?
5. action_history: What previous actions were taken? What were their results?
6. important_facts: Any critical facts that should not be ignored?

Respond with JSON only."""


OUTPUT_EXTRACTION_PROMPT = """Extract facts from this agent OUTPUT (thinking + action).

## Agent's Thinking
{thinking}

## Agent's Action
{action}

## Instructions
Extract:
1. stated_observations: What does the agent claim to see?
2. reasoning_steps: What are the agent's reasoning steps?
3. stated_intent: What does the agent say it's trying to do?
4. action_target: What element/entity is the action targeting?
5. action_type: What type of action is this (click, type, scroll, etc.)?
6. references_made: What specific elements/entities does the agent reference?

Respond with JSON only."""


# =============================================================================
# Fact Extractor
# =============================================================================

class GenericFactExtractor:
    """
    Extracts structured facts from agent input and output.

    This enables:
    1. Consistency checking (do claims match observations?)
    2. Common sense checking (is the behavior rational?)
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.0

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.logger = logger or logging.getLogger(__name__)

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def extract_input_facts(self, input_text: str) -> ExtractedInputFacts:
        """Extract facts from the agent's input."""
        prompt = INPUT_EXTRACTION_PROMPT.format(input_text=input_text[:8000])

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format=ExtractedInputFacts,
            )
            return response.choices[0].message.parsed

        except Exception as e:
            self.logger.warning(f"Input extraction failed: {e}")
            return ExtractedInputFacts()

    def extract_output_facts(self, thinking: str, action: str) -> ExtractedOutputFacts:
        """Extract facts from the agent's output."""
        prompt = OUTPUT_EXTRACTION_PROMPT.format(
            thinking=thinking[:4000] if thinking else "(No thinking)",
            action=action[:1000] if action else "(No action)"
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format=ExtractedOutputFacts,
            )
            return response.choices[0].message.parsed

        except Exception as e:
            self.logger.warning(f"Output extraction failed: {e}")
            return ExtractedOutputFacts()

    def extract(
        self,
        input_text: str,
        thinking: str,
        action: str
    ) -> ExtractionResult:
        """
        Extract facts from both input and output.

        Args:
            input_text: The agent's input (observations, screen state, etc.)
            thinking: The agent's thinking/reasoning
            action: The agent's action

        Returns:
            ExtractionResult with input_facts and output_facts
        """
        input_facts = self.extract_input_facts(input_text)
        output_facts = self.extract_output_facts(thinking, action)

        return ExtractionResult(
            input_facts=input_facts,
            output_facts=output_facts,
            extraction_confidence="medium",
            extraction_notes=""
        )


# =============================================================================
# Consistency Checking
# =============================================================================

class ConsistencyViolation(BaseModel):
    """A violation found during consistency checking."""
    violation_type: str
    description: str
    input_evidence: str = ""
    output_evidence: str = ""
    severity: str = "medium"


class ConsistencyResult(BaseModel):
    """Result of consistency check between input and output facts."""
    is_consistent: bool
    violations: List[ConsistencyViolation] = Field(default_factory=list)
    score: int = Field(description="0-2 score for compatibility")
    reasoning: str = ""


CONSISTENCY_CHECK_PROMPT = """Compare these extracted facts and identify any inconsistencies.

## Input Facts (What the agent ACTUALLY observes)
- Task Goal: {task_goal}
- Visible Elements: {visible_elements}
- Error Messages: {error_messages}
- State Info: {state_info}
- Action History: {action_history}
- Important Facts: {important_facts}

## Output Facts (What the agent CLAIMS and DOES)
- Stated Observations: {stated_observations}
- Reasoning Steps: {reasoning_steps}
- Stated Intent: {stated_intent}
- Action Target: {action_target}
- Action Type: {action_type}
- References Made: {references_made}

## Instructions
Check for these inconsistencies:

1. **Ungrounded References**: Does the agent reference elements NOT in visible_elements?
2. **Ignored Errors**: Does the agent ignore error_messages?
3. **Ignored History**: Does the agent repeat failed actions from action_history?
4. **State Mismatch**: Do stated_observations contradict state_info?
5. **Fabrication**: Does the agent claim to see things not in the input?

Respond with JSON:
{{
    "is_consistent": boolean,
    "violations": [
        {{
            "violation_type": "ungrounded_reference|ignored_error|ignored_history|state_mismatch|fabrication",
            "description": "string",
            "input_evidence": "quote from input facts",
            "output_evidence": "quote from output facts",
            "severity": "low|medium|high"
        }}
    ],
    "score": 0-2 (0=inconsistent, 1=partial, 2=consistent),
    "reasoning": "explanation"
}}"""


def check_consistency(
    input_facts: ExtractedInputFacts,
    output_facts: ExtractedOutputFacts,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    logger: Optional[logging.Logger] = None
) -> ConsistencyResult:
    """
    Check consistency between input and output facts.

    This is Step 1 of verification: Does the output match the input?
    """
    logger = logger or logging.getLogger(__name__)

    prompt = CONSISTENCY_CHECK_PROMPT.format(
        task_goal=input_facts.task_goal,
        visible_elements=", ".join(input_facts.visible_elements) or "None",
        error_messages=", ".join(input_facts.error_messages) or "None",
        state_info=", ".join(input_facts.state_info) or "None",
        action_history=", ".join(input_facts.action_history) or "None",
        important_facts=", ".join(input_facts.important_facts) or "None",
        stated_observations=", ".join(output_facts.stated_observations) or "None",
        reasoning_steps=", ".join(output_facts.reasoning_steps) or "None",
        stated_intent=output_facts.stated_intent or "None",
        action_target=output_facts.action_target or "None",
        action_type=output_facts.action_type or "None",
        references_made=", ".join(output_facts.references_made) or "None",
    )

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You check if agent output is consistent with input. Respond with JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format=ConsistencyResult,
        )
        return response.choices[0].message.parsed

    except Exception as e:
        logger.error(f"Consistency check failed: {e}")
        return ConsistencyResult(
            is_consistent=True,
            violations=[],
            score=1,
            reasoning=f"Check failed: {e}"
        )
