#!/usr/bin/env python3
"""
LLM-based fact extractor for neuro-symbolic verification.

This module uses an LLM to extract structured facts from agent outputs.
It serves as the "perception" layer in the neuro-symbolic architecture:
    LLM (perception) -> Structured Facts -> Datalog (reasoning) -> Scores

The extraction prompt is designed to be reliable and consistent.
"""

import os
import json
import logging
from typing import Optional
from openai import OpenAI

from .fact_schema import (
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
)


# =============================================================================
# Extraction Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert at analyzing LLM agent behavior. Your task is to extract structured facts from an agent's thinking and action output.

You must respond with a JSON object that exactly matches the schema provided. Be precise and objective in your analysis.

Key principles:
1. SEMANTIC comparison - look at what actions mean, not just string matching
2. EVIDENCE-based - cite specific quotes or observations
3. CONSERVATIVE - when uncertain, use lower confidence/awareness levels
4. OBJECTIVE - avoid speculation, focus on what's explicitly stated or clearly implied"""


def get_extraction_prompt(
    thinking: str,
    action: str,
    reference_action: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """Generate the extraction prompt for the LLM."""

    reference_section = ""
    if reference_action:
        reference_section = f"""
## Reference Action (for comparison)
The following action was performed previously and may be relevant for comparison:
```
{reference_action}
```
"""

    context_section = ""
    if context:
        context_section = f"""
## Additional Context
{context}
"""

    return f"""# Task: Extract Structured Facts from Agent Output

Analyze the following agent output and extract structured facts.

## Agent's Thinking/Reasoning
```
{thinking if thinking else "(No thinking provided)"}
```

## Agent's Action
```
{action if action else "(No action provided)"}
```
{reference_section}{context_section}
## Instructions

Extract the following facts:

### 1. Current Action Semantics
- action_type: High-level category (click, type, scroll, navigate, wait, etc.)
- target: What element/entity is being acted upon
- parameters: Key parameters or values
- intent: One of [continue, retry, adapt, abandon, explore, unknown]
- normalized_form: A normalized string representation for comparison

### 2. Reference Action Semantics (if reference action provided)
Same fields as current action.

### 3. Action Comparison (if reference action provided)
- relation: Semantic relationship [identical, equivalent, similar, different, opposite, subset_of, superset_of, unknown]
  - "identical": Exact same action with same parameters
  - "equivalent": Same semantic meaning, different syntax (e.g., click(#btn) vs click(button[id='btn']))
  - "similar": Related but not the same (e.g., click different buttons in same area)
  - "different": Clearly different actions
- confidence: [low, medium, high]
- explanation: Brief explanation

### 4. Pattern Awareness
- awareness_level: [none, implicit, explicit]
  - "none": No indication of awareness
  - "implicit": Indirect signs (e.g., "let me try something else" without stating why)
  - "explicit": Direct acknowledgment (e.g., "I've been repeating this action")
- recognized_pattern: What pattern was recognized (if any)
- response_to_pattern: How did the agent respond
- evidence: Direct quotes supporting this assessment

### 5. Reasoning Assessment
- quality: [poor, adequate, good, excellent]
- considers_alternatives: Did the agent consider other options?
- considers_history: Did the agent consider what happened before?
- identifies_issues: Did the agent identify problems?
- adapts_approach: Did the agent change their approach?
- key_insights: Notable good observations
- key_failures: Notable reasoning failures

### 6. Meta
- extraction_confidence: Your confidence in this extraction [low, medium, high]
- extraction_notes: Any important notes

Respond with ONLY a JSON object matching this schema. No markdown code blocks."""


# =============================================================================
# Fact Extractor Class
# =============================================================================

class FactExtractor:
    """
    LLM-based fact extractor for agent outputs.

    Uses structured output parsing to ensure consistent extraction.
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

    def extract(
        self,
        thinking: str,
        action: str,
        reference_action: Optional[str] = None,
        context: Optional[str] = None
    ) -> ExtractedFacts:
        """
        Extract structured facts from agent output.

        Args:
            thinking: The agent's thinking/reasoning text
            action: The agent's action
            reference_action: Optional reference action for comparison
            context: Optional additional context

        Returns:
            ExtractedFacts with all extracted information
        """
        prompt = get_extraction_prompt(
            thinking=thinking,
            action=action,
            reference_action=reference_action,
            context=context
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format=ExtractedFacts,
            )

            facts = response.choices[0].message.parsed
            return facts

        except Exception as e:
            self.logger.error(f"Fact extraction failed: {e}")
            # Return a default/fallback extraction
            return self._create_fallback_extraction(thinking, action, reference_action)

    def _create_fallback_extraction(
        self,
        thinking: str,
        action: str,
        reference_action: Optional[str] = None
    ) -> ExtractedFacts:
        """Create a fallback extraction when LLM fails."""
        self.logger.warning("Using fallback extraction")

        # Simple heuristic-based fallback
        current_action = ActionSemantics(
            action_type="unknown",
            target=None,
            parameters=None,
            intent=IntentType.UNKNOWN,
            normalized_form=action.strip() if action else ""
        )

        reference = None
        comparison = None
        if reference_action:
            reference = ActionSemantics(
                action_type="unknown",
                target=None,
                parameters=None,
                intent=IntentType.UNKNOWN,
                normalized_form=reference_action.strip()
            )

            # Simple string comparison as fallback
            norm_action = action.strip().lower() if action else ""
            norm_ref = reference_action.strip().lower() if reference_action else ""

            if norm_action == norm_ref:
                relation = SemanticRelation.IDENTICAL
            elif norm_action in norm_ref or norm_ref in norm_action:
                relation = SemanticRelation.SIMILAR
            else:
                relation = SemanticRelation.DIFFERENT

            comparison = ActionComparison(
                relation=relation,
                confidence=ConfidenceLevel.LOW,
                explanation="Fallback: simple string comparison"
            )

        return ExtractedFacts(
            current_action=current_action,
            reference_action=reference,
            action_comparison=comparison,
            pattern_awareness=PatternAwareness(
                awareness_level=AwarenessLevel.NONE,
                evidence=[]
            ),
            reasoning_assessment=ReasoningAssessment(
                quality=ReasoningQuality.ADEQUATE,
                considers_alternatives=False,
                considers_history=False,
                identifies_issues=False,
                adapts_approach=False,
                key_insights=[],
                key_failures=["Extraction failed, using fallback"]
            ),
            extraction_confidence=ConfidenceLevel.LOW,
            extraction_notes="Fallback extraction due to LLM failure"
        )

    def extract_batch(
        self,
        items: list[dict],
        thinking_key: str = "thinking",
        action_key: str = "action",
        reference_key: str = "reference_action"
    ) -> list[ExtractedFacts]:
        """
        Extract facts from a batch of items.

        Args:
            items: List of dicts containing thinking/action/reference
            thinking_key: Key for thinking text
            action_key: Key for action text
            reference_key: Key for reference action

        Returns:
            List of ExtractedFacts
        """
        results = []
        for item in items:
            facts = self.extract(
                thinking=item.get(thinking_key, ""),
                action=item.get(action_key, ""),
                reference_action=item.get(reference_key)
            )
            results.append(facts)
        return results


# =============================================================================
# Utility Functions
# =============================================================================

def extract_facts_from_result(
    result_data: dict,
    extractor: FactExtractor
) -> ExtractedFacts:
    """
    Extract facts from a standard inference result dict.

    Expected structure:
    {
        "result": {"thinking": "...", "action": "..."},
        "repetitive_action": "..."  # optional
    }
    """
    result_info = result_data.get("result", {})
    thinking = result_info.get("thinking", "")
    action = result_info.get("action", "")
    reference_action = result_data.get("repetitive_action")

    return extractor.extract(
        thinking=thinking,
        action=action,
        reference_action=reference_action
    )
