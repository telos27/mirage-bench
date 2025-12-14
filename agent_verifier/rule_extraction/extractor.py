"""
LLM-based rule extraction for generating common sense rules.

Uses LLM to extract domain-specific rules that can be compiled to Datalog
for transparent, auditable verification.
"""

import json
import logging
from typing import Any, Optional

from .schemas import (
    NaturalRule,
    RuleSeverity,
    RuleDomain,
    ExtractionResult,
)


# Prompt templates for rule extraction
DOMAIN_RULE_EXTRACTION_PROMPT = """You are an expert in {domain}. Your task is to generate common sense rules that an AI agent working in this domain should follow.

Domain: {domain}
Description: {description}

Generate {num_rules} important rules that cover:
1. Safety and security concerns
2. Best practices
3. Common pitfalls to avoid
4. Quality standards

For each rule, provide a JSON object with these fields:
- "name": A short, snake_case identifier (e.g., "no_eval_user_input")
- "description": Clear one-sentence description of the rule
- "conditions": When this rule applies (natural language)
- "violation_conditions": What constitutes a violation (natural language)
- "severity": One of "error", "warning", or "info"
- "examples": List of 1-2 example violations
- "rationale": Why this rule matters
- "tags": List of relevant tags

Output ONLY a JSON array of rule objects. No other text.

Example output format:
[
  {{
    "name": "no_hardcoded_secrets",
    "description": "Never include hardcoded API keys, passwords, or secrets in code",
    "conditions": "When generating or modifying code that handles authentication or external services",
    "violation_conditions": "Code contains string literals that appear to be API keys, passwords, or secrets",
    "severity": "error",
    "examples": ["api_key = 'sk-1234567890abcdef'", "password = 'admin123'"],
    "rationale": "Hardcoded secrets can be exposed in version control and lead to security breaches",
    "tags": ["security", "secrets", "authentication"]
  }}
]
"""

REFINEMENT_PROMPT = """Review these extracted rules and improve them:

Rules:
{rules_json}

For each rule:
1. Make conditions more specific and actionable
2. Ensure violation conditions are clearly testable
3. Add missing edge cases
4. Improve clarity and precision

Output the refined rules as a JSON array with the same structure.
"""

CONFLICT_CHECK_PROMPT = """Analyze these rules for conflicts or redundancies:

Rules:
{rules_json}

Identify:
1. Rules that contradict each other
2. Rules that are redundant (one subsumes another)
3. Rules that could conflict in edge cases

Output a JSON object with:
{{
  "conflicts": [
    {{"rule1": "rule_id_1", "rule2": "rule_id_2", "description": "How they conflict"}}
  ],
  "redundancies": [
    {{"rule1": "rule_id_1", "rule2": "rule_id_2", "description": "Why redundant"}}
  ],
  "edge_cases": [
    {{"rules": ["rule_id_1", "rule_id_2"], "scenario": "Edge case description"}}
  ]
}}
"""


class LLMClient:
    """
    Simple LLM client interface.

    This can be replaced with any LLM provider (OpenAI, Anthropic, etc.)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier
            api_key: API key (uses OPENAI_API_KEY env var if not provided)
            base_url: Custom API base URL
        """
        self.model = model
        self._client = None
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, dict[str, int]]:
        """
        Get a completion from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, usage_dict)
        """
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        return response.choices[0].message.content or "", usage


class RuleExtractor:
    """
    Extracts common sense rules from LLMs for a given domain.

    Example:
        extractor = RuleExtractor()
        result = extractor.extract_domain_rules(
            domain="coding",
            description="Python programming with security focus"
        )
        for rule in result.rules:
            print(f"{rule.name}: {rule.description}")
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the rule extractor.

        Args:
            llm_client: LLM client for rule extraction
            logger: Logger instance
        """
        self.llm = llm_client or LLMClient()
        self.logger = logger or logging.getLogger(__name__)

    def extract_domain_rules(
        self,
        domain: str,
        description: str,
        num_rules: int = 10,
        refine: bool = True,
    ) -> ExtractionResult:
        """
        Extract common sense rules for a domain from the LLM.

        Args:
            domain: Domain name (e.g., "coding", "customer_service")
            description: Detailed description of the domain and context
            num_rules: Target number of rules to generate
            refine: Whether to do a refinement pass

        Returns:
            ExtractionResult with extracted rules
        """
        self.logger.info(f"Extracting {num_rules} rules for domain: {domain}")

        # Initial extraction
        prompt = DOMAIN_RULE_EXTRACTION_PROMPT.format(
            domain=domain,
            description=description,
            num_rules=num_rules,
        )

        response, usage = self.llm.complete(
            prompt=prompt,
            system_prompt="You are an expert rule generator. Output only valid JSON.",
            temperature=0.7,
        )

        total_usage = usage.copy()

        # Parse rules
        rules = self._parse_rules_json(response, domain)
        self.logger.info(f"Initial extraction: {len(rules)} rules")

        # Refinement pass
        if refine and rules:
            refined_rules, refine_usage = self._refine_rules(rules)
            if refined_rules:
                rules = refined_rules
                total_usage["prompt_tokens"] += refine_usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += refine_usage.get("completion_tokens", 0)
            self.logger.info(f"After refinement: {len(rules)} rules")

        # Calculate cost estimate (rough, based on GPT-4o-mini pricing)
        cost = self._estimate_cost(
            total_usage["prompt_tokens"],
            total_usage["completion_tokens"],
            self.llm.model,
        )

        return ExtractionResult(
            domain=domain,
            description=description,
            rules=rules,
            model=self.llm.model,
            prompt_tokens=total_usage["prompt_tokens"],
            completion_tokens=total_usage["completion_tokens"],
            total_cost=cost,
        )

    def extract_rules_from_examples(
        self,
        domain: str,
        examples: list[dict[str, str]],
    ) -> ExtractionResult:
        """
        Extract rules by analyzing good/bad examples.

        Args:
            domain: Domain name
            examples: List of {"input": ..., "good_output": ..., "bad_output": ..., "reason": ...}

        Returns:
            ExtractionResult with inferred rules
        """
        examples_text = "\n\n".join([
            f"Example {i+1}:\n"
            f"Input: {ex.get('input', 'N/A')}\n"
            f"Good output: {ex.get('good_output', 'N/A')}\n"
            f"Bad output: {ex.get('bad_output', 'N/A')}\n"
            f"Reason: {ex.get('reason', 'N/A')}"
            for i, ex in enumerate(examples)
        ])

        prompt = f"""Analyze these examples of good and bad AI agent behavior and extract rules.

Domain: {domain}

{examples_text}

Based on these examples, generate rules that distinguish good from bad behavior.
Output a JSON array of rule objects with: name, description, conditions, violation_conditions, severity, examples, rationale, tags.
"""

        response, usage = self.llm.complete(
            prompt=prompt,
            system_prompt="You are an expert rule generator. Output only valid JSON.",
            temperature=0.5,
        )

        rules = self._parse_rules_json(response, domain)

        cost = self._estimate_cost(
            usage["prompt_tokens"],
            usage["completion_tokens"],
            self.llm.model,
        )

        return ExtractionResult(
            domain=domain,
            description=f"Rules extracted from {len(examples)} examples",
            rules=rules,
            model=self.llm.model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_cost=cost,
        )

    def check_conflicts(
        self,
        rules: list[NaturalRule],
    ) -> dict[str, Any]:
        """
        Check rules for conflicts and redundancies.

        Args:
            rules: List of rules to check

        Returns:
            Dictionary with conflicts, redundancies, and edge_cases
        """
        rules_json = json.dumps([r.to_dict() for r in rules], indent=2)

        prompt = CONFLICT_CHECK_PROMPT.format(rules_json=rules_json)

        response, _ = self.llm.complete(
            prompt=prompt,
            system_prompt="You are a careful rule analyst. Output only valid JSON.",
            temperature=0.3,
        )

        try:
            return json.loads(self._extract_json(response))
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse conflict check response")
            return {"conflicts": [], "redundancies": [], "edge_cases": []}

    def _refine_rules(
        self,
        rules: list[NaturalRule],
    ) -> tuple[list[NaturalRule], dict[str, int]]:
        """Refine rules with a second LLM pass."""
        rules_json = json.dumps([r.to_dict() for r in rules], indent=2)

        prompt = REFINEMENT_PROMPT.format(rules_json=rules_json)

        response, usage = self.llm.complete(
            prompt=prompt,
            system_prompt="You are an expert rule refinement specialist. Output only valid JSON.",
            temperature=0.5,
        )

        domain = rules[0].domain if rules else "general"
        refined = self._parse_rules_json(response, domain)

        return refined, usage

    def _parse_rules_json(
        self,
        response: str,
        domain: str,
    ) -> list[NaturalRule]:
        """Parse JSON response into NaturalRule objects."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            if not isinstance(data, list):
                data = [data]

            rules = []
            for item in data:
                try:
                    # Handle severity conversion
                    severity_str = item.get("severity", "error").lower()
                    try:
                        severity = RuleSeverity(severity_str)
                    except ValueError:
                        severity = RuleSeverity.ERROR

                    rule = NaturalRule(
                        name=item.get("name", "unnamed_rule"),
                        description=item.get("description", ""),
                        domain=item.get("domain", domain),
                        conditions=item.get("conditions", ""),
                        violation_conditions=item.get("violation_conditions", ""),
                        severity=severity,
                        examples=item.get("examples", []),
                        rationale=item.get("rationale", ""),
                        tags=item.get("tags", []),
                        confidence=item.get("confidence", 0.9),
                        source="llm_extraction",
                    )
                    rules.append(rule)
                except Exception as e:
                    self.logger.warning(f"Failed to parse rule: {e}")
                    continue

            return rules

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return []

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain other content."""
        # Try to find JSON array or object
        text = text.strip()

        # Look for JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]

        # Look for JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]

        return text

    def _estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """Estimate cost based on model and token counts."""
        # Approximate pricing (per 1M tokens)
        pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "claude-3-opus": {"input": 15.00, "output": 75.00},
            "claude-3-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
        }

        # Find matching model or default
        model_pricing = None
        for model_name, prices in pricing.items():
            if model_name in model.lower():
                model_pricing = prices
                break

        if model_pricing is None:
            model_pricing = {"input": 1.00, "output": 3.00}  # Default estimate

        cost = (
            prompt_tokens * model_pricing["input"] / 1_000_000 +
            completion_tokens * model_pricing["output"] / 1_000_000
        )

        return round(cost, 6)


# Predefined domain descriptions for common use cases
DOMAIN_DESCRIPTIONS = {
    RuleDomain.CODING: """
        Software development and code generation. Agents write, review, and modify code.
        Focus areas: security (injection, XSS, CSRF), code quality, error handling,
        performance, maintainability, testing, documentation.
    """,
    RuleDomain.CUSTOMER_SERVICE: """
        Customer support and service interactions. Agents handle inquiries, complaints,
        and requests. Focus areas: empathy, accuracy, escalation, privacy, legal compliance,
        response appropriateness, resolution tracking.
    """,
    RuleDomain.DATA_ANALYSIS: """
        Data analysis and reporting. Agents analyze datasets, generate insights, and
        create visualizations. Focus areas: statistical validity, data privacy, bias
        detection, reproducibility, clear communication of uncertainty.
    """,
    RuleDomain.CONTENT_GENERATION: """
        Content creation including writing, summarization, and editing. Agents generate
        text for various purposes. Focus areas: accuracy, attribution, tone consistency,
        audience appropriateness, originality, factual grounding.
    """,
    RuleDomain.GENERAL: """
        General-purpose AI assistant capabilities. Focus areas: truthfulness, safety,
        helpfulness, avoiding harmful content, following instructions, acknowledging
        limitations, privacy protection.
    """,
    RuleDomain.CUSTOM: """
        Custom domain. Provide your own description when extracting rules.
    """,
}


def create_extractor(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> RuleExtractor:
    """
    Create a rule extractor with default configuration.

    Args:
        model: LLM model to use
        api_key: Optional API key

    Returns:
        Configured RuleExtractor
    """
    llm = LLMClient(model=model, api_key=api_key)
    return RuleExtractor(llm_client=llm)


def extract_rules_for_domain(
    domain: RuleDomain,
    num_rules: int = 10,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> ExtractionResult:
    """
    Convenience function to extract rules for a predefined domain.

    Args:
        domain: Domain to extract rules for
        num_rules: Number of rules to generate
        model: LLM model to use
        api_key: Optional API key

    Returns:
        ExtractionResult with extracted rules
    """
    extractor = create_extractor(model=model, api_key=api_key)
    description = DOMAIN_DESCRIPTIONS.get(domain, DOMAIN_DESCRIPTIONS[RuleDomain.GENERAL])
    return extractor.extract_domain_rules(
        domain=domain.value,
        description=description.strip(),
        num_rules=num_rules,
    )
