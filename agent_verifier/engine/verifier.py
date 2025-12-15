"""Main verification engine that orchestrates all layers."""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..schemas.request import VerificationRequest
from ..schemas.result import VerificationResult, Violation, ReasoningStep, Severity
from ..schemas.session import Session
from ..layers.base_layer import BaseLayer, LayerResult


@dataclass
class EngineConfig:
    """Configuration for the verification engine."""

    # Which layers to enable (1-6)
    enabled_layers: list[int] = field(default_factory=lambda: [1])

    # Stop on first critical violation?
    fail_fast: bool = False

    # Severity threshold for fail_fast
    fail_fast_severity: Severity = Severity.ERROR

    # Include reasoning in results?
    include_reasoning: bool = True

    # Max time for verification (ms)
    timeout_ms: int = 30000


class VerificationEngine:
    """
    Main verification engine.

    Orchestrates verification across all configured layers,
    accumulating context and violations as it progresses.

    Usage:
        engine = VerificationEngine(config)
        engine.register_layer(CommonKnowledgeLayer())

        result = engine.verify(request)
    """

    def __init__(self, config: EngineConfig | None = None):
        """
        Initialize the verification engine.

        Args:
            config: Engine configuration (uses defaults if not provided)
        """
        self.config = config or EngineConfig()
        self._layers: dict[int, BaseLayer] = {}

    def register_layer(self, layer: BaseLayer) -> None:
        """
        Register a layer with the engine.

        Args:
            layer: The layer to register (replaces existing layer at same number)
        """
        self._layers[layer.layer_number] = layer

    def unregister_layer(self, layer_number: int) -> None:
        """
        Remove a layer from the engine.

        Args:
            layer_number: The layer number to remove
        """
        self._layers.pop(layer_number, None)

    def get_layer(self, layer_number: int) -> BaseLayer | None:
        """
        Get a registered layer by number.

        Args:
            layer_number: The layer number

        Returns:
            The layer if registered, None otherwise
        """
        return self._layers.get(layer_number)

    @property
    def active_layers(self) -> list[BaseLayer]:
        """Get list of active layers in order."""
        return [
            self._layers[n]
            for n in sorted(self._layers.keys())
            if n in self.config.enabled_layers and n in self._layers
        ]

    def _has_critical_violation(self, violations: list[Violation]) -> bool:
        """Check if any violation meets the fail_fast threshold."""
        if not self.config.fail_fast:
            return False

        threshold = self.config.fail_fast_severity
        for v in violations:
            if threshold == Severity.ERROR and v.severity == Severity.ERROR:
                return True
            if threshold == Severity.WARNING and v.severity in (Severity.ERROR, Severity.WARNING):
                return True
            if threshold == Severity.INFO:
                return True  # Any violation is critical

        return False

    def _determine_verdict(self, violations: list[Violation]) -> str:
        """
        Determine overall verdict from violations.

        Args:
            violations: All violations found

        Returns:
            "pass" or "fail"
        """
        # Fail if any error-level violations
        if any(v.severity == Severity.ERROR for v in violations):
            return "fail"
        return "pass"

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """
        Verify an LLM output against all configured layers.

        Args:
            request: The verification request

        Returns:
            VerificationResult with verdict, violations, and reasoning
        """
        start_time = time.time()

        # Initialize result components
        all_violations: list[Violation] = []
        all_reasoning: list[ReasoningStep] = []
        layers_checked: list[int] = []

        # Load initial context
        context = self._load_context(request)

        # Process each active layer
        for layer in self.active_layers:
            layer_start = time.time()

            # Check this layer
            try:
                layer_result = layer.check(request, context)
            except Exception as e:
                # Layer failed - add error reasoning and continue
                all_reasoning.append(ReasoningStep(
                    layer=layer.layer_number,
                    step_type="error",
                    description=f"Layer {layer.layer_number} ({layer.layer_name}) failed: {e}",
                    inputs={},
                    outputs={"error": str(e)},
                ))
                continue

            layers_checked.append(layer.layer_number)

            # Collect violations and reasoning
            all_violations.extend(layer_result.violations)
            if self.config.include_reasoning:
                all_reasoning.extend(layer_result.reasoning)

            # Update context with facts from this layer
            if layer_result.facts_extracted:
                context[f"layer_{layer.layer_number}_facts"] = layer_result.facts_extracted

            # Add layer timing to reasoning
            layer_time_ms = int((time.time() - layer_start) * 1000)
            if self.config.include_reasoning:
                all_reasoning.append(ReasoningStep(
                    layer=layer.layer_number,
                    step_type="layer_complete",
                    description=f"Layer {layer.layer_number} ({layer.layer_name}) completed",
                    inputs={},
                    outputs={
                        "violations_found": len(layer_result.violations),
                        "time_ms": layer_time_ms,
                    },
                ))

            # Check for fail_fast
            if self._has_critical_violation(layer_result.violations):
                all_reasoning.append(ReasoningStep(
                    layer=layer.layer_number,
                    step_type="fail_fast",
                    description="Critical violation found, stopping verification",
                    inputs={},
                    outputs={"stopped_at_layer": layer.layer_number},
                ))
                break

        # Calculate total time
        total_time_ms = int((time.time() - start_time) * 1000)

        # Determine verdict
        verdict = self._determine_verdict(all_violations)

        return VerificationResult(
            request_id=request.request_id,
            verdict=verdict,
            violations=all_violations,
            reasoning=all_reasoning if self.config.include_reasoning else [],
            latency_ms=total_time_ms,
            layers_checked=layers_checked,
            metadata={
                "config": {
                    "enabled_layers": self.config.enabled_layers,
                    "fail_fast": self.config.fail_fast,
                }
            },
        )

    def verify_batch(
        self,
        requests: list[VerificationRequest],
    ) -> list[VerificationResult]:
        """
        Verify multiple requests.

        Args:
            requests: List of verification requests

        Returns:
            List of verification results (same order as requests)
        """
        return [self.verify(req) for req in requests]

    def set_session(self, session: Session) -> None:
        """
        Set session for Layer 5 (Session History).

        Args:
            session: Session object to use
        """
        layer5 = self.get_layer(5)
        if layer5 and hasattr(layer5, '_sessions'):
            from .layer5_session import SessionState
            layer5._sessions[session.session_id] = SessionState(session=session)

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set system prompt for Layer 6 (Prompt Constraints).

        This will be included in context for all verifications.

        Args:
            system_prompt: The system prompt text
        """
        self._system_prompt = system_prompt

    def _load_context(self, request: VerificationRequest) -> dict[str, Any]:
        """
        Load initial context for verification.

        Args:
            request: The verification request

        Returns:
            Initial context dict
        """
        context = {
            "deployment_id": request.deployment_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }

        # Additional context from request
        if request.additional_context:
            context.update(request.additional_context)

        # Add system prompt if set
        if hasattr(self, '_system_prompt') and self._system_prompt:
            context["system_prompt"] = self._system_prompt

        return context


# ============================================
# Factory Functions
# ============================================

def create_engine(
    layers: list[int] | None = None,
    fail_fast: bool = False,
    include_reasoning: bool = True,
    **layer_configs: Any,
) -> VerificationEngine:
    """
    Create a verification engine with specified layers.

    Args:
        layers: List of layer numbers to enable (default: all 6)
        fail_fast: Stop on first critical violation
        include_reasoning: Include reasoning in results
        **layer_configs: Additional layer-specific configs:
            - domains: List of domains for Layer 2
            - strict_safety: Strict safety mode for Layer 6
            - storage: SQLiteStore for persistence

    Returns:
        Configured VerificationEngine

    Example:
        # Create engine with all layers
        engine = create_engine()

        # Create engine with specific layers
        engine = create_engine(layers=[1, 2, 3])

        # Create engine with custom domain config
        engine = create_engine(domains=["coding", "customer_service"])
    """
    from ..layers import (
        CommonKnowledgeLayer,
        DomainBestPracticesLayer,
        BusinessPoliciesLayer,
        UserPreferencesLayer,
        SessionHistoryLayer,
        PromptConstraintsLayer,
    )

    if layers is None:
        layers = [1, 2, 3, 4, 5, 6]

    config = EngineConfig(
        enabled_layers=layers,
        fail_fast=fail_fast,
        include_reasoning=include_reasoning,
    )

    engine = VerificationEngine(config)

    # Extract layer-specific configs
    storage = layer_configs.get("storage")
    domains = layer_configs.get("domains")
    strict_safety = layer_configs.get("strict_safety", True)

    # Register enabled layers
    if 1 in layers:
        engine.register_layer(CommonKnowledgeLayer())

    if 2 in layers:
        if domains:
            engine.register_layer(DomainBestPracticesLayer(domains=domains))
        else:
            engine.register_layer(DomainBestPracticesLayer())

    if 3 in layers:
        engine.register_layer(BusinessPoliciesLayer(storage=storage))

    if 4 in layers:
        engine.register_layer(UserPreferencesLayer(storage=storage))

    if 5 in layers:
        engine.register_layer(SessionHistoryLayer(storage=storage))

    if 6 in layers:
        engine.register_layer(PromptConstraintsLayer(strict_safety=strict_safety))

    return engine


def create_full_engine(
    storage: Any = None,
    domains: list[str] | None = None,
    fail_fast: bool = False,
) -> VerificationEngine:
    """
    Create a fully-configured verification engine with all 6 layers.

    Args:
        storage: Optional SQLiteStore for persistence
        domains: Optional list of domains for Layer 2
        fail_fast: Stop on first critical violation

    Returns:
        VerificationEngine with all layers registered

    Example:
        engine = create_full_engine()
        result = engine.verify(request)
    """
    return create_engine(
        layers=[1, 2, 3, 4, 5, 6],
        fail_fast=fail_fast,
        storage=storage,
        domains=domains,
    )


def create_lightweight_engine() -> VerificationEngine:
    """
    Create a lightweight engine with only essential layers.

    Includes:
    - Layer 1: Common Knowledge (consistency)
    - Layer 6: Prompt Constraints (instruction following)

    Returns:
        Lightweight VerificationEngine
    """
    return create_engine(layers=[1, 6])


def create_coding_engine(
    storage: Any = None,
    fail_fast: bool = False,
) -> VerificationEngine:
    """
    Create an engine optimized for coding/development scenarios.

    Includes:
    - Layer 1: Common Knowledge
    - Layer 2: Domain Best Practices (coding)
    - Layer 6: Prompt Constraints

    Returns:
        Coding-focused VerificationEngine
    """
    return create_engine(
        layers=[1, 2, 6],
        fail_fast=fail_fast,
        storage=storage,
        domains=["coding"],
    )


def quick_verify(
    prompt: str,
    output: str,
    system_prompt: str | None = None,
    layers: list[int] | None = None,
) -> VerificationResult:
    """
    Quick verification of a single prompt/output pair.

    Args:
        prompt: User prompt
        output: LLM output
        system_prompt: Optional system prompt
        layers: Layers to check (default: [1, 6])

    Returns:
        VerificationResult

    Example:
        result = quick_verify(
            prompt="Write hello world in Python",
            output="print('Hello, World!')",
        )
        print(result.verdict)  # "pass" or "fail"
    """
    if layers is None:
        layers = [1, 6]

    engine = create_engine(layers=layers)

    request = VerificationRequest(
        request_id="quick_verify",
        deployment_id="default",
        prompt=prompt,
        llm_output=output,
        llm_model="unknown",
    )

    context = {}
    if system_prompt:
        engine._system_prompt = system_prompt

    return engine.verify(request)
