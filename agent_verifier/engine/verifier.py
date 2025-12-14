"""Main verification engine that orchestrates all layers."""

import time
from dataclasses import dataclass, field
from typing import Any

from ..schemas.request import VerificationRequest
from ..schemas.result import VerificationResult, Violation, ReasoningStep, Severity
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

    def _load_context(self, request: VerificationRequest) -> dict[str, Any]:
        """
        Load initial context for verification.

        This can be extended to load from session store, user preferences, etc.

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

        return context

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
