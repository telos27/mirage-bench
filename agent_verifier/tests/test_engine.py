"""Tests for VerificationEngine."""

import pytest
from datetime import datetime

from agent_verifier import (
    VerificationEngine,
    EngineConfig,
    VerificationRequest,
    CommonKnowledgeLayer,
    Severity,
)
from agent_verifier.layers.base_layer import BaseLayer, LayerResult
from agent_verifier.schemas.rules import Rule


class MockLayer(BaseLayer):
    """Mock layer for testing."""

    def __init__(self, layer_number: int, violations_to_return=None):
        super().__init__(layer_number, f"Mock Layer {layer_number}")
        self.violations_to_return = violations_to_return or []
        self.check_called = False
        self.last_request = None
        self.last_context = None

    def check(self, request, context):
        self.check_called = True
        self.last_request = request
        self.last_context = context

        result = LayerResult(layer=self.layer_number)
        for v in self.violations_to_return:
            result.add_violation(self.create_violation(**v))
        result.add_reasoning(self.create_reasoning_step(
            step_type="mock_check",
            description="Mock check performed",
        ))
        return result

    def load_rules(self, deployment_id):
        return []


def make_request(request_id="test-1", prompt="Test prompt", output="Test output"):
    """Helper to create test requests."""
    return VerificationRequest(
        request_id=request_id,
        deployment_id="test-deployment",
        prompt=prompt,
        llm_output=output,
        llm_model="test-model",
    )


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_default_config(self):
        config = EngineConfig()
        assert config.enabled_layers == [1]
        assert config.fail_fast is False
        assert config.include_reasoning is True
        assert config.timeout_ms == 30000

    def test_custom_config(self):
        config = EngineConfig(
            enabled_layers=[1, 2, 3],
            fail_fast=True,
            fail_fast_severity=Severity.WARNING,
        )
        assert config.enabled_layers == [1, 2, 3]
        assert config.fail_fast is True
        assert config.fail_fast_severity == Severity.WARNING


class TestVerificationEngine:
    """Tests for VerificationEngine."""

    def test_create_engine(self):
        engine = VerificationEngine()
        assert engine.config is not None
        assert len(engine.active_layers) == 0

    def test_register_layer(self):
        engine = VerificationEngine()
        layer = MockLayer(1)
        engine.register_layer(layer)
        assert engine.get_layer(1) is layer
        assert len(engine.active_layers) == 1

    def test_unregister_layer(self):
        engine = VerificationEngine()
        layer = MockLayer(1)
        engine.register_layer(layer)
        engine.unregister_layer(1)
        assert engine.get_layer(1) is None
        assert len(engine.active_layers) == 0

    def test_verify_no_layers(self):
        engine = VerificationEngine()
        request = make_request()
        result = engine.verify(request)

        assert result.request_id == "test-1"
        assert result.verdict == "pass"
        assert len(result.violations) == 0
        assert len(result.layers_checked) == 0

    def test_verify_with_mock_layer_pass(self):
        engine = VerificationEngine()
        layer = MockLayer(1)
        engine.register_layer(layer)

        request = make_request()
        result = engine.verify(request)

        assert layer.check_called
        assert result.verdict == "pass"
        assert len(result.violations) == 0
        assert 1 in result.layers_checked

    def test_verify_with_mock_layer_fail(self):
        engine = VerificationEngine()
        layer = MockLayer(1, violations_to_return=[
            {
                "violation_type": "test_violation",
                "message": "Test violation message",
                "evidence": {"test": True},
                "severity": "error",
            }
        ])
        engine.register_layer(layer)

        request = make_request()
        result = engine.verify(request)

        assert result.verdict == "fail"
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "test_violation"
        assert result.violations[0].severity == Severity.ERROR

    def test_verify_warning_passes(self):
        """Warnings should not cause failure."""
        engine = VerificationEngine()
        layer = MockLayer(1, violations_to_return=[
            {
                "violation_type": "warning_violation",
                "message": "Warning message",
                "evidence": {},
                "severity": "warning",
            }
        ])
        engine.register_layer(layer)

        request = make_request()
        result = engine.verify(request)

        assert result.verdict == "pass"
        assert len(result.violations) == 1
        assert result.has_warnings

    def test_verify_multiple_layers(self):
        config = EngineConfig(enabled_layers=[1, 2, 3])
        engine = VerificationEngine(config)

        layer1 = MockLayer(1)
        layer2 = MockLayer(2, violations_to_return=[
            {"violation_type": "v2", "message": "m", "evidence": {}, "severity": "warning"}
        ])
        layer3 = MockLayer(3)

        engine.register_layer(layer1)
        engine.register_layer(layer2)
        engine.register_layer(layer3)

        request = make_request()
        result = engine.verify(request)

        assert layer1.check_called
        assert layer2.check_called
        assert layer3.check_called
        assert result.layers_checked == [1, 2, 3]
        assert len(result.violations) == 1

    def test_fail_fast(self):
        config = EngineConfig(enabled_layers=[1, 2, 3], fail_fast=True)
        engine = VerificationEngine(config)

        layer1 = MockLayer(1, violations_to_return=[
            {"violation_type": "critical", "message": "stop", "evidence": {}, "severity": "error"}
        ])
        layer2 = MockLayer(2)
        layer3 = MockLayer(3)

        engine.register_layer(layer1)
        engine.register_layer(layer2)
        engine.register_layer(layer3)

        request = make_request()
        result = engine.verify(request)

        assert layer1.check_called
        assert not layer2.check_called
        assert not layer3.check_called
        assert result.layers_checked == [1]

    def test_layers_run_in_order(self):
        config = EngineConfig(enabled_layers=[1, 2, 3])
        engine = VerificationEngine(config)

        # Register out of order
        engine.register_layer(MockLayer(3))
        engine.register_layer(MockLayer(1))
        engine.register_layer(MockLayer(2))

        # Should still be in order
        active = engine.active_layers
        assert [l.layer_number for l in active] == [1, 2, 3]

    def test_disabled_layer_not_run(self):
        config = EngineConfig(enabled_layers=[1, 3])  # Layer 2 disabled
        engine = VerificationEngine(config)

        layer1 = MockLayer(1)
        layer2 = MockLayer(2)
        layer3 = MockLayer(3)

        engine.register_layer(layer1)
        engine.register_layer(layer2)
        engine.register_layer(layer3)

        request = make_request()
        result = engine.verify(request)

        assert layer1.check_called
        assert not layer2.check_called
        assert layer3.check_called
        assert result.layers_checked == [1, 3]

    def test_context_passed_between_layers(self):
        config = EngineConfig(enabled_layers=[1, 2])
        engine = VerificationEngine(config)

        layer1 = MockLayer(1)
        layer2 = MockLayer(2)

        engine.register_layer(layer1)
        engine.register_layer(layer2)

        request = make_request()
        request.additional_context = {"custom_key": "custom_value"}

        result = engine.verify(request)

        # Layer 2 should receive context including deployment_id and custom key
        assert layer2.last_context["deployment_id"] == "test-deployment"
        assert layer2.last_context["custom_key"] == "custom_value"

    def test_reasoning_included_by_default(self):
        engine = VerificationEngine()
        engine.register_layer(MockLayer(1))

        request = make_request()
        result = engine.verify(request)

        assert len(result.reasoning) > 0

    def test_reasoning_excluded_when_disabled(self):
        config = EngineConfig(include_reasoning=False)
        engine = VerificationEngine(config)
        engine.register_layer(MockLayer(1))

        request = make_request()
        result = engine.verify(request)

        assert len(result.reasoning) == 0

    def test_latency_measured(self):
        engine = VerificationEngine()
        engine.register_layer(MockLayer(1))

        request = make_request()
        result = engine.verify(request)

        assert result.latency_ms >= 0

    def test_verify_batch(self):
        engine = VerificationEngine()
        engine.register_layer(MockLayer(1))

        requests = [make_request(f"req-{i}") for i in range(3)]
        results = engine.verify_batch(requests)

        assert len(results) == 3
        assert [r.request_id for r in results] == ["req-0", "req-1", "req-2"]
