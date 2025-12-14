"""Unit tests for agent_verifier schemas."""

import unittest
from datetime import datetime

from agent_verifier.schemas import (
    VerificationRequest,
    VerificationResult,
    Violation,
    ReasoningStep,
    Severity,
    ExtractedFacts,
    InputFacts,
    OutputFacts,
    Rule,
    RuleCondition,
    PolicySpec,
    RuleType,
    ConditionOperator,
    Session,
    Turn,
    EstablishedFact,
)


class TestVerificationRequest(unittest.TestCase):
    """Tests for VerificationRequest."""

    def test_create_minimal(self):
        """Test creating a request with minimal fields."""
        req = VerificationRequest(
            request_id="test-001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi there!",
            llm_model="gpt-4",
        )
        self.assertEqual(req.request_id, "test-001")
        self.assertEqual(req.deployment_id, "my-app")
        self.assertIsNone(req.session_id)
        self.assertIsNone(req.user_id)
        self.assertIsInstance(req.timestamp, datetime)

    def test_create_full(self):
        """Test creating a request with all fields."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        req = VerificationRequest(
            request_id="test-002",
            deployment_id="my-app",
            prompt="What is 2+2?",
            llm_output="4",
            llm_model="claude-3",
            session_id="sess-001",
            user_id="user-001",
            timestamp=ts,
            additional_context={"key": "value"},
        )
        self.assertEqual(req.session_id, "sess-001")
        self.assertEqual(req.user_id, "user-001")
        self.assertEqual(req.timestamp, ts)
        self.assertEqual(req.additional_context, {"key": "value"})

    def test_to_dict(self):
        """Test serialization to dict."""
        req = VerificationRequest(
            request_id="test-003",
            deployment_id="app",
            prompt="test",
            llm_output="output",
            llm_model="model",
        )
        d = req.to_dict()
        self.assertEqual(d["request_id"], "test-003")
        self.assertIsInstance(d["timestamp"], str)

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "request_id": "test-004",
            "deployment_id": "app",
            "prompt": "test",
            "llm_output": "output",
            "llm_model": "model",
            "timestamp": "2024-01-01T12:00:00",
        }
        req = VerificationRequest.from_dict(d)
        self.assertEqual(req.request_id, "test-004")
        self.assertEqual(req.timestamp.year, 2024)


class TestViolation(unittest.TestCase):
    """Tests for Violation."""

    def test_create(self):
        """Test creating a violation."""
        v = Violation(
            layer=1,
            violation_type="ungrounded_reference",
            severity=Severity.ERROR,
            message="Reference not found",
            evidence={"reference": "foo"},
        )
        self.assertEqual(v.layer, 1)
        self.assertEqual(v.violation_type, "ungrounded_reference")
        self.assertEqual(v.severity, Severity.ERROR)

    def test_to_dict(self):
        """Test serialization."""
        v = Violation(
            layer=2,
            violation_type="test",
            severity=Severity.WARNING,
            message="msg",
            evidence={},
            rule_id="rule-001",
            suggestion="Fix it",
        )
        d = v.to_dict()
        self.assertEqual(d["severity"], "warning")
        self.assertEqual(d["rule_id"], "rule-001")

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "layer": 3,
            "violation_type": "policy_breach",
            "severity": "info",
            "message": "info msg",
            "evidence": {"a": 1},
        }
        v = Violation.from_dict(d)
        self.assertEqual(v.severity, Severity.INFO)


class TestVerificationResult(unittest.TestCase):
    """Tests for VerificationResult."""

    def test_create(self):
        """Test creating a result."""
        result = VerificationResult(
            request_id="req-001",
            verdict="pass",
            violations=[],
            reasoning=[],
            latency_ms=50,
            layers_checked=[1, 2, 3],
        )
        self.assertEqual(result.verdict, "pass")
        self.assertFalse(result.has_errors)
        self.assertFalse(result.has_warnings)

    def test_with_violations(self):
        """Test result with violations."""
        v1 = Violation(1, "type1", Severity.ERROR, "msg", {})
        v2 = Violation(2, "type2", Severity.WARNING, "msg", {})
        result = VerificationResult(
            request_id="req-002",
            verdict="fail",
            violations=[v1, v2],
            reasoning=[],
            latency_ms=100,
            layers_checked=[1, 2],
        )
        self.assertTrue(result.has_errors)
        self.assertTrue(result.has_warnings)
        self.assertEqual(len(result.violations_by_layer(1)), 1)
        self.assertEqual(len(result.violations_by_type("type2")), 1)


class TestReasoningStep(unittest.TestCase):
    """Tests for ReasoningStep."""

    def test_create(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            layer=1,
            step_type="rule_application",
            description="Applied rule X",
            inputs={"fact": "value"},
            outputs={"violation": True},
            rule_applied="rule-001",
        )
        self.assertEqual(step.layer, 1)
        self.assertEqual(step.step_type, "rule_application")

    def test_to_dict(self):
        """Test serialization."""
        step = ReasoningStep(layer=2, step_type="inference", description="desc")
        d = step.to_dict()
        self.assertEqual(d["inputs"], {})
        self.assertEqual(d["outputs"], {})


class TestExtractedFacts(unittest.TestCase):
    """Tests for fact extraction schemas."""

    def test_input_facts(self):
        """Test InputFacts."""
        facts = InputFacts(
            task_goal="Find information",
            visible_elements=["button", "link"],
            error_messages=["Error 404"],
        )
        self.assertEqual(facts.task_goal, "Find information")
        self.assertEqual(len(facts.visible_elements), 2)

    def test_output_facts(self):
        """Test OutputFacts."""
        facts = OutputFacts(
            stated_observations=["I see a button"],
            action_target="button[1]",
        )
        self.assertEqual(facts.action_target, "button[1]")

    def test_extracted_facts_container(self):
        """Test ExtractedFacts container."""
        ef = ExtractedFacts(
            input_facts=InputFacts(task_goal="test"),
            output_facts=OutputFacts(action_target="target"),
        )
        d = ef.to_dict()
        self.assertEqual(d["input_facts"]["task_goal"], "test")


class TestRuleSchemas(unittest.TestCase):
    """Tests for rule and policy schemas."""

    def test_rule_condition(self):
        """Test RuleCondition."""
        cond = RuleCondition(
            field="output.format",
            operator=ConditionOperator.EQUALS,
            value="json",
        )
        d = cond.to_dict()
        self.assertEqual(d["operator"], "equals")

    def test_rule(self):
        """Test Rule."""
        rule = Rule(
            rule_id="rule-001",
            name="JSON Format Required",
            description="Output must be JSON",
            rule_type=RuleType.CONSTRAINT,
            layer=6,
            conditions=[
                RuleCondition("output.format", ConditionOperator.EQUALS, "json")
            ],
            severity=Severity.ERROR,
            tags=["format"],
        )
        self.assertEqual(rule.layer, 6)
        self.assertEqual(len(rule.conditions), 1)

    def test_policy_spec(self):
        """Test PolicySpec."""
        policy = PolicySpec(
            policy_id="pol-001",
            deployment_id="my-app",
            name="Data Policy",
            description="No PII in output",
            policy_type=RuleType.PROHIBITION,
        )
        d = policy.to_dict()
        self.assertEqual(d["policy_type"], "prohibition")


class TestSessionSchemas(unittest.TestCase):
    """Tests for session schemas."""

    def test_turn(self):
        """Test Turn."""
        turn = Turn(
            turn_id="turn-001",
            prompt="Hello",
            response="Hi!",
        )
        self.assertEqual(turn.prompt, "Hello")
        self.assertIsInstance(turn.timestamp, datetime)

    def test_established_fact(self):
        """Test EstablishedFact."""
        fact = EstablishedFact(
            fact_id="fact-001",
            fact_type="user_preference",
            key="language",
            value="English",
            source_turn="turn-001",
        )
        self.assertEqual(fact.confidence, 1.0)

    def test_session(self):
        """Test Session."""
        session = Session(
            session_id="sess-001",
            user_id="user-001",
            deployment_id="app-001",
        )
        turn = Turn("t1", "prompt", "response")
        session.add_turn(turn)
        self.assertEqual(len(session.turns), 1)

        fact = EstablishedFact("f1", "pref", "key", "val", "t1")
        session.add_fact(fact)
        self.assertEqual(len(session.established_facts), 1)
        self.assertEqual(session.get_fact("key"), fact)


if __name__ == "__main__":
    unittest.main()
