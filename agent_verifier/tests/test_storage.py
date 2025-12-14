"""Unit tests for agent_verifier storage."""

import os
import tempfile
import unittest

from agent_verifier.schemas import (
    Rule,
    RuleCondition,
    PolicySpec,
    RuleType,
    ConditionOperator,
    Severity,
)
from agent_verifier.storage import SQLiteStore


class TestSQLiteStore(unittest.TestCase):
    """Tests for SQLiteStore."""

    def setUp(self):
        """Set up test database."""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        os.close(self.db_fd)
        os.unlink(self.db_path)

    # --- Preference Tests ---

    def test_set_and_get_preference(self):
        """Test setting and getting a preference."""
        self.store.set_preference("user1", "app1", "theme", "dark")
        value = self.store.get_preference("user1", "app1", "theme")
        self.assertEqual(value, "dark")

    def test_get_nonexistent_preference(self):
        """Test getting a preference that doesn't exist."""
        value = self.store.get_preference("user1", "app1", "missing")
        self.assertIsNone(value)

    def test_update_preference(self):
        """Test updating an existing preference."""
        self.store.set_preference("user1", "app1", "format", "json")
        self.store.set_preference("user1", "app1", "format", "yaml")
        value = self.store.get_preference("user1", "app1", "format")
        self.assertEqual(value, "yaml")

    def test_get_all_preferences(self):
        """Test getting all preferences for a user."""
        self.store.set_preference("user1", "app1", "key1", "val1")
        self.store.set_preference("user1", "app1", "key2", "val2")
        self.store.set_preference("user1", "app2", "key3", "val3")  # Different app

        prefs = self.store.get_all_preferences("user1", "app1")
        self.assertEqual(len(prefs), 2)
        self.assertEqual(prefs["key1"], "val1")
        self.assertEqual(prefs["key2"], "val2")

    def test_delete_preference(self):
        """Test deleting a preference."""
        self.store.set_preference("user1", "app1", "temp", "value")
        self.assertTrue(self.store.delete_preference("user1", "app1", "temp"))
        value = self.store.get_preference("user1", "app1", "temp")
        self.assertIsNone(value)

    def test_delete_nonexistent_preference(self):
        """Test deleting a preference that doesn't exist."""
        result = self.store.delete_preference("user1", "app1", "missing")
        self.assertFalse(result)

    def test_preference_with_complex_value(self):
        """Test storing complex values."""
        complex_value = {"nested": {"list": [1, 2, 3], "bool": True}}
        self.store.set_preference("user1", "app1", "complex", complex_value)
        value = self.store.get_preference("user1", "app1", "complex")
        self.assertEqual(value, complex_value)

    # --- Policy Tests ---

    def test_add_and_get_policy(self):
        """Test adding and getting a policy."""
        policy = PolicySpec(
            policy_id="pol-001",
            deployment_id="my-app",
            name="Test Policy",
            description="A test policy",
            policy_type=RuleType.CONSTRAINT,
        )
        self.store.add_policy(policy)

        retrieved = self.store.get_policy("pol-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Policy")

    def test_get_policies_for_deployment(self):
        """Test getting all policies for a deployment."""
        for i in range(3):
            policy = PolicySpec(
                policy_id=f"pol-{i}",
                deployment_id="my-app",
                name=f"Policy {i}",
                description="",
                policy_type=RuleType.CONSTRAINT,
                priority=i,
            )
            self.store.add_policy(policy)

        policies = self.store.get_policies_for_deployment("my-app")
        self.assertEqual(len(policies), 3)
        # Should be ordered by priority (descending)
        self.assertEqual(policies[0].policy_id, "pol-2")

    def test_update_policy(self):
        """Test updating a policy."""
        policy = PolicySpec(
            policy_id="pol-update",
            deployment_id="my-app",
            name="Original Name",
            description="",
            policy_type=RuleType.CONSTRAINT,
        )
        self.store.add_policy(policy)

        result = self.store.update_policy("pol-update", {"name": "Updated Name"})
        self.assertTrue(result)

        retrieved = self.store.get_policy("pol-update")
        self.assertEqual(retrieved.name, "Updated Name")

    def test_delete_policy(self):
        """Test deleting a policy."""
        policy = PolicySpec(
            policy_id="pol-delete",
            deployment_id="my-app",
            name="To Delete",
            description="",
            policy_type=RuleType.CONSTRAINT,
        )
        self.store.add_policy(policy)

        result = self.store.delete_policy("pol-delete")
        self.assertTrue(result)

        retrieved = self.store.get_policy("pol-delete")
        self.assertIsNone(retrieved)

    # --- Rule Tests ---

    def test_add_and_get_rule(self):
        """Test adding and getting a rule."""
        rule = Rule(
            rule_id="rule-001",
            name="Test Rule",
            description="A test rule",
            rule_type=RuleType.CONSTRAINT,
            layer=1,
            conditions=[
                RuleCondition("field", ConditionOperator.EQUALS, "value")
            ],
            severity=Severity.ERROR,
            tags=["test"],
        )
        self.store.add_rule(rule, "my-app")

        retrieved = self.store.get_rule("rule-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Rule")

    def test_get_rules_for_layer(self):
        """Test getting rules for a specific layer."""
        for i, layer in enumerate([1, 1, 2]):
            rule = Rule(
                rule_id=f"rule-layer-{layer}-idx{i}",
                name=f"Layer {layer} Rule {i}",
                description="",
                rule_type=RuleType.CONSTRAINT,
                layer=layer,
                conditions=[],
            )
            self.store.add_rule(rule, "my-app")

        layer1_rules = self.store.get_rules_for_layer(1, "my-app")
        self.assertEqual(len(layer1_rules), 2)

        layer2_rules = self.store.get_rules_for_layer(2, "my-app")
        self.assertEqual(len(layer2_rules), 1)

    def test_get_rules_by_tag(self):
        """Test getting rules by tag."""
        rule1 = Rule(
            rule_id="rule-tag-1",
            name="Rule 1",
            description="",
            rule_type=RuleType.CONSTRAINT,
            layer=1,
            conditions=[],
            tags=["security", "common"],
        )
        rule2 = Rule(
            rule_id="rule-tag-2",
            name="Rule 2",
            description="",
            rule_type=RuleType.CONSTRAINT,
            layer=1,
            conditions=[],
            tags=["format"],
        )
        self.store.add_rule(rule1, "my-app")
        self.store.add_rule(rule2, "my-app")

        security_rules = self.store.get_rules_by_tag("security", "my-app")
        self.assertEqual(len(security_rules), 1)
        self.assertEqual(security_rules[0].rule_id, "rule-tag-1")

    def test_delete_rule(self):
        """Test deleting a rule."""
        rule = Rule(
            rule_id="rule-delete",
            name="To Delete",
            description="",
            rule_type=RuleType.CONSTRAINT,
            layer=1,
            conditions=[],
        )
        self.store.add_rule(rule, "my-app")

        result = self.store.delete_rule("rule-delete")
        self.assertTrue(result)

        retrieved = self.store.get_rule("rule-delete")
        self.assertIsNone(retrieved)


if __name__ == "__main__":
    unittest.main()
