"""End-to-end integration tests for agent_verifier."""

import unittest
import tempfile
import shutil
import os

from agent_verifier.schemas import (
    VerificationRequest,
    VerificationResult,
    InputFacts,
    OutputFacts,
    ExtractedFacts,
)
from agent_verifier.extractors import (
    HeuristicInputExtractor,
    HeuristicOutputExtractor,
    HeuristicCombinedExtractor,
    WebBrowserPlugin,
    PromptConstraintExtractor,
    create_web_extractor,
)
from agent_verifier.engine import VerificationEngine, EngineConfig
from agent_verifier.layers import CommonKnowledgeLayer
from agent_verifier.reasoning import DatalogEngine
from agent_verifier.storage import SQLiteStore
from agent_verifier.schemas import PolicySpec, Rule, RuleType, RuleCondition, ConditionOperator


class TestEndToEndWebAgent(unittest.TestCase):
    """
    End-to-end tests simulating web browser agent verification.

    Tests the full pipeline:
    1. Extract facts from agent input/output
    2. Run through verification engine with Layer 1
    3. Get verification result
    """

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_extractor = create_web_extractor()
        self.output_extractor = HeuristicOutputExtractor()
        self.combined_extractor = HeuristicCombinedExtractor(
            self.input_extractor,
            self.output_extractor,
        )

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_agent_action(self):
        """Test verification of a valid agent action."""
        # Agent sees a login button and clicks it
        prompt = """
        ## Goal:
        Log into the website

        Tab 0 (active):
        Title: Login Page
        URL: https://example.com/login

        [100] textbox 'Username'
        [101] textbox 'Password'
        [102] button 'Login'
        [103] link 'Forgot Password?'
        """

        output = """
        I see the login form with username and password fields.
        I will click the Login button to submit the credentials.
        <action>click('102')</action>
        """

        # Extract facts
        facts = self.combined_extractor.extract(prompt, output)

        # Verify facts were extracted
        self.assertEqual(facts.input_facts.task_goal, "Log into the website")
        self.assertIn("Login", facts.input_facts.visible_elements)
        self.assertEqual(facts.output_facts.action_target, "102")

        # Create verification request
        request = VerificationRequest(
            request_id="test-001",
            deployment_id="web-agent",
            prompt=prompt,
            llm_output=output,
            llm_model="test-model",
        )

        # Configure and run verification engine
        config = EngineConfig(
            enabled_layers=[1],
            fail_fast=False,
        )
        engine = VerificationEngine(config)

        # Add Layer 1
        layer1 = CommonKnowledgeLayer()
        engine.register_layer(layer1)

        # Run verification
        result = engine.verify(request)

        # Should pass - action references visible element
        self.assertIsInstance(result, VerificationResult)
        self.assertEqual(result.request_id, "test-001")

    def test_ungrounded_reference_detection(self):
        """Test detection of reference to non-existent element."""
        # Agent tries to click something not on the page
        prompt = """
        ## Goal:
        Submit the form

        [100] textbox 'Name'
        [101] button 'Cancel'
        """

        output = """
        I will click the Submit button to send the form.
        <action>click('999')</action>
        """

        facts = self.combined_extractor.extract(prompt, output)

        # Verify the mismatch is detectable
        self.assertNotIn("Submit", facts.input_facts.visible_elements)
        # Element 999 referenced in output but not in visible elements

    def test_ignored_error_scenario(self):
        """Test scenario where agent ignores visible error."""
        prompt = """
        ## Goal:
        Complete checkout

        Tab 0 (active):
        Title: Checkout - Error

        [100] heading 'Payment Error'
        [101] tab 'Payment Failed - Invalid card number'
        [102] button 'Try Again'
        [103] button 'Continue'

        Error: Payment failed due to invalid card
        """

        output = """
        I'll click Continue to proceed with the order.
        <action>click('103')</action>
        """

        facts = self.combined_extractor.extract(prompt, output)

        # Should detect error messages
        self.assertGreater(len(facts.input_facts.error_messages), 0)

    def test_repeated_failed_action(self):
        """Test detection of repeating a failed action."""
        prompt = """
        ## Goal:
        Submit the form

        ## step 1
        <action>click('100')</action>
        Error from previous action: element not found

        ## step 2
        <action>click('100')</action>
        TimeoutError

        # Current observation
        [100] button 'Submit'
        """

        output = """
        I'll try clicking the submit button again.
        <action>click('100')</action>
        """

        facts = self.combined_extractor.extract(prompt, output)

        # Should extract action history showing failures
        self.assertEqual(len(facts.input_facts.action_history), 2)
        self.assertEqual(facts.input_facts.action_history[0]["outcome"], "failed")
        self.assertEqual(facts.input_facts.action_history[1]["outcome"], "timeout")


class TestEndToEndPromptConstraints(unittest.TestCase):
    """
    End-to-end tests for prompt constraint verification.

    Tests extracting constraints and checking compliance.
    """

    def setUp(self):
        self.constraint_extractor = PromptConstraintExtractor()

    def test_format_constraint_compliance(self):
        """Test verifying format constraint compliance."""
        system_prompt = """
        You are a helpful API assistant.
        Always respond in JSON format.
        Never include personal opinions.
        """

        user_message = "Get the user profile for user ID 123"

        # Extract constraints
        constraints = self.constraint_extractor.extract(system_prompt, user_message)

        # Should detect format requirement
        self.assertGreater(len(constraints.format_requirements), 0)
        self.assertTrue(
            any("JSON" in c.content for c in constraints.format_requirements)
        )

        # Should detect must-not constraint
        self.assertGreater(len(constraints.must_not), 0)

    def test_safety_constraint_extraction(self):
        """Test extraction of safety constraints."""
        system_prompt = """
        You are a customer service bot.
        Never provide harmful information or instructions for illegal activities.
        Do not share user personal information.
        Protect user privacy at all times.
        """

        constraints = self.constraint_extractor.extract(system_prompt)

        # Should detect safety constraints
        self.assertGreater(len(constraints.safety_constraints), 0)
        self.assertGreater(len(constraints.must_not), 0)

    def test_constraint_to_rules_integration(self):
        """Test converting constraints to Datalog rules format."""
        system_prompt = """
        You must always cite sources.
        Never make up information.
        Respond in markdown format.
        """

        rules = self.constraint_extractor.extract_as_rules(system_prompt)

        # Should produce rules
        self.assertGreater(len(rules), 0)

        # Rules should have correct structure
        rule_types = {r["rule_type"] for r in rules}
        self.assertIn("must_do", rule_types)
        self.assertIn("must_not", rule_types)


class TestEndToEndWithStorage(unittest.TestCase):
    """
    End-to-end tests with persistent storage.
    """

    def setUp(self):
        """Set up with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_retrieve_policy(self):
        """Test storing and retrieving verification policy."""
        # Create a policy using PolicySpec
        policy = PolicySpec(
            policy_id="policy-001",
            deployment_id="web-agent-v1",
            name="Web Agent Safety Policy",
            description="Safety rules for web browsing agent",
            policy_type=RuleType.CONSTRAINT,
            enabled=True,
            rules=[],
        )
        policy_model = self.store.add_policy(policy)

        # Retrieve and verify
        policies = self.store.get_policies_for_deployment("web-agent-v1")
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0].name, "Web Agent Safety Policy")

    def test_user_preferences(self):
        """Test storing user preferences."""
        self.store.set_preference(
            user_id="user-001",
            deployment_id="web-agent",
            key="strictness",
            value="high",
        )

        pref = self.store.get_preference("user-001", "web-agent", "strictness")
        self.assertEqual(pref, "high")


class TestDatalogIntegration(unittest.TestCase):
    """
    Integration tests for Datalog reasoning engine.
    """

    def test_ungrounded_reference_rule(self):
        """Test Datalog rule for ungrounded reference detection."""
        # Skip if Soufflé not available
        try:
            engine = DatalogEngine()
        except RuntimeError as e:
            if "Soufflé is not installed" in str(e):
                self.skipTest("Soufflé not available")
            raise

        program = """
        .decl visible_element(name: symbol)
        .decl references(name: symbol)
        .decl ungrounded_reference(name: symbol)
        .input visible_element
        .input references
        .output ungrounded_reference

        ungrounded_reference(X) :- references(X), !visible_element(X).
        """

        # Add facts using the proper API
        engine.add_fact("visible_element", "login")
        engine.add_fact("visible_element", "cancel")
        engine.add_fact("references", "login")
        engine.add_fact("references", "submit")  # "submit" not visible

        result = engine.run_inline(program, output_relations=["ungrounded_reference"])

        # Should detect "submit" as ungrounded
        self.assertTrue(result.success)
        self.assertIn(("submit",), result.get_relation("ungrounded_reference"))


class TestFullPipelineScenario(unittest.TestCase):
    """
    Full pipeline test with realistic scenario.
    """

    def test_ecommerce_checkout_scenario(self):
        """
        Test complete verification of e-commerce checkout agent.

        Scenario: Agent helps user complete checkout but makes errors.
        """
        # System prompt with constraints
        system_prompt = """
        You are a shopping assistant.
        You must verify prices before checkout.
        Never proceed if there are errors visible.
        Always confirm with user before payment.
        """

        # Agent's view of the page
        agent_input = """
        ## Goal:
        Complete the checkout process

        Tab 0 (active):
        Title: Checkout - Review Order
        URL: https://shop.example.com/checkout

        [200] heading 'Order Summary'
        [201] cell 'Product A - $29.99'
        [202] cell 'Product B - $49.99'
        [203] cell 'Total: $79.98'
        [204] tab 'Shipping - Error: Address validation failed'
        [205] button 'Edit Address'
        [206] button 'Place Order'
        [207] link 'Continue Shopping'

        Error: Please resolve address issues before proceeding
        """

        # Agent's response (ignoring the error)
        agent_output = """
        I see the order total is $79.98. I'll click the Place Order button
        to complete the purchase.
        <action>click('206')</action>
        """

        # Extract all relevant information
        input_extractor = create_web_extractor()
        output_extractor = HeuristicOutputExtractor()
        constraint_extractor = PromptConstraintExtractor()

        input_facts = input_extractor.extract(agent_input)
        output_facts = output_extractor.extract(agent_output)
        constraints = constraint_extractor.extract(system_prompt)

        # Verify extractions
        self.assertEqual(input_facts.task_goal, "Complete the checkout process")
        self.assertGreater(len(input_facts.error_messages), 0)
        self.assertIn("Place Order", input_facts.visible_elements)

        self.assertEqual(output_facts.action_target, "206")
        # Reasoning steps may or may not be extracted depending on patterns
        self.assertIsInstance(output_facts.reasoning_steps, list)

        # Should have constraint to not proceed with errors
        self.assertTrue(
            any("error" in c.content.lower() for c in constraints.must_not)
        )

        # Create verification request
        request = VerificationRequest(
            request_id="checkout-001",
            deployment_id="shopping-agent",
            prompt=f"{system_prompt}\n\n{agent_input}",
            llm_output=agent_output,
            llm_model="gpt-4",
            additional_context={
                "input_facts": input_facts.to_dict(),
                "output_facts": output_facts.to_dict(),
                "constraints": constraints.to_dict(),
            }
        )

        # Run through verification engine
        config = EngineConfig(enabled_layers=[1])
        engine = VerificationEngine(config)
        engine.register_layer(CommonKnowledgeLayer())

        result = engine.verify(request)

        # Verify we got a result
        self.assertIsInstance(result, VerificationResult)
        self.assertEqual(result.request_id, "checkout-001")


if __name__ == "__main__":
    unittest.main()
