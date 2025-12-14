"""Unit tests for fact extractors."""

import unittest

from agent_verifier.extractors import (
    BaseInputExtractor,
    BaseOutputExtractor,
    HeuristicInputExtractor,
    HeuristicOutputExtractor,
    HeuristicCombinedExtractor,
    WebBrowserPlugin,
    CodeEditorPlugin,
    ChatPlugin,
    create_web_extractor,
    create_code_extractor,
    create_chat_extractor,
)
from agent_verifier.schemas import InputFacts, OutputFacts


class TestHeuristicInputExtractor(unittest.TestCase):
    """Tests for HeuristicInputExtractor."""

    def setUp(self):
        self.extractor = HeuristicInputExtractor()

    def test_extract_goal_standard(self):
        """Test goal extraction with standard format."""
        text = "## Goal:\nFind the cheapest flight to Paris"
        goal = self.extractor.extract_goal(text)
        self.assertEqual(goal, "Find the cheapest flight to Paris")

    def test_extract_goal_inline(self):
        """Test goal extraction with inline format."""
        text = "Goal: Book a hotel room for tomorrow"
        goal = self.extractor.extract_goal(text)
        self.assertEqual(goal, "Book a hotel room for tomorrow")

    def test_extract_goal_json_style(self):
        """Test goal extraction from JSON-like format."""
        text = "'goal': 'Complete the checkout process'"
        goal = self.extractor.extract_goal(text)
        self.assertEqual(goal, "Complete the checkout process")

    def test_extract_errors_standard(self):
        """Test error extraction with standard patterns."""
        text = """
        Error: Connection timeout
        Failed: Unable to load page
        Cannot find element with id 'submit'
        """
        errors = self.extractor.extract_errors(text)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("timeout" in e.lower() for e in errors))

    def test_extract_errors_deduplication(self):
        """Test that duplicate errors are removed."""
        text = """
        Error: Connection timeout
        Error: connection timeout
        Error: Connection Timeout
        """
        errors = self.extractor.extract_errors(text)
        # Should dedupe to ~1 error
        self.assertLess(len(errors), 3)

    def test_extract_action_history(self):
        """Test action history extraction."""
        text = """
        ## step 1
        <action>click('123')</action>
        Success

        ## step 2
        <action>type('456', 'hello')</action>
        Error from previous action: element not found

        ## step 3
        <action>scroll(0, 100)</action>
        TimeoutError
        """
        history = self.extractor.extract_action_history(text)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["outcome"], "success")
        self.assertEqual(history[1]["outcome"], "failed")
        self.assertEqual(history[2]["outcome"], "timeout")

    def test_extract_constraints(self):
        """Test constraint extraction."""
        text = """
        Must use the official API
        Should not exceed rate limits
        Do not include personal information
        Required: Authentication token
        """
        constraints = self.extractor.extract_constraints(text)
        self.assertGreater(len(constraints), 0)

    def test_extract_format_requirements_json(self):
        """Test format detection for JSON."""
        text = "Please respond in JSON format"
        fmt = self.extractor.extract_format_requirements(text)
        self.assertEqual(fmt, "json")

    def test_extract_format_requirements_code(self):
        """Test format detection for code."""
        text = "Write the code to solve this problem"
        fmt = self.extractor.extract_format_requirements(text)
        self.assertEqual(fmt, "code")

    def test_extract_returns_input_facts(self):
        """Test that extract returns proper InputFacts."""
        text = "Goal: Test task\nError: Something failed"
        facts = self.extractor.extract(text)
        self.assertIsInstance(facts, InputFacts)
        self.assertEqual(facts.task_goal, "Test task")

    def test_empty_input(self):
        """Test extraction from empty input."""
        facts = self.extractor.extract("")
        self.assertIsInstance(facts, InputFacts)
        self.assertIsNone(facts.task_goal)
        self.assertEqual(facts.visible_elements, [])


class TestWebBrowserPlugin(unittest.TestCase):
    """Tests for WebBrowserPlugin."""

    def setUp(self):
        self.plugin = WebBrowserPlugin()
        self.extractor = HeuristicInputExtractor().add_plugin(self.plugin)

    def test_extract_link_elements(self):
        """Test link extraction from AXTree."""
        text = """
        [123] link 'Home Page'
        [124] link 'About Us'
        [125] link 'Contact'
        """
        elements = self.plugin.extract_elements(text)
        self.assertIn("Home Page", elements)
        self.assertIn("About Us", elements)

    def test_extract_button_elements(self):
        """Test button extraction from AXTree."""
        text = "[456] button 'Submit Form'"
        elements = self.plugin.extract_elements(text)
        self.assertIn("Submit Form", elements)

    def test_extract_state_url(self):
        """Test URL extraction."""
        text = "URL: https://example.com/page"
        state = self.plugin.extract_state(text)
        self.assertEqual(state.get("url"), "https://example.com/page")

    def test_extract_state_title(self):
        """Test page title extraction."""
        text = """
        Tab 0 (active):
        Title: Welcome to Example
        URL: https://example.com
        """
        state = self.plugin.extract_state(text)
        self.assertEqual(state.get("page_title"), "Welcome to Example")

    def test_extract_tab_errors(self):
        """Test tab-related error extraction."""
        text = "[789] tab 'Form - Invalid data entered'"
        errors = self.plugin.extract_errors(text)
        self.assertGreater(len(errors), 0)

    def test_full_extraction_with_plugin(self):
        """Test full extraction with web plugin."""
        text = """
        ## Goal:
        Find product information

        Tab 0 (active):
        Title: Product Catalog
        URL: https://shop.example.com/products

        [100] link 'Electronics'
        [101] button 'Add to Cart'
        [102] heading 'Featured Products'

        Error: Product out of stock
        """
        facts = self.extractor.extract(text)
        self.assertEqual(facts.task_goal, "Find product information")
        self.assertIn("Electronics", facts.visible_elements)
        self.assertIn("Add to Cart", facts.visible_elements)
        self.assertEqual(facts.state_info.get("url"), "https://shop.example.com/products")


class TestCodeEditorPlugin(unittest.TestCase):
    """Tests for CodeEditorPlugin."""

    def setUp(self):
        self.plugin = CodeEditorPlugin()
        self.extractor = HeuristicInputExtractor().add_plugin(self.plugin)

    def test_extract_file_paths(self):
        """Test file path extraction."""
        text = "file: src/main.py\npath: tests/test_main.py"
        elements = self.plugin.extract_elements(text)
        self.assertTrue(any("main.py" in e for e in elements))

    def test_extract_symbols(self):
        """Test function/class extraction."""
        text = "def calculate_total\nclass OrderProcessor"
        elements = self.plugin.extract_elements(text)
        self.assertTrue(any("calculate_total" in e for e in elements))
        self.assertTrue(any("OrderProcessor" in e for e in elements))

    def test_extract_state(self):
        """Test editor state extraction."""
        text = "current file: app.py\ncursor at line 42"
        state = self.plugin.extract_state(text)
        self.assertEqual(state.get("current_file"), "app.py")

    def test_extract_errors(self):
        """Test code error extraction."""
        text = """
        error[E0001]: syntax error
        warning: unused variable
        type error: cannot assign str to int
        """
        errors = self.plugin.extract_errors(text)
        self.assertGreater(len(errors), 0)


class TestChatPlugin(unittest.TestCase):
    """Tests for ChatPlugin."""

    def setUp(self):
        self.plugin = ChatPlugin()

    def test_extract_roles(self):
        """Test role extraction from conversation."""
        text = """
        user: Hello
        assistant: Hi there!
        user: How are you?
        """
        elements = self.plugin.extract_elements(text)
        self.assertTrue(any("user" in e.lower() for e in elements))
        self.assertTrue(any("assistant" in e.lower() for e in elements))

    def test_extract_quotes(self):
        """Test quoted content extraction."""
        text = 'He said "this is important" and then left'
        elements = self.plugin.extract_elements(text)
        self.assertTrue(any("important" in e for e in elements))

    def test_extract_message_count(self):
        """Test message counting."""
        text = "user: msg1\nassistant: reply1\nuser: msg2"
        state = self.plugin.extract_state(text)
        self.assertEqual(state.get("message_count"), 3)


class TestHeuristicOutputExtractor(unittest.TestCase):
    """Tests for HeuristicOutputExtractor."""

    def setUp(self):
        self.extractor = HeuristicOutputExtractor()

    def test_extract_action_function_style(self):
        """Test action extraction with function style."""
        text = "I will click('submit-button')"
        action_type, target = self.extractor.extract_action(text)
        self.assertEqual(action_type, "click")
        self.assertEqual(target, "submit-button")

    def test_extract_action_tag_style(self):
        """Test action extraction with tag style."""
        text = "<action>type('search-box', 'hello')</action>"
        action_type, target = self.extractor.extract_action(text)
        self.assertEqual(action_type, "type")

    def test_extract_references(self):
        """Test reference extraction."""
        text = """
        I see the 'Submit' button and the 'Cancel' link.
        Let me click the form submit button.
        """
        refs = self.extractor.extract_references(text)
        self.assertGreater(len(refs), 0)
        self.assertTrue(any("submit" in r.lower() for r in refs))

    def test_extract_reasoning_steps(self):
        """Test reasoning extraction."""
        text = """
        First, I need to find the login form.
        Because the user wants to log in, I should look for credentials.
        Therefore I will click the submit button.
        """
        steps = self.extractor.extract_reasoning_steps(text)
        self.assertGreater(len(steps), 0)

    def test_extract_observations(self):
        """Test observation extraction."""
        text = "I see a search box at the top. There is a login button."
        obs = self.extractor.extract_stated_observations(text)
        self.assertGreater(len(obs), 0)

    def test_detect_format_json(self):
        """Test JSON format detection."""
        text = '{"key": "value"}'
        fmt = self.extractor.detect_format(text)
        self.assertEqual(fmt, "json")

    def test_detect_format_code(self):
        """Test code block detection."""
        text = "```python\nprint('hello')\n```"
        fmt = self.extractor.detect_format(text)
        self.assertEqual(fmt, "code")

    def test_detect_format_list(self):
        """Test bullet list detection."""
        text = "- Item 1\n- Item 2\n- Item 3"
        fmt = self.extractor.detect_format(text)
        self.assertEqual(fmt, "list")

    def test_extract_returns_output_facts(self):
        """Test that extract returns proper OutputFacts."""
        text = "I will click('button') because it's the submit button"
        facts = self.extractor.extract(text)
        self.assertIsInstance(facts, OutputFacts)
        self.assertEqual(facts.action_target, "button")

    def test_empty_output(self):
        """Test extraction from empty output."""
        facts = self.extractor.extract("")
        self.assertIsInstance(facts, OutputFacts)
        self.assertIsNone(facts.action_target)


class TestHeuristicCombinedExtractor(unittest.TestCase):
    """Tests for HeuristicCombinedExtractor."""

    def test_combined_extraction(self):
        """Test combined input/output extraction."""
        input_extractor = create_web_extractor()
        combined = HeuristicCombinedExtractor(input_extractor)

        prompt = """
        ## Goal:
        Click the login button

        [100] button 'Login'
        [101] link 'Sign Up'
        """
        output = "I will click('100') to log in"

        facts = combined.extract(prompt, output)
        self.assertEqual(facts.input_facts.task_goal, "Click the login button")
        self.assertIn("Login", facts.input_facts.visible_elements)
        self.assertEqual(facts.output_facts.action_target, "100")
        self.assertEqual(facts.metadata.get("extractor"), "heuristic")


class TestFactoryFunctions(unittest.TestCase):
    """Tests for factory functions."""

    def test_create_web_extractor(self):
        """Test web extractor factory."""
        extractor = create_web_extractor()
        self.assertIsInstance(extractor, HeuristicInputExtractor)
        self.assertEqual(len(extractor.plugins), 1)
        self.assertIsInstance(extractor.plugins[0], WebBrowserPlugin)

    def test_create_code_extractor(self):
        """Test code extractor factory."""
        extractor = create_code_extractor()
        self.assertIsInstance(extractor, HeuristicInputExtractor)
        self.assertEqual(len(extractor.plugins), 1)
        self.assertIsInstance(extractor.plugins[0], CodeEditorPlugin)

    def test_create_chat_extractor(self):
        """Test chat extractor factory."""
        extractor = create_chat_extractor()
        self.assertIsInstance(extractor, HeuristicInputExtractor)
        self.assertEqual(len(extractor.plugins), 1)
        self.assertIsInstance(extractor.plugins[0], ChatPlugin)


class TestPluginChaining(unittest.TestCase):
    """Tests for plugin chaining."""

    def test_multiple_plugins(self):
        """Test extractor with multiple plugins."""
        extractor = HeuristicInputExtractor()
        extractor.add_plugin(WebBrowserPlugin())
        extractor.add_plugin(CodeEditorPlugin())

        text = """
        [100] button 'Run Code'
        file: main.py
        def process_data
        """
        facts = extractor.extract(text)

        # Should find elements from both plugins
        self.assertTrue(any("Run Code" in e for e in facts.visible_elements))
        self.assertTrue(any("main.py" in e for e in facts.visible_elements))

    def test_add_plugin_returns_self(self):
        """Test that add_plugin returns self for chaining."""
        extractor = HeuristicInputExtractor()
        result = extractor.add_plugin(WebBrowserPlugin())
        self.assertIs(result, extractor)

    def test_fluent_plugin_addition(self):
        """Test fluent API for adding plugins."""
        extractor = (
            HeuristicInputExtractor()
            .add_plugin(WebBrowserPlugin())
            .add_plugin(ChatPlugin())
        )
        self.assertEqual(len(extractor.plugins), 2)


if __name__ == "__main__":
    unittest.main()
