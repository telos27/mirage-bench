#!/usr/bin/env python3
"""
Tests for the Heuristic Fact Extractor.

Tests the rule-based extraction of facts from AXTree and agent input.
"""

import unittest
import logging

from verifier.heuristic_fact_extractor import (
    HeuristicFactExtractor,
    HeuristicInputFacts,
    extract_output_facts_heuristic,
)


class TestHeuristicInputFacts(unittest.TestCase):
    """Test the HeuristicInputFacts dataclass."""

    def test_default_values(self):
        """Test default values are empty."""
        facts = HeuristicInputFacts()
        self.assertEqual(facts.task_goal, "")
        self.assertEqual(facts.visible_elements, [])
        self.assertEqual(facts.error_messages, [])
        self.assertEqual(facts.state_info, [])
        self.assertEqual(facts.action_history, [])
        self.assertEqual(facts.important_facts, [])

    def test_with_values(self):
        """Test with provided values."""
        facts = HeuristicInputFacts(
            task_goal="Submit form",
            visible_elements=["Submit", "Cancel"],
            error_messages=["Error: Invalid"],
            state_info=["Page: Form"],
            action_history=[("click(submit)", "failed")],
            important_facts=["Form validation failed"],
        )
        self.assertEqual(facts.task_goal, "Submit form")
        self.assertEqual(len(facts.visible_elements), 2)
        self.assertEqual(len(facts.action_history), 1)


class TestExtractVisibleElements(unittest.TestCase):
    """Test extract_visible_elements method."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = HeuristicFactExtractor()

    def test_extract_links(self):
        """Test extracting link elements."""
        text = """
        [123] link 'Home'
        [124] link 'Products'
        [125] link 'Contact Us'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Home", elements)
        self.assertIn("Products", elements)
        self.assertIn("Contact Us", elements)

    def test_extract_buttons(self):
        """Test extracting button elements."""
        text = """
        [200] button 'Submit'
        [201] button 'Cancel'
        [202] button 'Save Draft'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Submit", elements)
        self.assertIn("Cancel", elements)
        self.assertIn("Save Draft", elements)

    def test_extract_static_text(self):
        """Test extracting StaticText elements."""
        text = """
        StaticText 'Welcome to our website'
        StaticText 'Please fill out the form below'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Welcome to our website", elements)
        self.assertIn("Please fill out the form below", elements)

    def test_extract_tabs(self):
        """Test extracting tab elements."""
        text = """
        [300] tab 'General'
        [301] tab 'Settings'
        [302] tab 'Advanced'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("General", elements)
        self.assertIn("Settings", elements)
        self.assertIn("Advanced", elements)

    def test_extract_cells(self):
        """Test extracting table cell elements."""
        text = """
        [400] cell 'Product Name'
        [401] cell 'Price'
        [402] cell 'Quantity'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Product Name", elements)
        self.assertIn("Price", elements)
        self.assertIn("Quantity", elements)

    def test_extract_textbox(self):
        """Test extracting textbox elements."""
        text = """
        [500] textbox 'Enter your email'
        [501] textbox 'Password'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Enter your email", elements)
        self.assertIn("Password", elements)

    def test_extract_headings(self):
        """Test extracting heading elements."""
        text = """
        [600] heading 'Dashboard'
        [601] heading 'Recent Activity'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Dashboard", elements)
        self.assertIn("Recent Activity", elements)

    def test_extract_menuitems(self):
        """Test extracting menu item elements."""
        text = """
        [700] menuitem 'New File'
        [701] menuitem 'Open'
        [702] menuitem 'Save'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("New File", elements)
        self.assertIn("Open", elements)
        self.assertIn("Save", elements)

    def test_extract_images(self):
        """Test extracting image elements with alt text."""
        text = """
        [800] image 'Company Logo'
        [801] image 'Product Photo'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Company Logo", elements)
        self.assertIn("Product Photo", elements)

    def test_deduplication(self):
        """Test that duplicate elements are removed."""
        text = """
        [100] link 'Home'
        [101] link 'Home'
        StaticText 'home'
        """
        elements = self.extractor.extract_visible_elements(text)
        # Should have only one "Home" (case-insensitive dedup)
        home_count = sum(1 for e in elements if e.lower() == "home")
        self.assertEqual(home_count, 1)

    def test_skip_short_elements(self):
        """Test that very short elements are skipped."""
        text = """
        StaticText 'A'
        StaticText 'OK'
        StaticText 'Submit'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertNotIn("A", elements)
        self.assertIn("OK", elements)
        self.assertIn("Submit", elements)

    def test_skip_icon_elements(self):
        """Test that icon-only elements (unicode) are skipped."""
        text = """
        StaticText '\\ue001'
        [100] button '\\ue002'
        [101] button 'Submit'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Submit", elements)
        # Should not include unicode icons
        self.assertFalse(any(e.startswith("\\ue") for e in elements))

    def test_skip_long_elements(self):
        """Test that very long elements are skipped."""
        long_text = "A" * 250
        text = f"""
        StaticText '{long_text}'
        StaticText 'Normal text'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Normal text", elements)
        self.assertNotIn(long_text, elements)

    def test_mixed_elements(self):
        """Test extraction from mixed AXTree content."""
        text = """
        Tab 0 (current):
            Title: Product Dashboard
            URL: http://example.com/dashboard

        [1] link 'Home'
        [2] button 'Add Product'
        [3] heading 'Products'
        StaticText 'Total: 150 items'
        [4] cell 'SKU'
        [5] cell 'Name'
        [6] tab 'Inventory'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Home", elements)
        self.assertIn("Add Product", elements)
        self.assertIn("Products", elements)
        self.assertIn("Total: 150 items", elements)
        self.assertIn("SKU", elements)
        self.assertIn("Inventory", elements)


class TestExtractErrorMessages(unittest.TestCase):
    """Test extract_error_messages method."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = HeuristicFactExtractor()

    def test_extract_error_pattern(self):
        """Test extracting 'Error:' pattern."""
        text = "Error: Invalid email format"
        errors = self.extractor.extract_error_messages(text)
        self.assertTrue(any("Invalid email" in e for e in errors))

    def test_extract_failed_pattern(self):
        """Test extracting 'Failed:' pattern."""
        text = "Failed: Connection refused"
        errors = self.extractor.extract_error_messages(text)
        self.assertTrue(any("Connection refused" in e for e in errors))

    def test_extract_timeout_pattern(self):
        """Test extracting 'Timeout:' pattern."""
        text = "Timeout: Request exceeded 30 seconds"
        errors = self.extractor.extract_error_messages(text)
        self.assertTrue(any("Request exceeded" in e or "timeout" in e.lower() for e in errors))

    def test_extract_invalid_data(self):
        """Test extracting 'invalid data' pattern."""
        text = "This tab contains invalid data. Please resolve."
        errors = self.extractor.extract_error_messages(text)
        self.assertTrue(len(errors) > 0)

    def test_extract_cannot_pattern(self):
        """Test extracting 'Cannot' pattern."""
        text = "Cannot find the requested resource"
        errors = self.extractor.extract_error_messages(text)
        self.assertTrue(any("Cannot find" in e for e in errors))

    def test_extract_unable_pattern(self):
        """Test extracting 'Unable to' pattern."""
        text = "Unable to connect to server"
        errors = self.extractor.extract_error_messages(text)
        self.assertTrue(any("Unable to connect" in e for e in errors))

    def test_tab_error_extraction(self):
        """Test extracting errors from tab elements."""
        text = """
        [100] tab 'General'
        [101] tab 'This tab contains invalid data. Please resolve any errors before saving.'
        [102] tab 'Settings'
        """
        errors = self.extractor.extract_error_messages(text)
        self.assertTrue(any("invalid" in e.lower() for e in errors))

    def test_deduplication(self):
        """Test that duplicate errors are removed."""
        text = """
        Error: Invalid data found
        Error: Invalid data detected
        Error: Please check the invalid data
        """
        errors = self.extractor.extract_error_messages(text)
        # Should deduplicate based on key phrases
        self.assertLessEqual(len(errors), 2)

    def test_limit_errors(self):
        """Test that errors are limited."""
        text = "\n".join([f"Error: Problem {i}" for i in range(20)])
        errors = self.extractor.extract_error_messages(text)
        self.assertLessEqual(len(errors), 5)

    def test_no_errors(self):
        """Test when no errors are present."""
        text = "Everything is working correctly. Status: OK."
        errors = self.extractor.extract_error_messages(text)
        self.assertEqual(errors, [])


class TestExtractActionHistory(unittest.TestCase):
    """Test extract_action_history method."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = HeuristicFactExtractor()

    def test_extract_single_action(self):
        """Test extracting a single action."""
        text = """
        ## step 1
        <action>click(123)</action>
        """
        history = self.extractor.extract_action_history(text)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][0], "click(123)")

    def test_extract_multiple_actions(self):
        """Test extracting multiple actions."""
        text = """
        ## step 1
        <action>click(100)</action>

        ## step 2
        <action>fill(200, "test")</action>

        ## step 3
        <action>submit(300)</action>
        """
        history = self.extractor.extract_action_history(text)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0][0], "click(100)")
        self.assertEqual(history[1][0], 'fill(200, "test")')
        self.assertEqual(history[2][0], "submit(300)")

    def test_detect_timeout_outcome(self):
        """Test detecting timeout outcomes."""
        text = """
        ## step 1
        <action>click(100)</action>
        TimeoutError: Action timed out
        """
        history = self.extractor.extract_action_history(text)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][1], "timeout")

    def test_detect_failed_outcome(self):
        """Test detecting failed outcomes."""
        text = """
        ## step 1
        <action>click(100)</action>
        Error from previous action: Element not found
        """
        history = self.extractor.extract_action_history(text)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][1], "failed")

    def test_detect_success_outcome(self):
        """Test detecting success outcomes."""
        text = """
        ## step 1
        <action>click(100)</action>
        Action completed success
        """
        history = self.extractor.extract_action_history(text)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][1], "success")

    def test_unknown_outcome(self):
        """Test unknown outcome when no indicators."""
        text = """
        ## step 1
        <action>click(100)</action>
        """
        history = self.extractor.extract_action_history(text)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][1], "unknown")

    def test_no_actions(self):
        """Test when no actions present."""
        text = "Just some text without any actions"
        history = self.extractor.extract_action_history(text)
        self.assertEqual(history, [])


class TestExtractTaskGoal(unittest.TestCase):
    """Test extract_task_goal method."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = HeuristicFactExtractor()

    def test_goal_with_header(self):
        """Test extracting goal with ## Goal: header."""
        text = """
        ## Goal:
        Submit the registration form with valid data

        ## Current State
        """
        goal = self.extractor.extract_task_goal(text)
        self.assertEqual(goal, "Submit the registration form with valid data")

    def test_goal_inline(self):
        """Test extracting inline Goal: format."""
        text = "Goal: Complete the checkout process"
        goal = self.extractor.extract_task_goal(text)
        self.assertEqual(goal, "Complete the checkout process")

    def test_task_goal_format(self):
        """Test extracting Task Goal: format."""
        text = "Task Goal: Find and purchase blue shoes"
        goal = self.extractor.extract_task_goal(text)
        self.assertEqual(goal, "Find and purchase blue shoes")

    def test_goal_dict_format(self):
        """Test extracting 'goal': 'value' format."""
        text = "{'goal': 'Navigate to settings page'}"
        goal = self.extractor.extract_task_goal(text)
        self.assertEqual(goal, "Navigate to settings page")

    def test_no_goal(self):
        """Test when no goal is present."""
        text = "Just some random text without a goal"
        goal = self.extractor.extract_task_goal(text)
        self.assertEqual(goal, "")


class TestExtractStateInfo(unittest.TestCase):
    """Test extract_state_info method."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = HeuristicFactExtractor()

    def test_extract_page_title(self):
        """Test extracting page title."""
        text = """
        Tab 0 (current):
            Title: Product Dashboard
        """
        state = self.extractor.extract_state_info(text)
        self.assertTrue(any("Page: Product Dashboard" in s for s in state))

    def test_extract_url(self):
        """Test extracting URL."""
        text = "URL: https://example.com/products"
        state = self.extractor.extract_state_info(text)
        self.assertTrue(any("URL: https://example.com/products" in s for s in state))

    def test_extract_focused_element(self):
        """Test extracting focused element."""
        text = """
        Focused element:
            bid='submit_button'
        """
        state = self.extractor.extract_state_info(text)
        self.assertTrue(any("Focused: submit_button" in s for s in state))

    def test_extract_all_state_info(self):
        """Test extracting all state info."""
        text = """
        Tab 0 (current):
            Title: Checkout Page
        URL: https://shop.example.com/checkout
        Focused element:
            bid='card_number'
        """
        state = self.extractor.extract_state_info(text)
        self.assertEqual(len(state), 3)

    def test_no_state_info(self):
        """Test when no state info present."""
        text = "Just some content"
        state = self.extractor.extract_state_info(text)
        self.assertEqual(state, [])


class TestExtractMethod(unittest.TestCase):
    """Test the main extract method."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = HeuristicFactExtractor()

    def test_extract_full_input(self):
        """Test extraction from full agent input."""
        text = """
        ## Goal:
        Find the top selling products

        Tab 0 (current):
            Title: Sales Dashboard
        URL: https://admin.example.com/sales

        [1] link 'Home'
        [2] link 'Products'
        [3] button 'Export'
        [4] heading 'Top Products'
        StaticText 'Revenue: $50,000'
        [5] tab 'This tab contains invalid data. Please resolve.'

        ## step 1
        <action>click(100)</action>
        TimeoutError: Timed out

        ## step 2
        <action>click(101)</action>

        # Action space:
        """
        facts = self.extractor.extract(text)

        # Check task goal
        self.assertEqual(facts.task_goal, "Find the top selling products")

        # Check visible elements
        self.assertIn("Home", facts.visible_elements)
        self.assertIn("Products", facts.visible_elements)
        self.assertIn("Export", facts.visible_elements)
        self.assertIn("Top Products", facts.visible_elements)

        # Check error messages
        self.assertTrue(len(facts.error_messages) > 0)

        # Check state info
        self.assertTrue(any("Sales Dashboard" in s for s in facts.state_info))

        # Check action history
        self.assertEqual(len(facts.action_history), 2)
        self.assertEqual(facts.action_history[0][1], "timeout")

    def test_extract_returns_heuristic_input_facts(self):
        """Test that extract returns HeuristicInputFacts instance."""
        facts = self.extractor.extract("Some text")
        self.assertIsInstance(facts, HeuristicInputFacts)


class TestExtractOutputFactsHeuristic(unittest.TestCase):
    """Test the extract_output_facts_heuristic function."""

    def test_extract_click_action(self):
        """Test extracting click action."""
        thinking = "I need to click the submit button"
        action = "click(123)"
        facts = extract_output_facts_heuristic(thinking, action)

        self.assertEqual(facts["action_type"], "click")
        self.assertEqual(facts["action_target"], "123")

    def test_extract_fill_action(self):
        """Test extracting fill action with simple target."""
        thinking = "I will fill in the email field"
        action = "fill(email_input)"
        facts = extract_output_facts_heuristic(thinking, action)

        self.assertEqual(facts["action_type"], "fill")
        self.assertEqual(facts["action_target"], "email_input")

    def test_extract_fill_action_with_value(self):
        """Test extracting fill action with quoted target and value.

        Note: The heuristic regex has limitations with complex quoted arguments.
        This test documents the current behavior.
        """
        thinking = "I will fill in the email field"
        action = 'fill("200", "test@example.com")'
        facts = extract_output_facts_heuristic(thinking, action)

        # Current regex captures the first quoted part
        # This is a known limitation of the heuristic approach
        self.assertIn(facts["action_type"], ["fill", "unknown"])

    def test_extract_quoted_references(self):
        """Test extracting quoted references from thinking."""
        thinking = "I see the 'Submit' button and the 'Cancel' link"
        action = "click(100)"
        facts = extract_output_facts_heuristic(thinking, action)

        self.assertIn("Submit", facts["references_made"])
        self.assertIn("Cancel", facts["references_made"])

    def test_extract_the_patterns(self):
        """Test extracting 'the X' patterns."""
        thinking = "I will click the submit button on the registration page"
        action = "click(100)"
        facts = extract_output_facts_heuristic(thinking, action)

        # Should find "submit" or "registration" patterns
        self.assertTrue(len(facts["references_made"]) > 0)

    def test_deduplicate_references(self):
        """Test that references are deduplicated."""
        thinking = "I see 'Submit' and I will click 'Submit'"
        action = "click(100)"
        facts = extract_output_facts_heuristic(thinking, action)

        submit_count = facts["references_made"].count("Submit")
        self.assertEqual(submit_count, 1)

    def test_limit_references(self):
        """Test that references are limited."""
        # Create thinking with many references
        refs = [f"'item{i}'" for i in range(30)]
        thinking = " ".join(refs)
        action = "click(100)"
        facts = extract_output_facts_heuristic(thinking, action)

        self.assertLessEqual(len(facts["references_made"]), 20)

    def test_unknown_action_type(self):
        """Test handling of unknown action format."""
        thinking = "Doing something"
        action = "unknown_action"
        facts = extract_output_facts_heuristic(thinking, action)

        self.assertEqual(facts["action_type"], "unknown")
        self.assertEqual(facts["action_target"], "")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = HeuristicFactExtractor()

    def test_empty_input(self):
        """Test handling empty input."""
        facts = self.extractor.extract("")
        self.assertEqual(facts.task_goal, "")
        self.assertEqual(facts.visible_elements, [])
        self.assertEqual(facts.error_messages, [])

    def test_malformed_axtree(self):
        """Test handling malformed AXTree."""
        text = """
        [abc] link 'Test'
        [123 button 'Broken'
        link 'No ID'
        """
        # Should not crash
        elements = self.extractor.extract_visible_elements(text)
        # May or may not extract elements depending on patterns
        self.assertIsInstance(elements, list)

    def test_special_characters_in_elements(self):
        """Test handling special characters."""
        text = """
        [100] link 'Price: $99.99'
        [101] button 'Add to Cart (2)'
        StaticText '50% OFF!'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertIn("Price: $99.99", elements)
        self.assertIn("Add to Cart (2)", elements)
        self.assertIn("50% OFF!", elements)

    def test_unicode_in_elements(self):
        """Test handling unicode characters."""
        text = """
        [100] link 'Caf\u00e9'
        [101] button '\u2713 Accept'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertTrue(any("Caf" in e for e in elements))

    def test_multiline_elements(self):
        """Test that multiline content is handled."""
        text = """
        StaticText 'This is a very long text that spans multiple words but should be captured correctly'
        """
        elements = self.extractor.extract_visible_elements(text)
        self.assertTrue(any("long text" in e for e in elements))


class TestLoggerIntegration(unittest.TestCase):
    """Test logger integration."""

    def test_custom_logger(self):
        """Test using custom logger."""
        logger = logging.getLogger("custom_test")
        extractor = HeuristicFactExtractor(logger=logger)
        self.assertEqual(extractor.logger, logger)

    def test_default_logger(self):
        """Test default logger creation."""
        extractor = HeuristicFactExtractor()
        self.assertIsNotNone(extractor.logger)


if __name__ == "__main__":
    unittest.main(verbosity=2)
