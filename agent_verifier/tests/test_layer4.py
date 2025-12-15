"""Tests for Layer 4: User Preferences."""

import pytest
from ..layers.layer4_preferences import (
    UserPreferencesLayer,
    UserPreferenceSet,
    ResponseStyle,
    ResponseLength,
    ResponseFormat,
    Tone,
    ExpertiseLevel,
    create_developer_preferences,
    create_beginner_preferences,
    create_executive_preferences,
)
from ..schemas.request import VerificationRequest


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def layer():
    """Create a basic layer without storage."""
    return UserPreferencesLayer()


@pytest.fixture
def developer_prefs():
    """Create developer preferences for testing."""
    return UserPreferenceSet(
        user_id="dev_user",
        response_style=ResponseStyle.TECHNICAL,
        response_format=ResponseFormat.CODE_HEAVY,
        tone=Tone.PROFESSIONAL,
        expertise_level=ExpertiseLevel.EXPERT,
        code_languages=["python", "javascript"],
    )


@pytest.fixture
def beginner_prefs():
    """Create beginner preferences for testing."""
    return UserPreferenceSet(
        user_id="beginner_user",
        response_style=ResponseStyle.SIMPLE,
        response_length=ResponseLength.MEDIUM,
        response_format=ResponseFormat.MIXED,
        tone=Tone.FRIENDLY,
        expertise_level=ExpertiseLevel.BEGINNER,
    )


@pytest.fixture
def topic_prefs():
    """Create preferences with topic settings."""
    return UserPreferenceSet(
        user_id="topic_user",
        interested_topics=["programming", "technology"],
        avoid_topics=["politics", "finance"],
    )


def make_request(
    prompt: str = "Test prompt",
    output: str = "Test output",
    deployment_id: str = "test_deployment",
    request_id: str = "test_001",
    user_id: str | None = None,
) -> VerificationRequest:
    """Helper to create test requests."""
    return VerificationRequest(
        request_id=request_id,
        deployment_id=deployment_id,
        prompt=prompt,
        llm_output=output,
        llm_model="test-model",
        user_id=user_id,
    )


# ============================================
# Enum Tests
# ============================================

class TestEnums:
    """Tests for preference enums."""

    def test_response_style_values(self):
        """Test ResponseStyle enum values."""
        assert ResponseStyle.CONCISE.value == "concise"
        assert ResponseStyle.DETAILED.value == "detailed"
        assert ResponseStyle.TECHNICAL.value == "technical"
        assert ResponseStyle.SIMPLE.value == "simple"
        assert ResponseStyle.BALANCED.value == "balanced"

    def test_response_length_values(self):
        """Test ResponseLength enum values."""
        assert ResponseLength.SHORT.value == "short"
        assert ResponseLength.MEDIUM.value == "medium"
        assert ResponseLength.LONG.value == "long"

    def test_response_format_values(self):
        """Test ResponseFormat enum values."""
        assert ResponseFormat.PROSE.value == "prose"
        assert ResponseFormat.BULLETS.value == "bullets"
        assert ResponseFormat.NUMBERED.value == "numbered"
        assert ResponseFormat.CODE_HEAVY.value == "code_heavy"
        assert ResponseFormat.MIXED.value == "mixed"

    def test_tone_values(self):
        """Test Tone enum values."""
        assert Tone.FORMAL.value == "formal"
        assert Tone.CASUAL.value == "casual"
        assert Tone.FRIENDLY.value == "friendly"
        assert Tone.PROFESSIONAL.value == "professional"
        assert Tone.NEUTRAL.value == "neutral"

    def test_expertise_level_values(self):
        """Test ExpertiseLevel enum values."""
        assert ExpertiseLevel.BEGINNER.value == "beginner"
        assert ExpertiseLevel.INTERMEDIATE.value == "intermediate"
        assert ExpertiseLevel.EXPERT.value == "expert"


# ============================================
# UserPreferenceSet Tests
# ============================================

class TestUserPreferenceSet:
    """Tests for UserPreferenceSet dataclass."""

    def test_basic_creation(self):
        """Test basic preference set creation."""
        prefs = UserPreferenceSet(user_id="test_user")
        assert prefs.user_id == "test_user"
        assert prefs.response_style is None
        assert prefs.interested_topics == []
        assert prefs.avoid_topics == []

    def test_full_creation(self, developer_prefs):
        """Test preference set with all fields."""
        assert developer_prefs.user_id == "dev_user"
        assert developer_prefs.response_style == ResponseStyle.TECHNICAL
        assert developer_prefs.expertise_level == ExpertiseLevel.EXPERT
        assert "python" in developer_prefs.code_languages

    def test_custom_preferences(self):
        """Test custom preferences field."""
        prefs = UserPreferenceSet(
            user_id="test_user",
            custom_preferences={"theme": "dark", "font_size": 14},
        )
        assert prefs.custom_preferences["theme"] == "dark"
        assert prefs.custom_preferences["font_size"] == 14


# ============================================
# Convenience Function Tests
# ============================================

class TestConvenienceFunctions:
    """Tests for preference creation helper functions."""

    def test_create_developer_preferences(self):
        """Test create_developer_preferences helper."""
        prefs = create_developer_preferences(
            user_id="dev1",
            code_language="rust",
            expertise=ExpertiseLevel.EXPERT,
        )
        assert prefs.user_id == "dev1"
        assert prefs.response_style == ResponseStyle.TECHNICAL
        assert prefs.response_format == ResponseFormat.CODE_HEAVY
        assert "rust" in prefs.code_languages
        assert prefs.expertise_level == ExpertiseLevel.EXPERT

    def test_create_beginner_preferences(self):
        """Test create_beginner_preferences helper."""
        prefs = create_beginner_preferences(user_id="newbie")
        assert prefs.user_id == "newbie"
        assert prefs.response_style == ResponseStyle.SIMPLE
        assert prefs.tone == Tone.FRIENDLY
        assert prefs.expertise_level == ExpertiseLevel.BEGINNER

    def test_create_executive_preferences(self):
        """Test create_executive_preferences helper."""
        prefs = create_executive_preferences(user_id="ceo")
        assert prefs.user_id == "ceo"
        assert prefs.response_style == ResponseStyle.CONCISE
        assert prefs.response_length == ResponseLength.SHORT
        assert prefs.response_format == ResponseFormat.BULLETS
        assert prefs.tone == Tone.FORMAL


# ============================================
# Layer Basic Tests
# ============================================

class TestUserPreferencesLayerBasic:
    """Basic tests for UserPreferencesLayer."""

    def test_layer_creation(self, layer):
        """Test layer initialization."""
        assert layer.layer_number == 4
        assert layer.layer_name == "User Preferences"
        assert layer.datalog is not None
        assert layer.strict_mode is False

    def test_strict_mode(self):
        """Test strict mode initialization."""
        layer = UserPreferencesLayer(strict_mode=True)
        assert layer.strict_mode is True

    def test_set_preferences(self, layer, developer_prefs):
        """Test setting preferences."""
        layer.set_preferences("dev_user", developer_prefs)
        retrieved = layer.get_preferences("dev_user")
        assert retrieved is not None
        assert retrieved.user_id == "dev_user"
        assert retrieved.response_style == ResponseStyle.TECHNICAL

    def test_get_preferences_not_found(self, layer):
        """Test getting preferences for unknown user."""
        prefs = layer.get_preferences("unknown_user")
        assert prefs is None

    def test_update_preference(self, layer):
        """Test updating a single preference."""
        layer.update_preference("user1", "response_style", ResponseStyle.CONCISE)
        prefs = layer.get_preferences("user1")
        assert prefs is not None
        assert prefs.response_style == ResponseStyle.CONCISE

    def test_update_custom_preference(self, layer):
        """Test updating a custom preference."""
        layer.update_preference("user1", "custom_key", "custom_value")
        prefs = layer.get_preferences("user1")
        assert prefs is not None
        assert prefs.custom_preferences["custom_key"] == "custom_value"

    def test_clear_preferences(self, layer, developer_prefs):
        """Test clearing preferences."""
        layer.set_preferences("dev_user", developer_prefs)
        layer.clear_preferences("dev_user")
        assert layer.get_preferences("dev_user") is None

    def test_layer_with_initial_preferences(self, developer_prefs, beginner_prefs):
        """Test layer creation with initial preferences."""
        layer = UserPreferencesLayer(preferences={
            "dev_user": developer_prefs,
            "beginner_user": beginner_prefs,
        })
        assert layer.get_preferences("dev_user") is not None
        assert layer.get_preferences("beginner_user") is not None


# ============================================
# Session Preferences Tests
# ============================================

class TestSessionPreferences:
    """Tests for session-based preferences."""

    def test_set_session_preference(self, layer):
        """Test setting a session preference."""
        layer.set_session_preference("user1", "temp_key", "temp_value")
        assert layer._session_preferences["user1"]["temp_key"] == "temp_value"

    def test_clear_session_preference_single(self, layer):
        """Test clearing session preferences for one user."""
        layer.set_session_preference("user1", "key1", "value1")
        layer.set_session_preference("user2", "key2", "value2")
        layer.clear_session_preferences("user1")
        assert "user1" not in layer._session_preferences
        assert "user2" in layer._session_preferences

    def test_clear_session_preferences_all(self, layer):
        """Test clearing all session preferences."""
        layer.set_session_preference("user1", "key1", "value1")
        layer.set_session_preference("user2", "key2", "value2")
        layer.clear_session_preferences()
        assert len(layer._session_preferences) == 0


# ============================================
# Check Method Tests - No User
# ============================================

class TestCheckNoUser:
    """Tests for check() when no user is specified."""

    def test_no_user_skips(self, layer):
        """Test that check skips when no user specified."""
        request = make_request(output="Any output here", user_id=None)
        result = layer.check(request, {})

        assert len(result.violations) == 0
        skip_steps = [r for r in result.reasoning if r.step_type == "skip"]
        assert len(skip_steps) > 0

    def test_no_preferences_skips(self, layer):
        """Test that check skips when user has no preferences."""
        request = make_request(output="Any output here", user_id="unknown_user")
        result = layer.check(request, {})

        assert len(result.violations) == 0
        skip_steps = [r for r in result.reasoning if r.step_type == "skip"]
        assert len(skip_steps) > 0


# ============================================
# Style Detection Tests
# ============================================

class TestStyleDetection:
    """Tests for output style detection."""

    def test_detect_technical_style(self, layer):
        """Test detection of technical response style."""
        output = """
        The algorithm implementation uses O(n log n) complexity.
        The function parameter accepts an interface type.
        Exception handling is done via the API wrapper.
        """
        style = layer._detect_output_style(output)
        assert style == "technical"

    def test_detect_simple_style(self, layer):
        """Test detection of simple response style."""
        output = "Simply put, just do this. It's easy and straightforward."
        style = layer._detect_output_style(output)
        assert style == "simple"

    def test_detect_detailed_style(self, layer):
        """Test detection of detailed response style."""
        output = " ".join(["word"] * 250)  # Long text
        style = layer._detect_output_style(output)
        assert style == "detailed"

    def test_detect_concise_style(self, layer):
        """Test detection of concise response style."""
        # Concise: 50-100 words, no simple/technical indicators
        output = "Here is the information you requested about the system configuration. " * 5
        style = layer._detect_output_style(output)
        assert style == "concise"


# ============================================
# Length Detection Tests
# ============================================

class TestLengthDetection:
    """Tests for output length detection."""

    def test_detect_short_length(self, layer):
        """Test detection of short response."""
        output = "x" * 100
        length = layer._detect_output_length(output)
        assert length == "short"

    def test_detect_medium_length(self, layer):
        """Test detection of medium response."""
        output = "x" * 500
        length = layer._detect_output_length(output)
        assert length == "medium"

    def test_detect_long_length(self, layer):
        """Test detection of long response."""
        output = "x" * 1500
        length = layer._detect_output_length(output)
        assert length == "long"


# ============================================
# Format Detection Tests
# ============================================

class TestFormatDetection:
    """Tests for output format detection."""

    def test_detect_bullets_format(self, layer):
        """Test detection of bullet format."""
        output = """
        Here are the steps:
        - First item
        - Second item
        - Third item
        - Fourth item
        """
        fmt = layer._detect_output_format(output)
        assert fmt == "bullets"

    def test_detect_numbered_format(self, layer):
        """Test detection of numbered format."""
        output = """Steps:
1. First step
2. Second step
3. Third step
4. Fourth step"""
        fmt = layer._detect_output_format(output)
        assert fmt == "numbered"

    def test_detect_code_heavy_format(self, layer):
        """Test detection of code-heavy format."""
        output = """
        Here's the code:
        ```python
        def hello():
            print("Hello")
        ```
        And more code:
        ```python
        def world():
            print("World")
        ```
        """
        fmt = layer._detect_output_format(output)
        assert fmt == "code_heavy"

    def test_detect_prose_format(self, layer):
        """Test detection of prose format."""
        output = "This is a paragraph of text without any special formatting or lists."
        fmt = layer._detect_output_format(output)
        assert fmt == "prose"


# ============================================
# Tone Detection Tests
# ============================================

class TestToneDetection:
    """Tests for output tone detection."""

    def test_detect_formal_tone(self, layer):
        """Test detection of formal tone."""
        output = "Therefore, furthermore, it is consequently important to note accordingly."
        tone = layer._detect_output_tone(output)
        assert tone == "formal"

    def test_detect_casual_tone(self, layer):
        """Test detection of casual tone."""
        output = "Hey, that's cool! Gonna try this awesome thing, yeah!"
        tone = layer._detect_output_tone(output)
        assert tone == "casual"

    def test_detect_friendly_tone(self, layer):
        """Test detection of friendly tone."""
        output = "I'm happy to help! Hope this helps, feel free to let me know if you need more."
        tone = layer._detect_output_tone(output)
        assert tone == "friendly"

    def test_detect_neutral_tone(self, layer):
        """Test detection of neutral tone."""
        output = "Here is the information you requested."
        tone = layer._detect_output_tone(output)
        assert tone == "neutral"


# ============================================
# Topic Detection Tests
# ============================================

class TestTopicDetection:
    """Tests for topic detection."""

    def test_detect_programming_topic(self, layer):
        """Test detection of programming topic."""
        output = "The code defines a function with a variable inside the class method."
        topics = layer._detect_topics(output)
        assert "programming" in topics

    def test_detect_database_topic(self, layer):
        """Test detection of database topic."""
        output = "Run this SQL query against the database table schema."
        topics = layer._detect_topics(output)
        assert "database" in topics

    def test_detect_security_topic(self, layer):
        """Test detection of security topic."""
        output = "Use strong password encryption for authentication security."
        topics = layer._detect_topics(output)
        assert "security" in topics

    def test_detect_multiple_topics(self, layer):
        """Test detection of multiple topics."""
        output = "The code handles password encryption for the database query."
        topics = layer._detect_topics(output)
        assert len(topics) >= 2


# ============================================
# Code Language Detection Tests
# ============================================

class TestCodeLanguageDetection:
    """Tests for code language detection."""

    def test_detect_python(self, layer):
        """Test detection of Python code."""
        output = """
        def hello():
            import os
            print("Hello")
        """
        lang = layer._detect_code_language(output)
        assert lang == "python"

    def test_detect_javascript(self, layer):
        """Test detection of JavaScript code."""
        output = """
        function hello() {
            const x = 5;
            let y = () => x;
        }
        """
        lang = layer._detect_code_language(output)
        assert lang == "javascript"

    def test_no_code_detected(self, layer):
        """Test when no code is detected."""
        output = "This is just plain text with no code."
        lang = layer._detect_code_language(output)
        assert lang is None


# ============================================
# Style Mismatch Tests
# ============================================

class TestStyleMismatch:
    """Tests for style preference checking."""

    def test_style_mismatch_detected(self, layer, beginner_prefs):
        """Test detection of style mismatch."""
        layer.set_preferences("beginner_user", beginner_prefs)

        # Technical output for beginner who wants simple
        technical_output = """
        The algorithm implementation uses O(n log n) complexity.
        The function parameter accepts an interface type.
        Exception handling is done via the API layer.
        """
        request = make_request(
            output=technical_output,
            user_id="beginner_user",
        )
        result = layer.check(request, {})

        style_violations = [
            v for v in result.violations
            if v.violation_type == "style_mismatch"
        ]
        assert len(style_violations) >= 1

    def test_style_match_no_violation(self, layer, developer_prefs):
        """Test that matching style doesn't trigger violation."""
        layer.set_preferences("dev_user", developer_prefs)

        # Technical output for developer who wants technical
        technical_output = """
        The algorithm implementation uses O(n log n) complexity.
        The function parameter accepts an interface type.
        Exception handling is done via the API layer.
        """
        request = make_request(
            output=technical_output,
            user_id="dev_user",
        )
        result = layer.check(request, {})

        style_violations = [
            v for v in result.violations
            if v.violation_type == "style_mismatch"
        ]
        assert len(style_violations) == 0


# ============================================
# Length Mismatch Tests
# ============================================

class TestLengthMismatch:
    """Tests for length preference checking."""

    def test_length_mismatch_detected(self, layer):
        """Test detection of length mismatch."""
        prefs = UserPreferenceSet(
            user_id="short_user",
            response_length=ResponseLength.SHORT,
        )
        layer.set_preferences("short_user", prefs)

        # Long output for user who wants short
        long_output = "x" * 1500
        request = make_request(
            output=long_output,
            user_id="short_user",
        )
        result = layer.check(request, {})

        length_violations = [
            v for v in result.violations
            if v.violation_type == "length_mismatch"
        ]
        assert len(length_violations) >= 1

    def test_length_match_no_violation(self, layer):
        """Test that matching length doesn't trigger violation."""
        prefs = UserPreferenceSet(
            user_id="short_user",
            response_length=ResponseLength.SHORT,
        )
        layer.set_preferences("short_user", prefs)

        short_output = "x" * 100
        request = make_request(
            output=short_output,
            user_id="short_user",
        )
        result = layer.check(request, {})

        length_violations = [
            v for v in result.violations
            if v.violation_type == "length_mismatch"
        ]
        assert len(length_violations) == 0


# ============================================
# Avoided Topic Tests
# ============================================

class TestAvoidedTopics:
    """Tests for topic avoidance checking."""

    def test_avoided_topic_detected(self, layer, topic_prefs):
        """Test detection of avoided topic."""
        layer.set_preferences("topic_user", topic_prefs)

        # Output mentioning politics (avoided)
        output = "The government policy affects the election and political landscape."
        request = make_request(
            output=output,
            user_id="topic_user",
        )
        result = layer.check(request, {})

        topic_violations = [
            v for v in result.violations
            if v.violation_type == "avoided_topic_mentioned"
        ]
        assert len(topic_violations) >= 1

    def test_no_avoided_topic(self, layer, topic_prefs):
        """Test output without avoided topics."""
        layer.set_preferences("topic_user", topic_prefs)

        # Output about programming (allowed)
        output = "The code function uses a variable inside the class."
        request = make_request(
            output=output,
            user_id="topic_user",
        )
        result = layer.check(request, {})

        topic_violations = [
            v for v in result.violations
            if v.violation_type == "avoided_topic_mentioned"
        ]
        assert len(topic_violations) == 0


# ============================================
# Tone Mismatch Tests
# ============================================

class TestToneMismatch:
    """Tests for tone preference checking."""

    def test_tone_mismatch_detected(self, layer):
        """Test detection of tone mismatch."""
        prefs = UserPreferenceSet(
            user_id="formal_user",
            tone=Tone.FORMAL,
        )
        layer.set_preferences("formal_user", prefs)

        # Casual output for user who wants formal
        casual_output = "Hey, that's cool! Gonna try this awesome thing, yeah!"
        request = make_request(
            output=casual_output,
            user_id="formal_user",
        )
        result = layer.check(request, {})

        tone_violations = [
            v for v in result.violations
            if v.violation_type == "tone_mismatch"
        ]
        assert len(tone_violations) >= 1


# ============================================
# Format Mismatch Tests
# ============================================

class TestFormatMismatch:
    """Tests for format preference checking."""

    def test_format_mismatch_detected(self, layer):
        """Test detection of format mismatch."""
        prefs = UserPreferenceSet(
            user_id="bullets_user",
            response_format=ResponseFormat.BULLETS,
        )
        layer.set_preferences("bullets_user", prefs)

        # Prose output for user who wants bullets
        prose_output = "This is a paragraph without any bullets or lists."
        request = make_request(
            output=prose_output,
            user_id="bullets_user",
        )
        result = layer.check(request, {})

        format_violations = [
            v for v in result.violations
            if v.violation_type == "format_mismatch"
        ]
        assert len(format_violations) >= 1


# ============================================
# Strict Mode Tests
# ============================================

class TestStrictMode:
    """Tests for strict mode behavior."""

    def test_strict_mode_warnings(self):
        """Test that strict mode produces warnings."""
        layer = UserPreferencesLayer(strict_mode=True)
        prefs = UserPreferenceSet(
            user_id="test_user",
            response_style=ResponseStyle.SIMPLE,
        )
        layer.set_preferences("test_user", prefs)

        technical_output = """
        The algorithm implementation uses complexity analysis.
        The function parameter accepts an interface type.
        """
        request = make_request(
            output=technical_output,
            user_id="test_user",
        )
        result = layer.check(request, {})

        # In strict mode, mismatches should be warnings
        violations = [v for v in result.violations if v.violation_type == "style_mismatch"]
        if violations:
            assert violations[0].severity in ["warning", "info"]


# ============================================
# Extracted Rules Tests
# ============================================

class TestExtractedRules:
    """Tests for dynamically extracted rules."""

    def test_add_extracted_rule(self, layer):
        """Test adding an extracted Datalog rule."""
        layer.add_extracted_rule("custom_rule(x) :- some_fact(x).")
        assert len(layer._extracted_rules) == 1

    def test_clear_extracted_rules(self, layer):
        """Test clearing extracted rules."""
        layer.add_extracted_rule("rule1(x) :- fact1(x).")
        layer.add_extracted_rule("rule2(x) :- fact2(x).")
        layer.clear_extracted_rules()
        assert len(layer._extracted_rules) == 0


# ============================================
# Metadata Tests
# ============================================

class TestPreferenceMetadata:
    """Tests for preference metadata in results."""

    def test_facts_extracted_includes_user(self, layer, developer_prefs):
        """Test that facts include user info."""
        layer.set_preferences("dev_user", developer_prefs)
        request = make_request(output="Test output", user_id="dev_user")
        result = layer.check(request, {})

        assert "user_id" in result.facts_extracted
        assert result.facts_extracted["user_id"] == "dev_user"
        assert result.facts_extracted["has_preferences"] is True

    def test_preference_lookup_reasoning(self, layer, developer_prefs):
        """Test that preference lookup is recorded in reasoning."""
        layer.set_preferences("dev_user", developer_prefs)
        request = make_request(output="Test output", user_id="dev_user")
        result = layer.check(request, {})

        lookup_steps = [
            r for r in result.reasoning
            if r.step_type == "preference_lookup"
        ]
        assert len(lookup_steps) >= 1


# ============================================
# Load Rules Tests
# ============================================

class TestLoadRules:
    """Tests for load_rules method."""

    def test_load_rules_returns_basic_rules(self, layer):
        """Test that load_rules returns basic rules."""
        rules = layer.load_rules("test_deployment")
        assert len(rules) > 0
        rule_ids = [r.rule_id for r in rules]
        assert "up_style_alignment" in rule_ids
        assert "up_format_alignment" in rule_ids
        assert "up_topic_avoidance" in rule_ids


# ============================================
# Multiple Preferences Tests
# ============================================

class TestMultiplePreferences:
    """Tests for checking multiple preferences together."""

    def test_multiple_mismatches(self, layer):
        """Test that multiple mismatches are detected."""
        prefs = UserPreferenceSet(
            user_id="picky_user",
            response_style=ResponseStyle.SIMPLE,
            response_length=ResponseLength.SHORT,
            response_format=ResponseFormat.BULLETS,
            tone=Tone.FORMAL,
        )
        layer.set_preferences("picky_user", prefs)

        # Technical, long, prose, casual output
        bad_output = "Hey, " + " ".join(["algorithm implementation complexity"] * 100)
        request = make_request(
            output=bad_output,
            user_id="picky_user",
        )
        result = layer.check(request, {})

        # Should have multiple violations
        assert len(result.violations) >= 2

    def test_all_preferences_satisfied(self, layer):
        """Test output that satisfies all preferences."""
        prefs = UserPreferenceSet(
            user_id="happy_user",
            response_length=ResponseLength.SHORT,
        )
        layer.set_preferences("happy_user", prefs)

        # Short output
        short_output = "Here is your answer."
        request = make_request(
            output=short_output,
            user_id="happy_user",
        )
        result = layer.check(request, {})

        length_violations = [
            v for v in result.violations
            if v.violation_type == "length_mismatch"
        ]
        assert len(length_violations) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
