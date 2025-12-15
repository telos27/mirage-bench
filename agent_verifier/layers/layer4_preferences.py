"""Layer 4: User Preferences - Per-user personalization and preferences."""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .base_layer import BaseLayer, LayerResult
from ..schemas.request import VerificationRequest
from ..schemas.result import Severity
from ..schemas.rules import Rule, RuleType
from ..reasoning.datalog_engine import DatalogEngine, DatalogResult


class ResponseStyle(str, Enum):
    """Response style preferences."""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    SIMPLE = "simple"
    BALANCED = "balanced"


class ResponseLength(str, Enum):
    """Response length preferences."""
    SHORT = "short"      # < 200 chars
    MEDIUM = "medium"    # 200-1000 chars
    LONG = "long"        # > 1000 chars


class ResponseFormat(str, Enum):
    """Response format preferences."""
    PROSE = "prose"
    BULLETS = "bullets"
    NUMBERED = "numbered"
    CODE_HEAVY = "code_heavy"
    MIXED = "mixed"


class Tone(str, Enum):
    """Tone preferences."""
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    NEUTRAL = "neutral"


class ExpertiseLevel(str, Enum):
    """User expertise level."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


@dataclass
class UserPreferenceSet:
    """
    Collection of preferences for a user.

    Attributes:
        user_id: The user identifier
        response_style: Preferred response style
        response_length: Preferred response length
        response_format: Preferred format
        tone: Preferred tone
        language: Preferred language
        expertise_level: User's expertise level
        interested_topics: Topics user is interested in
        avoid_topics: Topics user wants to avoid
        code_languages: Preferred programming languages
        custom_preferences: Additional custom preferences
    """
    user_id: str
    response_style: ResponseStyle | None = None
    response_length: ResponseLength | None = None
    response_format: ResponseFormat | None = None
    tone: Tone | None = None
    language: str | None = None
    expertise_level: ExpertiseLevel | None = None
    interested_topics: list[str] = field(default_factory=list)
    avoid_topics: list[str] = field(default_factory=list)
    code_languages: list[str] = field(default_factory=list)
    custom_preferences: dict[str, Any] = field(default_factory=dict)


class UserPreferencesLayer(BaseLayer):
    """
    Layer 4: User Preferences.

    Checks that responses align with user-specific preferences:
    - Response style (concise, detailed, technical, simple)
    - Format preferences (bullets, prose, code examples)
    - Length preferences (short, medium, long)
    - Tone preferences (formal, casual, friendly)
    - Topic preferences (interests, avoidances)
    - Language preferences
    - Expertise level alignment

    Preferences can be:
    1. Explicit (user-configured in storage)
    2. Inferred (learned from user behavior)
    3. Session-based (temporary preferences)

    All preference checking is done via Soufflé Datalog.
    """

    # Path to built-in preference rules
    BUILTIN_RULES = Path(__file__).parent.parent / "reasoning" / "rules" / "user_preferences.dl"

    def __init__(
        self,
        storage: Any = None,
        preferences: dict[str, UserPreferenceSet] | None = None,
        strict_mode: bool = False,
    ):
        """
        Initialize Layer 4.

        Args:
            storage: Optional SQLiteStore for loading preferences
            preferences: Optional pre-configured preferences (user_id -> preferences)
            strict_mode: If True, preference mismatches are warnings; if False, info
        """
        super().__init__(layer_number=4, layer_name="User Preferences")

        self.datalog = DatalogEngine()
        self.storage = storage
        self.strict_mode = strict_mode

        # User preferences (user_id -> preferences)
        self._preferences: dict[str, UserPreferenceSet] = {}

        # Pre-configured preferences
        if preferences:
            self._preferences.update(preferences)

        # Session-based temporary preferences (cleared per session)
        self._session_preferences: dict[str, dict[str, Any]] = {}

        # In-memory rule storage
        self._dynamic_rules: list[Rule] = []

        # Additional Datalog rules
        self._extracted_rules: list[str] = []

        self._load_basic_rules()

    def _load_basic_rules(self) -> None:
        """Load basic rule metadata for user preferences."""
        basic_rules = [
            Rule(
                rule_id="up_style_alignment",
                name="Style Alignment",
                description="Response should match user's preferred style",
                rule_type=RuleType.PREFERENCE,
                layer=4,
                conditions=[],
                severity=Severity.INFO,
                message_template="Style mismatch: {detail}",
                tags=["user", "style"],
            ),
            Rule(
                rule_id="up_format_alignment",
                name="Format Alignment",
                description="Response should match user's preferred format",
                rule_type=RuleType.PREFERENCE,
                layer=4,
                conditions=[],
                severity=Severity.INFO,
                message_template="Format mismatch: {detail}",
                tags=["user", "format"],
            ),
            Rule(
                rule_id="up_topic_avoidance",
                name="Topic Avoidance",
                description="Response should avoid topics user wants to avoid",
                rule_type=RuleType.PREFERENCE,
                layer=4,
                conditions=[],
                severity=Severity.WARNING,
                message_template="Avoided topic mentioned: {detail}",
                tags=["user", "topics"],
            ),
        ]
        self._dynamic_rules.extend(basic_rules)

    def set_preferences(self, user_id: str, preferences: UserPreferenceSet) -> None:
        """
        Set preferences for a user.

        Args:
            user_id: The user identifier
            preferences: The preference set
        """
        self._preferences[user_id] = preferences

    def get_preferences(self, user_id: str) -> UserPreferenceSet | None:
        """
        Get preferences for a user.

        Args:
            user_id: The user identifier

        Returns:
            UserPreferenceSet or None if not found
        """
        return self._preferences.get(user_id)

    def update_preference(self, user_id: str, key: str, value: Any) -> None:
        """
        Update a single preference for a user.

        Args:
            user_id: The user identifier
            key: The preference key
            value: The preference value
        """
        if user_id not in self._preferences:
            self._preferences[user_id] = UserPreferenceSet(user_id=user_id)

        pref = self._preferences[user_id]
        if hasattr(pref, key):
            setattr(pref, key, value)
        else:
            pref.custom_preferences[key] = value

    def clear_preferences(self, user_id: str) -> None:
        """
        Clear all preferences for a user.

        Args:
            user_id: The user identifier
        """
        if user_id in self._preferences:
            del self._preferences[user_id]

    def set_session_preference(self, user_id: str, key: str, value: Any) -> None:
        """
        Set a temporary session-based preference.

        Args:
            user_id: The user identifier
            key: The preference key
            value: The preference value
        """
        if user_id not in self._session_preferences:
            self._session_preferences[user_id] = {}
        self._session_preferences[user_id][key] = value

    def clear_session_preferences(self, user_id: str | None = None) -> None:
        """
        Clear session preferences.

        Args:
            user_id: User to clear (None = all users)
        """
        if user_id:
            if user_id in self._session_preferences:
                del self._session_preferences[user_id]
        else:
            self._session_preferences.clear()

    def load_preferences_from_storage(self, user_id: str, deployment_id: str) -> UserPreferenceSet | None:
        """
        Load preferences from storage.

        Args:
            user_id: The user identifier
            deployment_id: The deployment identifier

        Returns:
            Loaded UserPreferenceSet or None
        """
        if not self.storage:
            return None

        prefs_dict = self.storage.get_all_preferences(
            user_id=user_id,
            deployment_id=deployment_id,
            enabled_only=True,
        )

        if not prefs_dict:
            return None

        pref_set = UserPreferenceSet(user_id=user_id)

        # Map storage keys to preference fields
        key_mapping = {
            "response_style": ("response_style", lambda v: ResponseStyle(v) if v else None),
            "response_length": ("response_length", lambda v: ResponseLength(v) if v else None),
            "response_format": ("response_format", lambda v: ResponseFormat(v) if v else None),
            "tone": ("tone", lambda v: Tone(v) if v else None),
            "language": ("language", lambda v: v),
            "expertise_level": ("expertise_level", lambda v: ExpertiseLevel(v) if v else None),
            "interested_topics": ("interested_topics", lambda v: v if isinstance(v, list) else []),
            "avoid_topics": ("avoid_topics", lambda v: v if isinstance(v, list) else []),
            "code_languages": ("code_languages", lambda v: v if isinstance(v, list) else []),
        }

        for storage_key, (attr_name, converter) in key_mapping.items():
            if storage_key in prefs_dict:
                try:
                    setattr(pref_set, attr_name, converter(prefs_dict[storage_key]))
                except (ValueError, KeyError):
                    pass  # Invalid enum value, skip

        # Store any unmapped preferences as custom
        for key, value in prefs_dict.items():
            if key not in key_mapping:
                pref_set.custom_preferences[key] = value

        self._preferences[user_id] = pref_set
        return pref_set

    def add_extracted_rule(self, datalog_rule: str) -> None:
        """
        Add an extracted Datalog rule string.

        Args:
            datalog_rule: Datalog rule code as string
        """
        self._extracted_rules.append(datalog_rule)

    def clear_extracted_rules(self) -> None:
        """Clear all extracted rules."""
        self._extracted_rules.clear()

    def _get_rules_program(self) -> str:
        """
        Combine all rules into a single Datalog program.

        Returns:
            Complete Datalog program string
        """
        program = ""

        # Start with built-in rules
        if self.BUILTIN_RULES.exists():
            with open(self.BUILTIN_RULES) as f:
                program = f.read()

        # Add extracted rules
        if self._extracted_rules:
            program += "\n\n// Extracted Rules\n"
            for rule in self._extracted_rules:
                program += rule + "\n"

        return program

    def _detect_output_style(self, output: str) -> str:
        """Detect the response style from output."""
        output_lower = output.lower()
        word_count = len(output.split())

        # Technical indicators
        technical_indicators = [
            "algorithm", "implementation", "complexity", "function",
            "parameter", "return", "exception", "interface", "api",
        ]
        technical_score = sum(1 for ind in technical_indicators if ind in output_lower)

        # Simple indicators
        simple_indicators = [
            "basically", "simply", "just", "easy", "straightforward",
        ]
        simple_score = sum(1 for ind in simple_indicators if ind in output_lower)

        if technical_score >= 3:
            return "technical"
        if simple_score >= 2 or word_count < 50:
            return "simple"
        if word_count > 200:
            return "detailed"
        if word_count < 100:
            return "concise"
        return "balanced"

    def _detect_output_length(self, output: str) -> str:
        """Detect the response length category."""
        length = len(output)
        if length < 200:
            return "short"
        if length < 1000:
            return "medium"
        return "long"

    def _detect_output_format(self, output: str) -> str:
        """Detect the response format."""
        bullet_count = output.count("- ") + output.count("* ")
        numbered_count = len(re.findall(r'^\d+\.', output, re.MULTILINE))
        code_blocks = output.count("```")

        if code_blocks >= 2:
            return "code_heavy"
        if numbered_count >= 3:
            return "numbered"
        if bullet_count >= 3:
            return "bullets"
        if bullet_count >= 1 or code_blocks >= 1:
            return "mixed"
        return "prose"

    def _detect_output_tone(self, output: str) -> str:
        """Detect the response tone."""
        output_lower = output.lower()

        formal_indicators = [
            "therefore", "furthermore", "consequently", "regarding",
            "pursuant", "hereby", "accordingly",
        ]
        casual_indicators = [
            "hey", "cool", "awesome", "gonna", "wanna", "yeah",
            "btw", "fyi", "lol",
        ]
        friendly_indicators = [
            "happy to help", "glad to", "hope this helps", "feel free",
            "let me know", "don't hesitate",
        ]

        formal_score = sum(1 for ind in formal_indicators if ind in output_lower)
        casual_score = sum(1 for ind in casual_indicators if ind in output_lower)
        friendly_score = sum(1 for ind in friendly_indicators if ind in output_lower)

        if casual_score >= 2:
            return "casual"
        if friendly_score >= 2:
            return "friendly"
        if formal_score >= 2:
            return "formal"
        return "neutral"

    def _detect_output_expertise(self, output: str) -> str:
        """Detect the expertise level of the response."""
        output_lower = output.lower()

        # Expert indicators
        expert_indicators = [
            "implementation detail", "under the hood", "optimization",
            "edge case", "trade-off", "complexity", "architecture",
        ]
        # Beginner indicators
        beginner_indicators = [
            "let me explain", "simply put", "in other words",
            "for example", "think of it as", "basically",
        ]

        expert_score = sum(1 for ind in expert_indicators if ind in output_lower)
        beginner_score = sum(1 for ind in beginner_indicators if ind in output_lower)

        if expert_score >= 2:
            return "expert"
        if beginner_score >= 2:
            return "beginner"
        return "intermediate"

    def _detect_topics(self, output: str) -> set[str]:
        """Detect topics discussed in output."""
        output_lower = output.lower()
        topics = set()

        # Topic keyword mapping
        topic_keywords = {
            "programming": ["code", "function", "variable", "class", "method"],
            "database": ["sql", "query", "table", "database", "schema"],
            "security": ["password", "encryption", "authentication", "security"],
            "networking": ["http", "api", "request", "response", "server"],
            "finance": ["money", "payment", "transaction", "account", "balance"],
            "health": ["medical", "health", "symptom", "treatment", "diagnosis"],
            "politics": ["government", "policy", "election", "political", "vote"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in output_lower for kw in keywords):
                topics.add(topic)

        return topics

    def _detect_code_language(self, output: str) -> str | None:
        """Detect programming language used in code blocks."""
        # Check for language-specific patterns
        lang_patterns = {
            "python": [r'def \w+\(', r'import \w+', r'class \w+:', r'print\('],
            "javascript": [r'function \w+', r'const \w+', r'let \w+', r'=>'],
            "java": [r'public class', r'public static void', r'System\.out'],
            "rust": [r'fn \w+', r'let mut', r'impl ', r'pub fn'],
            "go": [r'func \w+', r'package \w+', r':= '],
        }

        for lang, patterns in lang_patterns.items():
            if any(re.search(p, output) for p in patterns):
                return lang

        return None

    def _extract_output_facts(
        self,
        case_id: str,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract facts from output for Datalog reasoning.

        Args:
            case_id: The case identifier
            request: The verification request
            context: Additional context

        Returns:
            Dictionary of extracted facts
        """
        output = request.llm_output

        facts = {
            "style": self._detect_output_style(output),
            "length": self._detect_output_length(output),
            "format": self._detect_output_format(output),
            "tone": self._detect_output_tone(output),
            "language": "english",  # Default, would need NLP for real detection
            "expertise": self._detect_output_expertise(output),
            "topics": self._detect_topics(output),
            "code_language": self._detect_code_language(output),
        }

        # Use context if available
        if "extracted_facts" in context:
            ef = context["extracted_facts"]
            if "output_language" in ef:
                facts["language"] = ef["output_language"]
            if "output_topics" in ef:
                facts["topics"].update(ef["output_topics"])

        return facts

    def _populate_datalog_facts(
        self,
        case_id: str,
        request: VerificationRequest,
        context: dict[str, Any],
        user_prefs: UserPreferenceSet | None,
    ) -> None:
        """Populate the Datalog engine with facts for preference checking."""
        self.datalog.clear_facts()

        user_id = request.user_id or "anonymous"

        # Basic request info
        self.datalog.add_fact("request_id", case_id)
        self.datalog.add_fact("user_id", case_id, user_id)
        self.datalog.add_fact("deployment_id", case_id, request.deployment_id)

        # Extract output facts
        facts = self._extract_output_facts(case_id, request, context)

        # Output analysis facts
        self.datalog.add_fact("output_response_style", case_id, facts["style"])
        self.datalog.add_fact("output_length_category", case_id, facts["length"])
        self.datalog.add_fact("output_format_type", case_id, facts["format"])
        self.datalog.add_fact("output_tone", case_id, facts["tone"])
        self.datalog.add_fact("output_language", case_id, facts["language"])
        self.datalog.add_fact("output_expertise_level", case_id, facts["expertise"])

        for topic in facts["topics"]:
            self.datalog.add_fact("output_topic", case_id, topic)

        if facts["code_language"]:
            self.datalog.add_fact("output_code_language", case_id, facts["code_language"])

        # User preference facts
        if user_prefs:
            if user_prefs.response_style:
                self.datalog.add_fact("pref_response_style", case_id, user_id, user_prefs.response_style.value)

            if user_prefs.response_length:
                self.datalog.add_fact("pref_response_length", case_id, user_id, user_prefs.response_length.value)

            if user_prefs.response_format:
                self.datalog.add_fact("pref_response_format", case_id, user_id, user_prefs.response_format.value)

            if user_prefs.tone:
                self.datalog.add_fact("pref_tone", case_id, user_id, user_prefs.tone.value)

            if user_prefs.language:
                self.datalog.add_fact("pref_language", case_id, user_id, user_prefs.language.lower())

            if user_prefs.expertise_level:
                self.datalog.add_fact("pref_expertise_level", case_id, user_id, user_prefs.expertise_level.value)

            for topic in user_prefs.interested_topics:
                self.datalog.add_fact("pref_interested_topic", case_id, user_id, topic.lower())

            for topic in user_prefs.avoid_topics:
                self.datalog.add_fact("pref_avoid_topic", case_id, user_id, topic.lower())

            for lang in user_prefs.code_languages:
                self.datalog.add_fact("pref_code_language", case_id, user_id, lang.lower())

            # General preference tracking
            for key, value in user_prefs.custom_preferences.items():
                if isinstance(value, str):
                    self.datalog.add_fact("user_preference", case_id, user_id, key, value, "explicit", 100)

        # Session preferences
        session_prefs = self._session_preferences.get(user_id, {})
        for key, value in session_prefs.items():
            if isinstance(value, str):
                self.datalog.add_fact("user_preference", case_id, user_id, key, value, "session", 90)

    def _parse_datalog_violations(
        self,
        result: DatalogResult,
        case_id: str,
    ) -> list[tuple[str, str, str, str]]:
        """
        Parse violations from Datalog output.

        Returns:
            List of (user_id, violation_type, detail, severity) tuples
        """
        violations = []
        for row in result.get_relation("output_preference_violation"):
            if len(row) >= 5 and row[0] == case_id:
                violations.append((row[1], row[2], row[3], row[4]))  # user, type, detail, severity
        return violations

    def check(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> LayerResult:
        """
        Check user preferences against the response.

        Args:
            request: The verification request
            context: Accumulated context from previous layers

        Returns:
            LayerResult with violations and reasoning
        """
        result = LayerResult(layer=self.layer_number)
        case_id = request.request_id
        user_id = request.user_id

        # Step 1: Get user preferences
        user_prefs = None
        if user_id:
            user_prefs = self.get_preferences(user_id)

            # Try loading from storage if not already loaded
            if not user_prefs and self.storage:
                user_prefs = self.load_preferences_from_storage(user_id, request.deployment_id)

        result.add_reasoning(self.create_reasoning_step(
            step_type="preference_lookup",
            description="Retrieved user preferences",
            inputs={"user_id": user_id},
            outputs={
                "has_preferences": user_prefs is not None,
                "preference_keys": list(vars(user_prefs).keys()) if user_prefs else [],
            },
        ))

        if not user_id or not user_prefs:
            result.add_reasoning(self.create_reasoning_step(
                step_type="skip",
                description="No user preferences configured",
                inputs={"user_id": user_id},
                outputs={"skipped": True},
            ))
            return result

        # Step 2: Populate facts
        self._populate_datalog_facts(case_id, request, context, user_prefs)

        # Step 3: Run Datalog rules
        program = self._get_rules_program()

        if program.strip():
            datalog_result = self.datalog.run_inline(
                program,
                output_relations=[
                    "output_preference_alignment",
                    "output_preference_violation",
                    "output_preference_summary",
                    "output_violation_count",
                ]
            )

            if datalog_result.success:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description="Applied user preference Datalog rules",
                    inputs={"program_lines": len(program.split("\n"))},
                    outputs={"success": True},
                ))

                # Parse violations
                dl_violations = self._parse_datalog_violations(datalog_result, case_id)

                # Determine severity based on strict mode
                base_severity = "warning" if self.strict_mode else "info"

                for user, vtype, detail, severity in dl_violations:
                    # Override severity for non-critical mismatches
                    if vtype in ["style_mismatch", "length_mismatch", "format_mismatch", "tone_mismatch"]:
                        severity = base_severity

                    result.add_violation(self.create_violation(
                        violation_type=vtype,
                        message=f"[User: {user}] {vtype}: {detail}",
                        evidence={
                            "user_id": user,
                            "detail": detail,
                            "source": "datalog",
                        },
                        severity=severity,
                        rule_id=f"up_{vtype}",
                    ))

                # Store alignment info in metadata
                for row in datalog_result.get_relation("output_preference_alignment"):
                    if len(row) >= 4 and row[0] == case_id:
                        result.metadata["preferences_aligned"] = row[2] == "true"
                        result.metadata["alignment_score"] = int(row[3])

                # Store summary
                for row in datalog_result.get_relation("output_preference_summary"):
                    if len(row) >= 4 and row[0] == case_id:
                        result.metadata["total_preferences"] = int(row[2])
                        result.metadata["preference_violations"] = int(row[3])
            else:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description=f"Datalog execution failed: {datalog_result.error}",
                    inputs={"program_lines": len(program.split("\n"))},
                    outputs={"success": False, "error": datalog_result.error},
                ))
        else:
            result.add_reasoning(self.create_reasoning_step(
                step_type="rule_application",
                description="No Datalog rules to apply",
                inputs={},
                outputs={"skipped": True},
            ))

        # Store preference info for downstream layers
        result.facts_extracted = {
            "user_id": user_id,
            "has_preferences": user_prefs is not None,
            "preferences_aligned": result.metadata.get("preferences_aligned", True),
        }

        return result

    def load_rules(self, deployment_id: str) -> list[Rule]:
        """
        Load rules for this layer.

        Args:
            deployment_id: The deployment identifier

        Returns:
            List of active rules
        """
        rules = list(self._dynamic_rules)

        # Load from storage if available
        if self.storage:
            stored_rules = self.storage.get_rules_for_layer(
                layer=4,
                deployment_id=deployment_id,
                enabled_only=True,
            )
            for model in stored_rules:
                import json
                spec = json.loads(model.rule_spec)
                rules.append(Rule.from_dict(spec))

        return rules


# Convenience functions for common preference configurations

def create_developer_preferences(
    user_id: str,
    code_language: str = "python",
    expertise: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE,
) -> UserPreferenceSet:
    """
    Create preferences for a developer user.

    Args:
        user_id: The user identifier
        code_language: Preferred programming language
        expertise: User's expertise level

    Returns:
        UserPreferenceSet configured for developers
    """
    return UserPreferenceSet(
        user_id=user_id,
        response_style=ResponseStyle.TECHNICAL,
        response_format=ResponseFormat.CODE_HEAVY,
        tone=Tone.PROFESSIONAL,
        expertise_level=expertise,
        code_languages=[code_language],
    )


def create_beginner_preferences(user_id: str) -> UserPreferenceSet:
    """
    Create preferences for a beginner user.

    Args:
        user_id: The user identifier

    Returns:
        UserPreferenceSet configured for beginners
    """
    return UserPreferenceSet(
        user_id=user_id,
        response_style=ResponseStyle.SIMPLE,
        response_length=ResponseLength.MEDIUM,
        response_format=ResponseFormat.MIXED,
        tone=Tone.FRIENDLY,
        expertise_level=ExpertiseLevel.BEGINNER,
    )


def create_executive_preferences(user_id: str) -> UserPreferenceSet:
    """
    Create preferences for an executive user.

    Args:
        user_id: The user identifier

    Returns:
        UserPreferenceSet configured for executives
    """
    return UserPreferenceSet(
        user_id=user_id,
        response_style=ResponseStyle.CONCISE,
        response_length=ResponseLength.SHORT,
        response_format=ResponseFormat.BULLETS,
        tone=Tone.FORMAL,
        expertise_level=ExpertiseLevel.INTERMEDIATE,
    )
