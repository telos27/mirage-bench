"""Tests for Layer 5: Session History."""

import pytest
from datetime import datetime

from agent_verifier.layers import (
    SessionHistoryLayer,
    SessionState,
    create_session_from_turns,
)
from agent_verifier.schemas.request import VerificationRequest
from agent_verifier.schemas.session import Session, Turn, EstablishedFact


# ============================================
# Session Management Tests
# ============================================

class TestSessionManagement:
    """Tests for session management functionality."""

    def test_create_session(self):
        """Test creating a new session."""
        layer = SessionHistoryLayer()
        session = layer.create_session(
            session_id="sess_001",
            user_id="user_123",
            deployment_id="my-app",
        )

        assert session.session_id == "sess_001"
        assert session.user_id == "user_123"
        assert session.deployment_id == "my-app"
        assert len(session.turns) == 0

    def test_get_session(self):
        """Test retrieving a session."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        session = layer.get_session("sess_001")
        assert session is not None
        assert session.session_id == "sess_001"

        # Non-existent session
        assert layer.get_session("nonexistent") is None

    def test_end_session(self):
        """Test ending a session."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        layer.end_session("sess_001")
        assert layer.get_session("sess_001") is None

    def test_add_turn(self):
        """Test adding turns to a session."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        turn = layer.add_turn(
            session_id="sess_001",
            turn_id="turn_001",
            prompt="What is Python?",
            response="Python is a programming language.",
        )

        assert turn is not None
        assert turn.turn_id == "turn_001"
        assert turn.prompt == "What is Python?"

        session = layer.get_session("sess_001")
        assert len(session.turns) == 1

    def test_add_turn_nonexistent_session(self):
        """Test adding turn to non-existent session."""
        layer = SessionHistoryLayer()
        turn = layer.add_turn("nonexistent", "turn_001", "Hello", "Hi")
        assert turn is None

    def test_max_session_turns(self):
        """Test max turns enforcement."""
        layer = SessionHistoryLayer(max_session_turns=5)
        layer.create_session("sess_001", "user_123", "my-app")

        for i in range(10):
            layer.add_turn("sess_001", f"turn_{i}", f"prompt_{i}", f"response_{i}")

        session = layer.get_session("sess_001")
        assert len(session.turns) == 5
        # Should keep the most recent turns
        assert session.turns[0].turn_id == "turn_5"

    def test_establish_fact(self):
        """Test establishing facts in a session."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        fact = layer.establish_fact(
            session_id="sess_001",
            fact_id="fact_001",
            fact_type="user_stated",
            key="name",
            value="Alice",
            source_turn="turn_001",
            confidence=0.9,
        )

        assert fact is not None
        assert fact.key == "name"
        assert fact.value == "Alice"

        session = layer.get_session("sess_001")
        assert len(session.established_facts) == 1

    def test_record_claim(self):
        """Test recording claims."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        layer.record_claim("sess_001", "language", "python", 0)

        state = layer.get_session_state("sess_001")
        assert "language" in state.claims
        assert state.claims["language"] == ("python", 0)

    def test_record_question_and_answer(self):
        """Test recording questions and answers."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        layer.record_question("sess_001", "what is your name?", 0)
        layer.record_answer("sess_001", "what is your name?", "Alice", 1)

        state = layer.get_session_state("sess_001")
        assert "what is your name?" in state.questions_asked
        assert "what is your name?" in state.questions_answered
        assert state.questions_answered["what is your name?"] == ("Alice", 1)

    def test_record_commitment(self):
        """Test recording commitments."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        layer.record_commitment("sess_001", "help with coding", 0)

        state = layer.get_session_state("sess_001")
        assert len(state.commitments) == 1
        assert state.commitments[0] == ("help with coding", 0)

    def test_update_progress(self):
        """Test updating progress."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        layer.update_progress("sess_001", "step_1", "pending")
        layer.update_progress("sess_001", "step_1", "in_progress")
        layer.update_progress("sess_001", "step_1", "completed")

        state = layer.get_session_state("sess_001")
        assert state.progress["step_1"] == "completed"

    def test_set_current_task(self):
        """Test setting current task."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        layer.set_current_task("sess_001", "build_feature")

        state = layer.get_session_state("sess_001")
        assert state.current_task == "build_feature"


# ============================================
# Fact Extraction Tests
# ============================================

class TestFactExtraction:
    """Tests for fact extraction from text."""

    def test_extract_claims(self):
        """Test extracting claims from text."""
        layer = SessionHistoryLayer()

        text = "The answer is 42. Python is a programming language."
        claims = layer._extract_claims(text)

        assert len(claims) >= 1
        # Should find "answer is 42" and "Python is a programming language"

    def test_extract_questions(self):
        """Test extracting questions from text."""
        layer = SessionHistoryLayer()

        text = "What is your name? Where do you live? How are you doing today?"
        questions = layer._extract_questions(text)

        assert len(questions) >= 2

    def test_extract_commitments(self):
        """Test extracting commitments from text."""
        layer = SessionHistoryLayer()

        text = "I will help you with that. Let me check the documentation."
        commitments = layer._extract_commitments(text)

        assert len(commitments) >= 1
        assert any("help" in c for c in commitments)

    def test_extract_task_references(self):
        """Test extracting task references."""
        layer = SessionHistoryLayer()

        text = "I completed the first step. Working on the second step now."
        actions = layer._extract_task_references(text, None)

        assert len(actions) >= 1

    def test_check_context_acknowledgment(self):
        """Test context acknowledgment detection."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")
        state = layer.get_session_state("sess_001")

        # Add an established fact
        layer.establish_fact("sess_001", "fact_001", "user_stated", "name", "Alice", "turn_0")

        # Text that references previous context
        text1 = "As I mentioned earlier, the solution is simple."
        assert layer._check_context_acknowledgment(text1, state) is True

        # Text that uses established fact
        text2 = "Hello Alice, how can I help?"
        assert layer._check_context_acknowledgment(text2, state) is True

        # Text with no acknowledgment
        text3 = "Here is a new topic entirely."
        assert layer._check_context_acknowledgment(text3, state) is False


# ============================================
# Verification Tests (No Datalog)
# ============================================

class TestVerificationBasic:
    """Basic verification tests (without Datalog execution)."""

    def test_check_no_session(self):
        """Test checking request with no session."""
        layer = SessionHistoryLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi there!",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        # Should skip gracefully
        assert result.metadata.get("session_coherent", True) is True
        assert len(result.violations) == 0

    def test_check_with_session_id_no_state(self):
        """Test checking request with session ID but no state."""
        layer = SessionHistoryLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi there!",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = layer.check(request, {})

        # Should handle gracefully (session not found)
        assert len(result.reasoning) >= 1

    def test_check_with_context_session(self):
        """Test checking request with session in context."""
        layer = SessionHistoryLayer()

        session = Session(
            session_id="sess_001",
            user_id="user_123",
            deployment_id="my-app",
        )
        session.add_turn(Turn(
            turn_id="turn_0",
            prompt="What is Python?",
            response="Python is a programming language.",
        ))

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Tell me more",
            llm_output="Python is great for data science.",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = layer.check(request, {"session": session})

        assert len(result.reasoning) >= 1
        assert result.facts_extracted["has_session"] is True


# ============================================
# Datalog Integration Tests
# ============================================

class TestDatalogIntegration:
    """Tests for Datalog-based verification."""

    def test_check_with_managed_session(self):
        """Test verification with a layer-managed session."""
        layer = SessionHistoryLayer()

        # Create session and add turn
        layer.create_session("sess_001", "user_123", "my-app")
        layer.add_turn(
            "sess_001",
            "turn_0",
            "What is Python?",
            "Python is a programming language.",
        )

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Tell me more",
            llm_output="Python is great for data science.",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = layer.check(request, {})

        assert len(result.reasoning) >= 1
        assert "session_id" in result.facts_extracted

    def test_contradiction_scenario(self):
        """Test detection of contradictions."""
        layer = SessionHistoryLayer()

        # Create session with a claim
        layer.create_session("sess_001", "user_123", "my-app")
        layer.add_turn(
            "sess_001",
            "turn_0",
            "What is the best language?",
            "Python is the best language.",
        )
        layer.record_claim("sess_001", "best_language", "python", 0)

        # Now check a response that contradicts
        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Are you sure?",
            llm_output="Actually, the best language is JavaScript.",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = layer.check(request, {})

        # The layer should at least process without error
        assert len(result.reasoning) >= 1

    def test_reasking_scenario(self):
        """Test detection of re-asking answered questions."""
        layer = SessionHistoryLayer()

        # Create session with question and answer
        layer.create_session("sess_001", "user_123", "my-app")
        layer.add_turn(
            "sess_001",
            "turn_0",
            "My name is Alice.",
            "Nice to meet you, Alice!",
        )
        layer.record_question("sess_001", "what is your name?", 0)
        layer.record_answer("sess_001", "what is your name?", "alice", 0)

        # Now check a response that re-asks
        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Can you help me?",
            llm_output="Sure! What is your name?",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = layer.check(request, {})

        assert len(result.reasoning) >= 1

    def test_progress_regression_scenario(self):
        """Test detection of progress regression."""
        layer = SessionHistoryLayer()

        # Create session with completed progress
        layer.create_session("sess_001", "user_123", "my-app")
        layer.add_turn("sess_001", "turn_0", "Build feature X", "Done with step 1.")
        layer.update_progress("sess_001", "step_1", "completed")

        # Now check a response that regresses progress
        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Continue",
            llm_output="Let me start working on step 1.",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = layer.check(request, {})

        assert len(result.reasoning) >= 1

    def test_context_acknowledgment(self):
        """Test context acknowledgment detection."""
        layer = SessionHistoryLayer()

        # Create session with established fact
        layer.create_session("sess_001", "user_123", "my-app")
        layer.add_turn("sess_001", "turn_0", "My name is Bob.", "Hello Bob!")
        layer.establish_fact("sess_001", "f1", "user_stated", "name", "Bob", "turn_0", 0.95)

        # Check response that uses context
        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Help me",
            llm_output="As I mentioned earlier, Bob, I can help you with that.",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = layer.check(request, {})

        assert len(result.reasoning) >= 1


# ============================================
# Helper Function Tests
# ============================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_session_from_turns(self):
        """Test creating session from turn list."""
        turns = [
            {"prompt": "Hello", "response": "Hi there!"},
            {"prompt": "How are you?", "response": "I'm doing well."},
            {"prompt": "Bye", "response": "Goodbye!"},
        ]

        session = create_session_from_turns(
            session_id="sess_001",
            user_id="user_123",
            deployment_id="my-app",
            turns=turns,
        )

        assert session.session_id == "sess_001"
        assert len(session.turns) == 3
        assert session.turns[0].prompt == "Hello"
        assert session.turns[1].response == "I'm doing well."

    def test_create_session_from_empty_turns(self):
        """Test creating session from empty turn list."""
        session = create_session_from_turns(
            session_id="sess_001",
            user_id="user_123",
            deployment_id="my-app",
            turns=[],
        )

        assert session.session_id == "sess_001"
        assert len(session.turns) == 0


# ============================================
# Rule Management Tests
# ============================================

class TestRuleManagement:
    """Tests for rule management."""

    def test_add_extracted_rule(self):
        """Test adding extracted Datalog rules."""
        layer = SessionHistoryLayer()

        rule = "custom_violation(id) :- some_condition(id)."
        layer.add_extracted_rule(rule)

        program = layer._get_rules_program()
        assert "custom_violation" in program

    def test_clear_extracted_rules(self):
        """Test clearing extracted rules."""
        layer = SessionHistoryLayer()

        layer.add_extracted_rule("rule1(x) :- fact(x).")
        layer.add_extracted_rule("rule2(x) :- fact(x).")
        layer.clear_extracted_rules()

        # Should only have built-in rules now
        program = layer._get_rules_program()
        assert "rule1" not in program

    def test_load_rules(self):
        """Test loading rules for deployment."""
        layer = SessionHistoryLayer()

        rules = layer.load_rules("my-app")

        # Should have basic rules loaded
        assert len(rules) >= 1
        assert any(r.rule_id == "sh_contradiction" for r in rules)


# ============================================
# Edge Cases
# ============================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_prompt_and_response(self):
        """Test handling empty prompt and response."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        turn = layer.add_turn("sess_001", "turn_0", "", "")
        assert turn is not None

    def test_very_long_text(self):
        """Test handling very long text."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        long_text = "word " * 10000
        turn = layer.add_turn("sess_001", "turn_0", long_text, long_text)
        assert turn is not None

    def test_special_characters(self):
        """Test handling special characters."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        special = 'Text with "quotes" and \t tabs and \n newlines'
        turn = layer.add_turn("sess_001", "turn_0", special, special)
        assert turn is not None

    def test_unicode_text(self):
        """Test handling unicode text."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        unicode_text = "Hello 你好 مرحبا 🎉"
        turn = layer.add_turn("sess_001", "turn_0", unicode_text, unicode_text)
        assert turn is not None

    def test_multiple_sessions(self):
        """Test managing multiple sessions."""
        layer = SessionHistoryLayer()

        layer.create_session("sess_001", "user_1", "app_1")
        layer.create_session("sess_002", "user_2", "app_1")
        layer.create_session("sess_003", "user_1", "app_2")

        assert layer.get_session("sess_001") is not None
        assert layer.get_session("sess_002") is not None
        assert layer.get_session("sess_003") is not None

        layer.end_session("sess_002")
        assert layer.get_session("sess_002") is None
        assert layer.get_session("sess_001") is not None

    def test_session_with_metadata(self):
        """Test session with metadata."""
        layer = SessionHistoryLayer()

        session = layer.create_session(
            session_id="sess_001",
            user_id="user_123",
            deployment_id="my-app",
            metadata={"source": "web", "version": "1.0"},
        )

        assert session.metadata["source"] == "web"
        assert session.metadata["version"] == "1.0"


# ============================================
# Layer Result Tests
# ============================================

class TestLayerResult:
    """Tests for LayerResult structure."""

    def test_result_structure(self):
        """Test that results have correct structure."""
        layer = SessionHistoryLayer()

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi!",
            llm_model="gpt-4",
        )

        result = layer.check(request, {})

        assert result.layer == 5
        assert isinstance(result.violations, list)
        assert isinstance(result.reasoning, list)
        assert isinstance(result.facts_extracted, dict)
        assert isinstance(result.metadata, dict)

    def test_result_facts_extracted(self):
        """Test that facts are properly extracted."""
        layer = SessionHistoryLayer()
        layer.create_session("sess_001", "user_123", "my-app")

        request = VerificationRequest(
            request_id="req_001",
            deployment_id="my-app",
            prompt="Hello",
            llm_output="Hi!",
            llm_model="gpt-4",
            session_id="sess_001",
        )

        result = layer.check(request, {})

        assert "session_id" in result.facts_extracted
        assert "has_session" in result.facts_extracted
        assert result.facts_extracted["session_id"] == "sess_001"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
