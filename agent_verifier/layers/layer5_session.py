"""Layer 5: Session History - Multi-turn context and consistency tracking."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .base_layer import BaseLayer, LayerResult
from ..schemas.request import VerificationRequest
from ..schemas.result import Severity
from ..schemas.rules import Rule, RuleType
from ..schemas.session import Session, Turn, EstablishedFact
from ..reasoning.datalog_engine import DatalogEngine, DatalogResult


@dataclass
class SessionState:
    """
    Runtime state for a session.

    Tracks the current state of a multi-turn conversation including
    established facts, commitments, and progress.
    """
    session: Session
    claims: dict[str, tuple[str, int]] = field(default_factory=dict)  # claim_type -> (value, turn)
    questions_asked: dict[str, int] = field(default_factory=dict)  # question -> turn
    questions_answered: dict[str, tuple[str, int]] = field(default_factory=dict)  # question -> (answer, turn)
    commitments: list[tuple[str, int]] = field(default_factory=list)  # (commitment, turn)
    progress: dict[str, str] = field(default_factory=dict)  # step -> status
    current_task: str | None = None


class SessionHistoryLayer(BaseLayer):
    """
    Layer 5: Session History.

    Checks multi-turn consistency across a conversation:
    - Contradiction detection (agent contradicts previous statements)
    - Established fact tracking (agent remembers session context)
    - Progress tracking (agent doesn't regress on completed work)
    - Question/answer coherence (agent doesn't re-ask answered questions)
    - Commitment tracking (agent follows through on promises)

    Sessions can be:
    1. In-memory (managed by this layer)
    2. Loaded from storage
    3. Passed in context

    All session checking is done via Souffl Datalog.
    """

    BUILTIN_RULES = Path(__file__).parent.parent / "reasoning" / "rules" / "session_history.dl"

    def __init__(
        self,
        storage: Any = None,
        max_session_turns: int = 100,
        fact_confidence_threshold: float = 0.7,
    ):
        """
        Initialize Layer 5.

        Args:
            storage: Optional SQLiteStore for session persistence
            max_session_turns: Maximum turns to track per session
            fact_confidence_threshold: Minimum confidence for established facts
        """
        super().__init__(layer_number=5, layer_name="Session History")

        self.datalog = DatalogEngine()
        self.storage = storage
        self.max_session_turns = max_session_turns
        self.fact_confidence_threshold = fact_confidence_threshold

        # In-memory session storage (session_id -> SessionState)
        self._sessions: dict[str, SessionState] = {}

        # In-memory rule storage
        self._dynamic_rules: list[Rule] = []

        # Additional Datalog rules
        self._extracted_rules: list[str] = []

        self._load_basic_rules()

    def _load_basic_rules(self) -> None:
        """Load basic rule metadata for session history."""
        basic_rules = [
            Rule(
                rule_id="sh_contradiction",
                name="Contradiction Detection",
                description="Agent should not contradict previous statements",
                rule_type=RuleType.CONSTRAINT,
                layer=5,
                conditions=[],
                severity=Severity.ERROR,
                message_template="Contradiction: {detail}",
                tags=["session", "consistency"],
            ),
            Rule(
                rule_id="sh_forgotten_fact",
                name="Forgotten Fact",
                description="Agent should remember established facts",
                rule_type=RuleType.CONSTRAINT,
                layer=5,
                conditions=[],
                severity=Severity.WARNING,
                message_template="Forgotten fact: {detail}",
                tags=["session", "memory"],
            ),
            Rule(
                rule_id="sh_reasking",
                name="Re-asking Answered Question",
                description="Agent should not re-ask questions already answered",
                rule_type=RuleType.CONSTRAINT,
                layer=5,
                conditions=[],
                severity=Severity.WARNING,
                message_template="Re-asking: {detail}",
                tags=["session", "coherence"],
            ),
            Rule(
                rule_id="sh_progress_regression",
                name="Progress Regression",
                description="Agent should not lose track of progress",
                rule_type=RuleType.CONSTRAINT,
                layer=5,
                conditions=[],
                severity=Severity.ERROR,
                message_template="Progress regression: {detail}",
                tags=["session", "progress"],
            ),
        ]
        self._dynamic_rules.extend(basic_rules)

    # ========================================
    # Session Management
    # ========================================

    def create_session(
        self,
        session_id: str,
        user_id: str,
        deployment_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """
        Create a new session.

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            deployment_id: Deployment identifier
            metadata: Optional session metadata

        Returns:
            Created Session object
        """
        session = Session(
            session_id=session_id,
            user_id=user_id,
            deployment_id=deployment_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = SessionState(session=session)
        return session

    def get_session(self, session_id: str) -> Session | None:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session or None if not found
        """
        state = self._sessions.get(session_id)
        return state.session if state else None

    def get_session_state(self, session_id: str) -> SessionState | None:
        """
        Get the full session state.

        Args:
            session_id: Session identifier

        Returns:
            SessionState or None if not found
        """
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> None:
        """
        End and remove a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        turn_id: str,
        prompt: str,
        response: str,
        extracted_facts: dict[str, Any] | None = None,
    ) -> Turn | None:
        """
        Add a turn to a session.

        Args:
            session_id: Session identifier
            turn_id: Unique turn identifier
            prompt: User prompt
            response: Agent response
            extracted_facts: Optional extracted facts from this turn

        Returns:
            Created Turn or None if session not found
        """
        state = self._sessions.get(session_id)
        if not state:
            return None

        turn = Turn(
            turn_id=turn_id,
            prompt=prompt,
            response=response,
            extracted_facts=extracted_facts,
        )
        state.session.add_turn(turn)

        # Enforce max turns
        if len(state.session.turns) > self.max_session_turns:
            state.session.turns = state.session.turns[-self.max_session_turns:]

        return turn

    def establish_fact(
        self,
        session_id: str,
        fact_id: str,
        fact_type: str,
        key: str,
        value: Any,
        source_turn: str,
        confidence: float = 1.0,
    ) -> EstablishedFact | None:
        """
        Establish a fact in a session.

        Args:
            session_id: Session identifier
            fact_id: Unique fact identifier
            fact_type: Type of fact
            key: Fact key
            value: Fact value
            source_turn: Turn that established this fact
            confidence: Confidence level (0-1)

        Returns:
            Created EstablishedFact or None if session not found
        """
        state = self._sessions.get(session_id)
        if not state:
            return None

        fact = EstablishedFact(
            fact_id=fact_id,
            fact_type=fact_type,
            key=key,
            value=value,
            source_turn=source_turn,
            confidence=confidence,
        )
        state.session.add_fact(fact)
        return fact

    def record_claim(
        self,
        session_id: str,
        claim_type: str,
        claim_value: str,
        turn_num: int,
    ) -> None:
        """
        Record a claim made by the agent.

        Args:
            session_id: Session identifier
            claim_type: Type of claim
            claim_value: Value of the claim
            turn_num: Turn number when claim was made
        """
        state = self._sessions.get(session_id)
        if state:
            state.claims[claim_type] = (claim_value, turn_num)

    def record_question(
        self,
        session_id: str,
        question: str,
        turn_num: int,
    ) -> None:
        """
        Record a question asked by the agent.

        Args:
            session_id: Session identifier
            question: The question (normalized)
            turn_num: Turn number when question was asked
        """
        state = self._sessions.get(session_id)
        if state:
            state.questions_asked[question] = turn_num

    def record_answer(
        self,
        session_id: str,
        question: str,
        answer: str,
        turn_num: int,
    ) -> None:
        """
        Record an answer to a question.

        Args:
            session_id: Session identifier
            question: The question (normalized)
            answer: The answer provided
            turn_num: Turn number when answer was given
        """
        state = self._sessions.get(session_id)
        if state:
            state.questions_answered[question] = (answer, turn_num)

    def record_commitment(
        self,
        session_id: str,
        commitment: str,
        turn_num: int,
    ) -> None:
        """
        Record a commitment made by the agent.

        Args:
            session_id: Session identifier
            commitment: What the agent committed to
            turn_num: Turn number when commitment was made
        """
        state = self._sessions.get(session_id)
        if state:
            state.commitments.append((commitment, turn_num))

    def update_progress(
        self,
        session_id: str,
        step: str,
        status: str,
    ) -> None:
        """
        Update progress on a step.

        Args:
            session_id: Session identifier
            step: Step/task identifier
            status: New status (pending, in_progress, completed)
        """
        state = self._sessions.get(session_id)
        if state:
            state.progress[step] = status

    def set_current_task(
        self,
        session_id: str,
        task: str | None,
    ) -> None:
        """
        Set the current task being worked on.

        Args:
            session_id: Session identifier
            task: Task identifier or None
        """
        state = self._sessions.get(session_id)
        if state:
            state.current_task = task

    # ========================================
    # Fact Extraction
    # ========================================

    def _extract_claims(self, text: str) -> list[tuple[str, str]]:
        """Extract claims from text."""
        claims = []

        # Pattern: "X is Y" or "X are Y"
        is_pattern = re.findall(r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:is|are)\s+([^.!?]+)", text.lower())
        for subject, value in is_pattern:
            claims.append((subject.strip(), value.strip()))

        # Pattern: "I will/can/should X"
        will_pattern = re.findall(r"i\s+(?:will|can|should|must)\s+([^.!?]+)", text.lower())
        for action in will_pattern:
            claims.append(("agent_capability", action.strip()))

        return claims

    def _extract_questions(self, text: str) -> list[str]:
        """Extract questions from text."""
        questions = []

        # Direct questions ending with ?
        q_pattern = re.findall(r"([^.!?]*\?)", text)
        for q in q_pattern:
            normalized = q.strip().lower()
            if len(normalized) > 5:  # Filter out very short questions
                questions.append(normalized)

        # "What/where/how/when/who/why" patterns
        wh_pattern = re.findall(
            r"(?:what|where|how|when|who|why|which|could you|can you|would you)[^.!?]*[.!?]",
            text.lower()
        )
        for q in wh_pattern:
            normalized = q.strip()
            if normalized and normalized not in questions:
                questions.append(normalized)

        return questions

    def _extract_commitments(self, text: str) -> list[str]:
        """Extract commitments/promises from text."""
        commitments = []

        # "I will X" patterns
        will_pattern = re.findall(r"i\s+will\s+([^.!?,]+)", text.lower())
        for action in will_pattern:
            commitments.append(action.strip())

        # "Let me X" patterns
        let_pattern = re.findall(r"let\s+me\s+([^.!?,]+)", text.lower())
        for action in let_pattern:
            commitments.append(action.strip())

        return commitments

    def _extract_task_references(self, text: str, current_task: str | None) -> list[tuple[str, str]]:
        """Extract task-related actions from text."""
        actions = []

        # Common task action patterns
        action_patterns = [
            (r"(?:completed?|finished?|done with)\s+([^.!?,]+)", "completed"),
            (r"(?:working on|starting|beginning)\s+([^.!?,]+)", "in_progress"),
            (r"(?:will|going to|need to)\s+([^.!?,]+)", "pending"),
        ]

        for pattern, status in action_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                actions.append((match.strip(), status))

        return actions

    def _check_context_acknowledgment(self, text: str, session_state: SessionState) -> bool:
        """Check if text acknowledges session context."""
        text_lower = text.lower()

        # Check for references to previous turns
        if any(word in text_lower for word in ["earlier", "before", "previously", "as i mentioned", "as we discussed"]):
            return True

        # Check for references to established facts
        for fact in session_state.session.established_facts:
            if str(fact.value).lower() in text_lower:
                return True

        return False

    # ========================================
    # Datalog Integration
    # ========================================

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
        """Combine all rules into a single Datalog program."""
        program = ""

        if self.BUILTIN_RULES.exists():
            with open(self.BUILTIN_RULES) as f:
                program = f.read()

        if self._extracted_rules:
            program += "\n\n// Extracted Rules\n"
            for rule in self._extracted_rules:
                program += rule + "\n"

        return program

    def _populate_datalog_facts(
        self,
        case_id: str,
        request: VerificationRequest,
        context: dict[str, Any],
        session_state: SessionState | None,
    ) -> None:
        """Populate the Datalog engine with facts for session checking."""
        self.datalog.clear_facts()

        # Basic request info
        self.datalog.add_fact("request_id", case_id)

        session_id = request.session_id
        user_id = request.user_id or "anonymous"

        if session_id:
            self.datalog.add_fact("session_id", case_id, session_id)
        self.datalog.add_fact("user_id", case_id, user_id)

        if not session_state:
            # No session context, just add current turn as turn 0
            self.datalog.add_fact("current_turn", case_id, 0)
            return

        session = session_state.session
        current_turn = len(session.turns)
        self.datalog.add_fact("current_turn", case_id, current_turn)

        # Previous turn facts
        for i, turn in enumerate(session.turns):
            # Previous statements
            self.datalog.add_fact("previous_statement", case_id, i, "user", turn.prompt[:200])
            self.datalog.add_fact("previous_statement", case_id, i, "agent", turn.response[:200])

        # Previous claims
        for claim_type, (claim_value, turn_num) in session_state.claims.items():
            self.datalog.add_fact("previous_claim", case_id, turn_num, claim_type, claim_value)

        # Previous questions
        for question, turn_num in session_state.questions_asked.items():
            self.datalog.add_fact("previous_question", case_id, turn_num, question[:100])

        # Previous answers
        for question, (answer, turn_num) in session_state.questions_answered.items():
            self.datalog.add_fact("previous_answer", case_id, turn_num, question[:100], answer[:100])

        # Previous commitments
        for commitment, turn_num in session_state.commitments:
            self.datalog.add_fact("previous_commitment", case_id, turn_num, commitment[:100])

        # Previous progress
        for step, status in session_state.progress.items():
            # Find the turn where this progress was recorded (simplified: use current-1)
            self.datalog.add_fact("previous_progress", case_id, max(0, current_turn - 1), step, status)

        # Established facts
        for fact in session.established_facts:
            source = fact.fact_type
            confidence = int(fact.confidence * 100)
            self.datalog.add_fact(
                "established_fact",
                case_id,
                fact.key,
                str(fact.value)[:100],
                source,
                confidence
            )

        # Current task
        if session_state.current_task:
            self.datalog.add_fact("current_task", case_id, session_state.current_task)

        # Current progress
        for step, status in session_state.progress.items():
            self.datalog.add_fact("current_progress", case_id, step, status)

        # Extract facts from current output
        output = request.llm_output

        # Output claims
        claims = self._extract_claims(output)
        for claim_type, claim_value in claims:
            self.datalog.add_fact("output_claim", case_id, claim_type[:50], claim_value[:100])

        # Output questions
        questions = self._extract_questions(output)
        for question in questions:
            self.datalog.add_fact("output_question", case_id, question[:100])

        # Output commitments
        commitments = self._extract_commitments(output)
        for commitment in commitments:
            self.datalog.add_fact("output_commitment", case_id, commitment[:100])

        # Task actions
        task_actions = self._extract_task_references(output, session_state.current_task)
        for task, action in task_actions:
            self.datalog.add_fact("output_task_action", case_id, task[:50], action)

        # Context acknowledgment
        if self._check_context_acknowledgment(output, session_state):
            self.datalog.add_fact("acknowledges_context", case_id)

        # Check for corrections
        correction_patterns = [
            r"(?:actually|correction|i was wrong|let me correct)",
            r"(?:sorry|apologize).*(?:earlier|before|previous)",
        ]
        for pattern in correction_patterns:
            if re.search(pattern, output.lower()):
                # Generic correction acknowledgment
                self.datalog.add_fact("corrects_previous", case_id, current_turn - 1, "general")
                break

    def _parse_datalog_violations(
        self,
        result: DatalogResult,
        case_id: str,
    ) -> list[tuple[str, str, str]]:
        """
        Parse violations from Datalog output.

        Returns:
            List of (violation_type, detail, severity) tuples
        """
        violations = []
        for row in result.get_relation("output_session_violation"):
            if len(row) >= 4 and row[0] == case_id:
                violations.append((row[1], row[2], row[3]))  # type, detail, severity
        return violations

    # ========================================
    # Verification
    # ========================================

    def check(
        self,
        request: VerificationRequest,
        context: dict[str, Any],
    ) -> LayerResult:
        """
        Check session history for consistency violations.

        Args:
            request: The verification request
            context: Accumulated context from previous layers

        Returns:
            LayerResult with violations and reasoning
        """
        result = LayerResult(layer=self.layer_number)
        case_id = request.request_id
        session_id = request.session_id

        # Step 1: Get or create session state
        session_state = None
        if session_id:
            session_state = self.get_session_state(session_id)

            # Try loading from context if provided
            if not session_state and "session" in context:
                session_data = context["session"]
                if isinstance(session_data, Session):
                    session_state = SessionState(session=session_data)
                elif isinstance(session_data, dict):
                    session_state = SessionState(session=Session.from_dict(session_data))

        result.add_reasoning(self.create_reasoning_step(
            step_type="session_lookup",
            description="Retrieved session state",
            inputs={"session_id": session_id},
            outputs={
                "has_session": session_state is not None,
                "turn_count": len(session_state.session.turns) if session_state else 0,
                "fact_count": len(session_state.session.established_facts) if session_state else 0,
            },
        ))

        if not session_id:
            result.add_reasoning(self.create_reasoning_step(
                step_type="skip",
                description="No session context (single-turn request)",
                inputs={"session_id": None},
                outputs={"skipped": True},
            ))
            result.metadata["session_coherent"] = True
            result.metadata["coherence_score"] = 2
            return result

        # Step 2: Populate facts
        self._populate_datalog_facts(case_id, request, context, session_state)

        # Step 3: Run Datalog rules
        program = self._get_rules_program()

        if program.strip():
            datalog_result = self.datalog.run_inline(
                program,
                output_relations=[
                    "output_session_coherence",
                    "output_session_violation",
                    "output_context_utilization",
                    "output_session_summary",
                    "output_violation_count",
                ]
            )

            if datalog_result.success:
                result.add_reasoning(self.create_reasoning_step(
                    step_type="rule_application",
                    description="Applied session history Datalog rules",
                    inputs={"program_lines": len(program.split("\n"))},
                    outputs={"success": True},
                ))

                # Parse violations
                dl_violations = self._parse_datalog_violations(datalog_result, case_id)

                for vtype, detail, severity in dl_violations:
                    result.add_violation(self.create_violation(
                        violation_type=vtype,
                        message=f"Session violation: {vtype} - {detail}",
                        evidence={
                            "detail": detail,
                            "source": "datalog",
                        },
                        severity=severity,
                        rule_id=f"sh_{vtype}",
                    ))

                # Store coherence info in metadata
                for row in datalog_result.get_relation("output_session_coherence"):
                    if len(row) >= 3 and row[0] == case_id:
                        result.metadata["session_coherent"] = row[1] == "true"
                        result.metadata["coherence_score"] = int(row[2])

                # Store context utilization
                for row in datalog_result.get_relation("output_context_utilization"):
                    if len(row) >= 2 and row[0] == case_id:
                        result.metadata["context_utilization"] = row[1]

                # Store summary
                for row in datalog_result.get_relation("output_session_summary"):
                    if len(row) >= 4 and row[0] == case_id:
                        result.metadata["turn_count"] = int(row[1])
                        result.metadata["violation_count"] = int(row[2])
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

        # Store session info for downstream layers
        result.facts_extracted = {
            "session_id": session_id,
            "has_session": session_state is not None,
            "session_coherent": result.metadata.get("session_coherent", True),
            "turn_count": len(session_state.session.turns) if session_state else 0,
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

        if self.storage:
            stored_rules = self.storage.get_rules_for_layer(
                layer=5,
                deployment_id=deployment_id,
                enabled_only=True,
            )
            for model in stored_rules:
                import json
                spec = json.loads(model.rule_spec)
                rules.append(Rule.from_dict(spec))

        return rules


# Convenience functions

def create_session_from_turns(
    session_id: str,
    user_id: str,
    deployment_id: str,
    turns: list[dict[str, str]],
) -> Session:
    """
    Create a session from a list of turn dictionaries.

    Args:
        session_id: Session identifier
        user_id: User identifier
        deployment_id: Deployment identifier
        turns: List of {"prompt": ..., "response": ...} dicts

    Returns:
        Session object with all turns added
    """
    session = Session(
        session_id=session_id,
        user_id=user_id,
        deployment_id=deployment_id,
    )

    for i, turn_data in enumerate(turns):
        turn = Turn(
            turn_id=f"{session_id}_turn_{i}",
            prompt=turn_data.get("prompt", ""),
            response=turn_data.get("response", ""),
        )
        session.add_turn(turn)

    return session
