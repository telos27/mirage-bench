"""Session data schemas."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Turn:
    """
    A single turn in a conversation session.

    Attributes:
        turn_id: Unique identifier for this turn
        prompt: The user/system prompt
        response: The LLM response
        timestamp: When this turn occurred
        verification_result: Optional verification result for this turn
        extracted_facts: Facts extracted from this turn
    """
    turn_id: str
    prompt: str
    response: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    verification_result: Optional[dict[str, Any]] = None
    extracted_facts: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "prompt": self.prompt,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
            "verification_result": self.verification_result,
            "extracted_facts": self.extracted_facts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class EstablishedFact:
    """
    A fact established during the session.

    Attributes:
        fact_id: Unique identifier
        fact_type: Type of fact (e.g., "user_preference", "stated_constraint", "agreed_format")
        key: The fact key/name
        value: The fact value
        source_turn: Which turn established this fact
        confidence: Confidence in this fact (0-1)
    """
    fact_id: str
    fact_type: str
    key: str
    value: Any
    source_turn: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "key": self.key,
            "value": self.value,
            "source_turn": self.source_turn,
            "confidence": self.confidence,
        }


@dataclass
class Session:
    """
    A conversation session with established context.

    Attributes:
        session_id: Unique identifier for this session
        user_id: The user this session belongs to
        deployment_id: The deployment context
        turns: List of conversation turns
        established_facts: Facts established during the session
        created_at: When the session was created
        last_active: Last activity timestamp
        metadata: Additional session metadata
    """
    session_id: str
    user_id: str
    deployment_id: str
    turns: list[Turn] = field(default_factory=list)
    established_facts: list[EstablishedFact] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "deployment_id": self.deployment_id,
            "turns": [t.to_dict() for t in self.turns],
            "established_facts": [f.to_dict() for f in self.established_facts],
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        data = data.copy()
        data["turns"] = [Turn.from_dict(t) for t in data.get("turns", [])]
        data["established_facts"] = [
            EstablishedFact(**f) for f in data.get("established_facts", [])
        ]
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("last_active"), str):
            data["last_active"] = datetime.fromisoformat(data["last_active"])
        return cls(**data)

    def add_turn(self, turn: Turn) -> None:
        """Add a turn to the session."""
        self.turns.append(turn)
        self.last_active = datetime.utcnow()

    def add_fact(self, fact: EstablishedFact) -> None:
        """Add an established fact to the session."""
        self.established_facts.append(fact)

    def get_facts_by_type(self, fact_type: str) -> list[EstablishedFact]:
        """Get all facts of a specific type."""
        return [f for f in self.established_facts if f.fact_type == fact_type]

    def get_fact(self, key: str) -> Optional[EstablishedFact]:
        """Get a fact by key (returns most recent if multiple)."""
        matching = [f for f in self.established_facts if f.key == key]
        return matching[-1] if matching else None
