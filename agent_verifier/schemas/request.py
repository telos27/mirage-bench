"""Verification request schema."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class VerificationRequest:
    """
    A request to verify an LLM agent's output.

    Attributes:
        request_id: Unique identifier for this verification request
        deployment_id: Identifier for the deployment/application context
        prompt: The input prompt given to the LLM
        llm_output: The LLM's response to verify
        llm_model: The model that generated the output (e.g., "gpt-4", "claude-3")
        session_id: Optional session identifier for multi-turn conversations
        user_id: Optional user identifier for preference lookup
        timestamp: When the request was made
        additional_context: Optional extra context for verification
    """
    request_id: str
    deployment_id: str
    prompt: str
    llm_output: str
    llm_model: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_context: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "deployment_id": self.deployment_id,
            "prompt": self.prompt,
            "llm_output": self.llm_output,
            "llm_model": self.llm_model,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "additional_context": self.additional_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationRequest":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
