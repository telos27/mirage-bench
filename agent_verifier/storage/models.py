"""SQLAlchemy models for persistent storage."""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class PolicyModel(Base):
    """SQLAlchemy model for business policies (Layer 3)."""

    __tablename__ = "business_policies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(255), unique=True, nullable=False, index=True)
    deployment_id = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    policy_type = Column(String(50), nullable=False)  # constraint, requirement, prohibition
    policy_spec = Column(Text, nullable=False)  # JSON-encoded PolicySpec
    priority = Column(Integer, default=0)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "deployment_id": self.deployment_id,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type,
            "policy_spec": json.loads(self.policy_spec) if self.policy_spec else {},
            "priority": self.priority,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_policy_spec(cls, spec: "PolicySpec") -> "PolicyModel":
        """Create from PolicySpec schema."""
        from ..schemas.rules import PolicySpec
        return cls(
            policy_id=spec.policy_id,
            deployment_id=spec.deployment_id,
            name=spec.name,
            description=spec.description,
            policy_type=spec.policy_type.value,
            policy_spec=json.dumps(spec.to_dict()),
            priority=spec.priority,
            enabled=spec.enabled,
        )


class RuleModel(Base):
    """SQLAlchemy model for rules (used by multiple layers)."""

    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(String(255), unique=True, nullable=False, index=True)
    deployment_id = Column(String(255), nullable=False, index=True)
    layer = Column(Integer, nullable=False, index=True)  # 1-6
    name = Column(String(255), nullable=False)
    description = Column(Text)
    rule_type = Column(String(50), nullable=False)
    rule_spec = Column(Text, nullable=False)  # JSON-encoded Rule
    severity = Column(String(20), default="error")
    enabled = Column(Boolean, default=True)
    tags = Column(Text)  # JSON-encoded list
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "deployment_id": self.deployment_id,
            "layer": self.layer,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type,
            "rule_spec": json.loads(self.rule_spec) if self.rule_spec else {},
            "severity": self.severity,
            "enabled": self.enabled,
            "tags": json.loads(self.tags) if self.tags else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_rule(cls, rule: "Rule", deployment_id: str) -> "RuleModel":
        """Create from Rule schema."""
        from ..schemas.rules import Rule
        return cls(
            rule_id=rule.rule_id,
            deployment_id=deployment_id,
            layer=rule.layer,
            name=rule.name,
            description=rule.description,
            rule_type=rule.rule_type.value,
            rule_spec=json.dumps(rule.to_dict()),
            severity=rule.severity.value,
            enabled=rule.enabled,
            tags=json.dumps(rule.tags),
        )


class UserPreferenceModel(Base):
    """SQLAlchemy model for user preferences (Layer 4)."""

    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    deployment_id = Column(String(255), nullable=False, index=True)
    preference_key = Column(String(255), nullable=False)
    preference_value = Column(Text, nullable=False)  # JSON-encoded value
    preference_type = Column(String(50), default="explicit")  # explicit, inferred
    source = Column(String(255))  # Where this preference came from
    confidence = Column(Integer, default=100)  # 0-100, for inferred preferences
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite unique constraint
    __table_args__ = (
        # Unique constraint on user_id, deployment_id, preference_key
        # Note: Using index=True on individual columns for query performance
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "deployment_id": self.deployment_id,
            "preference_key": self.preference_key,
            "preference_value": json.loads(self.preference_value) if self.preference_value else None,
            "preference_type": self.preference_type,
            "source": self.source,
            "confidence": self.confidence,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def create_database(db_path: str = "verifier.db") -> tuple:
    """
    Create database engine and session factory.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Tuple of (engine, SessionLocal factory)
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return engine, SessionLocal
