"""SQLite storage operations for policies, rules, and preferences."""

import json
from typing import Any

from sqlalchemy.orm import Session as DBSession

from ..schemas.rules import PolicySpec, Rule
from .models import PolicyModel, RuleModel, UserPreferenceModel, create_database


class SQLiteStore:
    """
    SQLite-based storage for verification data.

    Handles persistence for:
    - Business policies (Layer 3)
    - Rules (Layers 1-6)
    - User preferences (Layer 4)
    """

    def __init__(self, db_path: str = "verifier.db"):
        """
        Initialize the store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine, self.SessionLocal = create_database(db_path)

    def _get_session(self) -> DBSession:
        """Get a new database session."""
        return self.SessionLocal()

    # --- Policy Operations (Layer 3) ---

    def add_policy(self, policy: PolicySpec) -> PolicyModel:
        """
        Add a business policy.

        Args:
            policy: The policy specification

        Returns:
            The created PolicyModel
        """
        with self._get_session() as session:
            model = PolicyModel.from_policy_spec(policy)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model

    def get_policy(self, policy_id: str) -> PolicyModel | None:
        """
        Get a policy by ID.

        Args:
            policy_id: The policy identifier

        Returns:
            PolicyModel or None if not found
        """
        with self._get_session() as session:
            return session.query(PolicyModel).filter(
                PolicyModel.policy_id == policy_id
            ).first()

    def get_policies_for_deployment(
        self,
        deployment_id: str,
        enabled_only: bool = True,
    ) -> list[PolicyModel]:
        """
        Get all policies for a deployment.

        Args:
            deployment_id: The deployment identifier
            enabled_only: Only return enabled policies

        Returns:
            List of PolicyModel objects
        """
        with self._get_session() as session:
            query = session.query(PolicyModel).filter(
                PolicyModel.deployment_id == deployment_id
            )
            if enabled_only:
                query = query.filter(PolicyModel.enabled == True)
            return query.order_by(PolicyModel.priority.desc()).all()

    def update_policy(self, policy_id: str, updates: dict[str, Any]) -> bool:
        """
        Update a policy.

        Args:
            policy_id: The policy identifier
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found
        """
        with self._get_session() as session:
            policy = session.query(PolicyModel).filter(
                PolicyModel.policy_id == policy_id
            ).first()
            if not policy:
                return False
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            session.commit()
            return True

    def delete_policy(self, policy_id: str) -> bool:
        """
        Delete a policy.

        Args:
            policy_id: The policy identifier

        Returns:
            True if deleted, False if not found
        """
        with self._get_session() as session:
            result = session.query(PolicyModel).filter(
                PolicyModel.policy_id == policy_id
            ).delete()
            session.commit()
            return result > 0

    # --- Rule Operations (All Layers) ---

    def add_rule(self, rule: Rule, deployment_id: str) -> RuleModel:
        """
        Add a rule.

        Args:
            rule: The rule specification
            deployment_id: The deployment this rule applies to

        Returns:
            The created RuleModel
        """
        with self._get_session() as session:
            model = RuleModel.from_rule(rule, deployment_id)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model

    def get_rule(self, rule_id: str) -> RuleModel | None:
        """
        Get a rule by ID.

        Args:
            rule_id: The rule identifier

        Returns:
            RuleModel or None if not found
        """
        with self._get_session() as session:
            return session.query(RuleModel).filter(
                RuleModel.rule_id == rule_id
            ).first()

    def get_rules_for_layer(
        self,
        layer: int,
        deployment_id: str,
        enabled_only: bool = True,
    ) -> list[RuleModel]:
        """
        Get all rules for a specific layer and deployment.

        Args:
            layer: The layer number (1-6)
            deployment_id: The deployment identifier
            enabled_only: Only return enabled rules

        Returns:
            List of RuleModel objects
        """
        with self._get_session() as session:
            query = session.query(RuleModel).filter(
                RuleModel.layer == layer,
                RuleModel.deployment_id == deployment_id,
            )
            if enabled_only:
                query = query.filter(RuleModel.enabled == True)
            return query.all()

    def get_rules_by_tag(
        self,
        tag: str,
        deployment_id: str,
        enabled_only: bool = True,
    ) -> list[RuleModel]:
        """
        Get all rules with a specific tag.

        Args:
            tag: The tag to filter by
            deployment_id: The deployment identifier
            enabled_only: Only return enabled rules

        Returns:
            List of RuleModel objects with the tag
        """
        with self._get_session() as session:
            query = session.query(RuleModel).filter(
                RuleModel.deployment_id == deployment_id,
                RuleModel.tags.contains(f'"{tag}"'),  # JSON contains
            )
            if enabled_only:
                query = query.filter(RuleModel.enabled == True)
            return query.all()

    def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            True if deleted, False if not found
        """
        with self._get_session() as session:
            result = session.query(RuleModel).filter(
                RuleModel.rule_id == rule_id
            ).delete()
            session.commit()
            return result > 0

    # --- User Preference Operations (Layer 4) ---

    def set_preference(
        self,
        user_id: str,
        deployment_id: str,
        key: str,
        value: Any,
        source: str | None = None,
        preference_type: str = "explicit",
        confidence: int = 100,
    ) -> UserPreferenceModel:
        """
        Set a user preference.

        Args:
            user_id: The user identifier
            deployment_id: The deployment identifier
            key: The preference key
            value: The preference value
            source: Where this preference came from
            preference_type: Type of preference (explicit, inferred)
            confidence: Confidence level (0-100)

        Returns:
            The created/updated UserPreferenceModel
        """
        with self._get_session() as session:
            # Check if preference exists
            pref = session.query(UserPreferenceModel).filter(
                UserPreferenceModel.user_id == user_id,
                UserPreferenceModel.deployment_id == deployment_id,
                UserPreferenceModel.preference_key == key,
            ).first()

            if pref:
                # Update existing
                pref.preference_value = json.dumps(value)
                pref.source = source
                pref.preference_type = preference_type
                pref.confidence = confidence
            else:
                # Create new
                pref = UserPreferenceModel(
                    user_id=user_id,
                    deployment_id=deployment_id,
                    preference_key=key,
                    preference_value=json.dumps(value),
                    source=source,
                    preference_type=preference_type,
                    confidence=confidence,
                )
                session.add(pref)

            session.commit()
            session.refresh(pref)
            return pref

    def get_preference(
        self,
        user_id: str,
        deployment_id: str,
        key: str,
    ) -> Any | None:
        """
        Get a user preference value.

        Args:
            user_id: The user identifier
            deployment_id: The deployment identifier
            key: The preference key

        Returns:
            The preference value or None if not found
        """
        with self._get_session() as session:
            pref = session.query(UserPreferenceModel).filter(
                UserPreferenceModel.user_id == user_id,
                UserPreferenceModel.deployment_id == deployment_id,
                UserPreferenceModel.preference_key == key,
                UserPreferenceModel.enabled == True,
            ).first()
            if pref:
                return json.loads(pref.preference_value)
            return None

    def get_all_preferences(
        self,
        user_id: str,
        deployment_id: str,
        enabled_only: bool = True,
    ) -> dict[str, Any]:
        """
        Get all preferences for a user.

        Args:
            user_id: The user identifier
            deployment_id: The deployment identifier
            enabled_only: Only return enabled preferences

        Returns:
            Dictionary of preference key -> value
        """
        with self._get_session() as session:
            query = session.query(UserPreferenceModel).filter(
                UserPreferenceModel.user_id == user_id,
                UserPreferenceModel.deployment_id == deployment_id,
            )
            if enabled_only:
                query = query.filter(UserPreferenceModel.enabled == True)

            prefs = query.all()
            return {
                p.preference_key: json.loads(p.preference_value)
                for p in prefs
            }

    def delete_preference(
        self,
        user_id: str,
        deployment_id: str,
        key: str,
    ) -> bool:
        """
        Delete a user preference.

        Args:
            user_id: The user identifier
            deployment_id: The deployment identifier
            key: The preference key

        Returns:
            True if deleted, False if not found
        """
        with self._get_session() as session:
            result = session.query(UserPreferenceModel).filter(
                UserPreferenceModel.user_id == user_id,
                UserPreferenceModel.deployment_id == deployment_id,
                UserPreferenceModel.preference_key == key,
            ).delete()
            session.commit()
            return result > 0
