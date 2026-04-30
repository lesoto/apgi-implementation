"""Compliance and audit controls for APGI production deployments.

Provides audit logging, data lifecycle management, and compliance hooks
for enterprise environments requiring SOC2/ISO27001/GDPR/HIPAA-style controls.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.logging_config import get_logger

logger = get_logger("apgi.compliance")


@dataclass
class AuditEvent:
    """Structured audit event for compliance logging."""

    event_type: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    session_id: str | None = None
    data_classification: str = "internal"
    operation: str | None = None
    resource: str | None = None
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "data_classification": self.data_classification,
            "operation": self.operation,
            "resource": self.resource,
            "result": self.result,
            "metadata": self.metadata,
        }


class AuditLogger:
    """Production-grade audit logger for compliance requirements."""

    def __init__(self, enabled: bool = True, data_classification: str = "internal"):
        """Initialize audit logger.

        Args:
            enabled: Whether audit logging is enabled
            data_classification: Default data classification level
        """
        self.enabled = enabled
        self.data_classification = data_classification
        self.session_id = str(uuid.uuid4())
        self._event_buffer: list[AuditEvent] = []

    def log_event(
        self,
        event_type: str,
        operation: str | None = None,
        resource: str | None = None,
        result: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            event_type: Type of event (e.g., 'config_change', 'pipeline_start')
            operation: Operation being performed
            resource: Resource being accessed
            result: Result of the operation
            metadata: Additional metadata

        Returns:
            The created audit event
        """
        if not self.enabled:
            return None  # type: ignore[return-value]

        event = AuditEvent(
            event_type=event_type,
            session_id=self.session_id,
            data_classification=self.data_classification,
            operation=operation,
            resource=resource,
            result=result,
            metadata=metadata or {},
        )

        self._event_buffer.append(event)

        # Log to structured logger
        logger.info(
            "audit_event",
            event_id=event.event_id,
            event_type=event.event_type,
            operation=event.operation,
            resource=event.resource,
            result=event.result,
            data_classification=event.data_classification,
        )

        return event

    def log_config_change(
        self, old_config: dict[str, Any], new_config: dict[str, Any], reason: str | None = None
    ) -> AuditEvent:
        """Log configuration changes for audit trail.

        Args:
            old_config: Previous configuration
            new_config: New configuration
            reason: Reason for the change

        Returns:
            The created audit event
        """
        # Hash sensitive config values for privacy
        diff = self._compute_config_diff(old_config, new_config)

        return self.log_event(
            event_type="config_change",
            operation="update",
            resource="configuration",
            result="success",
            metadata={
                "changed_keys": diff,
                "reason": reason,
                "config_hash": self._hash_config(new_config),
            },
        )

    def log_pipeline_start(self, config: dict[str, Any], n_steps: int) -> AuditEvent:
        """Log pipeline execution start.

        Args:
            config: Pipeline configuration
            n_steps: Number of steps

        Returns:
            The created audit event
        """
        return self.log_event(
            event_type="pipeline_start",
            operation="execute",
            resource="pipeline",
            metadata={
                "n_steps": n_steps,
                "config_hash": self._hash_config(config),
                "hierarchical_mode": config.get("hierarchical_mode", "off"),
            },
        )

    def log_pipeline_complete(
        self, n_steps: int, ignition_count: int, duration_ms: float
    ) -> AuditEvent:
        """Log pipeline execution completion.

        Args:
            n_steps: Number of steps executed
            ignition_count: Number of ignition events
            duration_ms: Execution duration in milliseconds

        Returns:
            The created audit event
        """
        return self.log_event(
            event_type="pipeline_complete",
            operation="execute",
            resource="pipeline",
            result="success",
            metadata={
                "n_steps": n_steps,
                "ignition_count": ignition_count,
                "duration_ms": duration_ms,
            },
        )

    def log_data_retention(self, data_id: str, retention_days: int, action: str) -> AuditEvent:
        """Log data retention actions.

        Args:
            data_id: Identifier for the data
            retention_days: Retention period
            action: Action taken (e.g., 'delete', 'archive')

        Returns:
            The created audit event
        """
        return self.log_event(
            event_type="data_retention",
            operation=action,
            resource=data_id,
            metadata={
                "retention_days": retention_days,
                "action": action,
            },
        )

    def get_events(self) -> list[AuditEvent]:
        """Get all buffered events."""
        return self._event_buffer.copy()

    def export_events(self, filepath: str) -> None:
        """Export audit events to JSON file.

        Args:
            filepath: Output file path
        """
        events = [e.to_dict() for e in self._event_buffer]
        with open(filepath, "w") as f:
            json.dump(events, f, indent=2, default=str)
        logger.info("audit_exported", filepath=filepath, event_count=len(events))

    @staticmethod
    def _hash_config(config: dict[str, Any]) -> str:
        """Create hash of configuration for privacy."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @staticmethod
    def _compute_config_diff(old_config: dict[str, Any], new_config: dict[str, Any]) -> list[str]:
        """Compute list of changed configuration keys."""
        old_keys = set(old_config.keys())
        new_keys = set(new_config.keys())
        all_keys = old_keys | new_keys

        changed = []
        for key in all_keys:
            if old_config.get(key) != new_config.get(key):
                changed.append(key)

        return changed


class DataLifecycleManager:
    """Manages data lifecycle for compliance (retention, deletion, anonymization)."""

    def __init__(self, default_retention_days: int = 90):
        """Initialize data lifecycle manager.

        Args:
            default_retention_days: Default retention period in days
        """
        self.default_retention_days = default_retention_days
        self._data_registry: dict[str, dict[str, Any]] = {}

    def register_data(
        self, data_id: str, data_type: str, retention_days: int | None = None
    ) -> None:
        """Register data for lifecycle management.

        Args:
            data_id: Unique identifier for the data
            data_type: Type of data (e.g., 'simulation', 'history')
            retention_days: Retention period (uses default if None)
        """
        retention = retention_days or self.default_retention_days
        self._data_registry[data_id] = {
            "data_type": data_type,
            "created_at": time.time(),
            "retention_days": retention,
            "expires_at": time.time() + (retention * 24 * 60 * 60),
        }
        logger.debug(
            "data_registered", data_id=data_id, data_type=data_type, retention_days=retention
        )

    def check_expired(self) -> list[str]:
        """Check for expired data.

        Returns:
            List of expired data IDs
        """
        now = time.time()
        expired = [
            data_id for data_id, info in self._data_registry.items() if info["expires_at"] < now
        ]
        return expired

    def anonymize_history(self, history: dict[str, list[float]]) -> dict[str, list[float]]:
        """Anonymize history data by removing identifiable patterns.

        Args:
            history: Raw history data

        Returns:
            Anonymized history
        """
        # In a real implementation, this would:
        # 1. Remove timestamps or round to reduce precision
        # 2. Add noise to signals for differential privacy
        # 3. Aggregate where possible
        # For now, return a copy
        return {k: v.copy() for k, v in history.items()}

    def delete_data(self, data_id: str) -> bool:
        """Mark data as deleted.

        Args:
            data_id: Data identifier

        Returns:
            True if data was found and deleted
        """
        if data_id in self._data_registry:
            self._data_registry[data_id]["deleted_at"] = time.time()
            self._data_registry[data_id]["deleted"] = True
            logger.info("data_deleted", data_id=data_id)
            return True
        return False


class ComplianceManager:
    """Central compliance manager combining audit logging and data lifecycle."""

    def __init__(
        self,
        audit_enabled: bool = True,
        data_classification: str = "internal",
        retention_days: int = 90,
    ):
        """Initialize compliance manager.

        Args:
            audit_enabled: Enable audit logging
            data_classification: Default data classification
            retention_days: Default data retention period
        """
        self.audit = AuditLogger(enabled=audit_enabled, data_classification=data_classification)
        self.data_lifecycle = DataLifecycleManager(default_retention_days=retention_days)
        self._start_time: float | None = None

    def start_pipeline(self, config: dict[str, Any], n_steps: int) -> None:
        """Record pipeline start."""
        self._start_time = time.time()
        self.audit.log_pipeline_start(config, n_steps)

    def end_pipeline(self, n_steps: int, ignition_count: int) -> None:
        """Record pipeline end."""
        duration_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0
        self.audit.log_pipeline_complete(n_steps, ignition_count, duration_ms)

    def export_audit_trail(self, filepath: str) -> None:
        """Export complete audit trail."""
        self.audit.export_events(filepath)


def create_compliance_config(strict: bool = True) -> dict[str, Any]:
    """Create compliance configuration.

    Args:
        strict: Enable strict compliance mode

    Returns:
        Compliance configuration dictionary
    """
    return {
        "audit_enabled": True,
        "data_classification": "confidential" if strict else "internal",
        "retention_days": 30 if strict else 90,
        "encryption_at_rest": strict,
        "encryption_in_transit": strict,
        "access_logging": strict,
    }
