"""Tests for core/compliance.py - Audit logging and compliance management."""

from __future__ import annotations

import json
import os
import tempfile
import time

from core.compliance import (
    AuditEvent,
    AuditLogger,
    ComplianceManager,
    DataLifecycleManager,
    create_compliance_config,
)


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_audit_event_creation_defaults(self):
        """Should create event with default values."""
        event = AuditEvent(event_type="test_event")
        assert event.event_type == "test_event"
        assert event.timestamp > 0
        assert len(event.event_id) == 36  # UUID format
        assert event.user_id is None
        assert event.session_id is None
        assert event.data_classification == "internal"
        assert event.operation is None
        assert event.resource is None
        assert event.result is None
        assert event.metadata == {}

    def test_audit_event_creation_custom(self):
        """Should create event with custom values."""
        event = AuditEvent(
            event_type="config_change",
            user_id="user123",
            session_id="session456",
            data_classification="confidential",
            operation="update",
            resource="config.yaml",
            result="success",
            metadata={"key": "value"},
        )
        assert event.event_type == "config_change"
        assert event.user_id == "user123"
        assert event.session_id == "session456"
        assert event.data_classification == "confidential"
        assert event.operation == "update"
        assert event.resource == "config.yaml"
        assert event.result == "success"
        assert event.metadata == {"key": "value"}

    def test_audit_event_to_dict(self):
        """Should convert to dictionary correctly."""
        event = AuditEvent(
            event_type="test",
            user_id="user1",
            metadata={"foo": "bar"},
        )
        d = event.to_dict()
        assert d["event_type"] == "test"
        assert d["user_id"] == "user1"
        assert d["metadata"] == {"foo": "bar"}
        assert "event_id" in d
        assert "timestamp" in d


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_logger_creation_defaults(self):
        """Should create logger with defaults."""
        logger = AuditLogger()
        assert logger.enabled is True
        assert logger.data_classification == "internal"
        assert len(logger.session_id) == 36
        assert logger._event_buffer == []

    def test_logger_creation_custom(self):
        """Should create logger with custom values."""
        logger = AuditLogger(enabled=False, data_classification="confidential")
        assert logger.enabled is False
        assert logger.data_classification == "confidential"

    def test_log_event_when_enabled(self):
        """Should log event when enabled."""
        logger = AuditLogger()
        event = logger.log_event(
            event_type="test",
            operation="read",
            resource="file.txt",
            result="success",
            metadata={"size": 100},
        )
        assert event is not None
        assert event.event_type == "test"
        assert event.operation == "read"
        assert event.resource == "file.txt"
        assert event.result == "success"
        assert event.metadata == {"size": 100}
        assert len(logger._event_buffer) == 1

    def test_log_event_when_disabled(self):
        """Should return None when disabled."""
        logger = AuditLogger(enabled=False)
        event = logger.log_event(event_type="test")
        assert event is None
        assert len(logger._event_buffer) == 0

    def test_log_config_change(self):
        """Should log configuration changes."""
        logger = AuditLogger()
        old_config = {"param1": 1, "param2": 2}
        new_config = {"param1": 1, "param2": 3, "param3": 4}

        event = logger.log_config_change(old_config, new_config, reason="update")

        assert event.event_type == "config_change"
        assert event.operation == "update"
        assert event.resource == "configuration"
        assert event.result == "success"
        assert "changed_keys" in event.metadata
        assert set(event.metadata["changed_keys"]) == {"param2", "param3"}
        assert "config_hash" in event.metadata
        assert "reason" in event.metadata

    def test_log_config_change_no_diff(self):
        """Should handle identical configs."""
        logger = AuditLogger()
        config = {"param": 1}
        event = logger.log_config_change(config, config)
        assert event.metadata["changed_keys"] == []

    def test_log_pipeline_start(self):
        """Should log pipeline start."""
        logger = AuditLogger()
        config = {"hierarchical_mode": "full", "steps": 100}
        event = logger.log_pipeline_start(config, n_steps=100)

        assert event.event_type == "pipeline_start"
        assert event.operation == "execute"
        assert event.resource == "pipeline"
        assert event.metadata["n_steps"] == 100
        assert event.metadata["hierarchical_mode"] == "full"
        assert "config_hash" in event.metadata

    def test_log_pipeline_complete(self):
        """Should log pipeline completion."""
        logger = AuditLogger()
        event = logger.log_pipeline_complete(n_steps=100, ignition_count=5, duration_ms=1234.5)

        assert event.event_type == "pipeline_complete"
        assert event.operation == "execute"
        assert event.resource == "pipeline"
        assert event.result == "success"
        assert event.metadata["n_steps"] == 100
        assert event.metadata["ignition_count"] == 5
        assert event.metadata["duration_ms"] == 1234.5

    def test_log_data_retention(self):
        """Should log data retention actions."""
        logger = AuditLogger()
        event = logger.log_data_retention("data_123", 30, "delete")

        assert event.event_type == "data_retention"
        assert event.operation == "delete"
        assert event.resource == "data_123"
        assert event.metadata["retention_days"] == 30
        assert event.metadata["action"] == "delete"

    def test_get_events(self):
        """Should return copy of events buffer."""
        logger = AuditLogger()
        logger.log_event(event_type="test1")
        logger.log_event(event_type="test2")

        events = logger.get_events()
        assert len(events) == 2
        assert events[0].event_type == "test1"
        assert events[1].event_type == "test2"

        # Modifying returned list should not affect buffer
        events.clear()
        assert len(logger.get_events()) == 2

    def test_export_events(self):
        """Should export events to JSON file."""
        logger = AuditLogger()
        logger.log_event(event_type="test", metadata={"key": "value"})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            logger.export_events(filepath)
            assert os.path.exists(filepath)

            with open(filepath) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["event_type"] == "test"
            assert data[0]["metadata"] == {"key": "value"}
        finally:
            os.unlink(filepath)

    def test_hash_config(self):
        """Should create consistent hash."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}  # Different order
        config3 = {"a": 1, "b": 3}  # Different value

        hash1 = AuditLogger._hash_config(config1)
        hash2 = AuditLogger._hash_config(config2)
        hash3 = AuditLogger._hash_config(config3)

        assert len(hash1) == 16
        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash

    def test_compute_config_diff(self):
        """Should compute config differences."""
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 1, "b": 3, "d": 4}  # b changed, c removed, d added

        diff = AuditLogger._compute_config_diff(old, new)
        assert set(diff) == {"b", "c", "d"}

    def test_compute_config_diff_no_changes(self):
        """Should return empty list for identical configs."""
        config = {"a": 1, "b": 2}
        diff = AuditLogger._compute_config_diff(config, config)
        assert diff == []


class TestDataLifecycleManager:
    """Tests for DataLifecycleManager class."""

    def test_manager_creation_defaults(self):
        """Should create manager with defaults."""
        manager = DataLifecycleManager()
        assert manager.default_retention_days == 90
        assert manager._data_registry == {}

    def test_manager_creation_custom(self):
        """Should create manager with custom retention."""
        manager = DataLifecycleManager(default_retention_days=30)
        assert manager.default_retention_days == 30

    def test_register_data_defaults(self):
        """Should register data with default retention."""
        manager = DataLifecycleManager()
        manager.register_data("data_1", "simulation")

        assert "data_1" in manager._data_registry
        info = manager._data_registry["data_1"]
        assert info["data_type"] == "simulation"
        assert info["retention_days"] == 90
        assert "created_at" in info
        assert "expires_at" in info
        assert info["expires_at"] > info["created_at"]

    def test_register_data_custom_retention(self):
        """Should register data with custom retention."""
        manager = DataLifecycleManager(default_retention_days=90)
        manager.register_data("data_1", "history", retention_days=30)

        info = manager._data_registry["data_1"]
        assert info["retention_days"] == 30

    def test_check_expired_empty(self):
        """Should return empty list when no data."""
        manager = DataLifecycleManager()
        assert manager.check_expired() == []

    def test_check_expired_with_data(self):
        """Should find expired data."""
        manager = DataLifecycleManager()
        # Register data that expires immediately
        manager.register_data("expired", "test", retention_days=-1)
        manager.register_data("valid", "test", retention_days=365)

        expired = manager.check_expired()
        assert "expired" in expired
        assert "valid" not in expired

    def test_anonymize_history(self):
        """Should anonymize history data."""
        manager = DataLifecycleManager()
        history = {"signal": [1.0, 2.0, 3.0], "threshold": [0.5, 0.6, 0.7]}

        result = manager.anonymize_history(history)

        assert result.keys() == history.keys()
        assert result["signal"] == history["signal"]
        assert result["signal"] is not history["signal"]  # Should be copy

    def test_delete_data_success(self):
        """Should mark data as deleted."""
        manager = DataLifecycleManager()
        manager.register_data("data_1", "test")

        result = manager.delete_data("data_1")

        assert result is True
        assert manager._data_registry["data_1"]["deleted"] is True
        assert "deleted_at" in manager._data_registry["data_1"]

    def test_delete_data_not_found(self):
        """Should return False for non-existent data."""
        manager = DataLifecycleManager()
        result = manager.delete_data("non_existent")
        assert result is False


class TestComplianceManager:
    """Tests for ComplianceManager class."""

    def test_manager_creation_defaults(self):
        """Should create manager with defaults."""
        manager = ComplianceManager()
        assert manager.audit.enabled is True
        assert manager.audit.data_classification == "internal"
        assert manager.data_lifecycle.default_retention_days == 90
        assert manager._start_time is None

    def test_manager_creation_custom(self):
        """Should create manager with custom settings."""
        manager = ComplianceManager(
            audit_enabled=False,
            data_classification="confidential",
            retention_days=30,
        )
        assert manager.audit.enabled is False
        assert manager.audit.data_classification == "confidential"
        assert manager.data_lifecycle.default_retention_days == 30

    def test_start_pipeline(self):
        """Should record pipeline start."""
        manager = ComplianceManager()
        config = {"hierarchical_mode": "full"}

        manager.start_pipeline(config, n_steps=100)

        assert manager._start_time is not None
        assert len(manager.audit.get_events()) == 1
        assert manager.audit.get_events()[0].event_type == "pipeline_start"

    def test_end_pipeline_with_start(self):
        """Should record pipeline end with duration."""
        manager = ComplianceManager()
        manager.start_pipeline({}, n_steps=100)

        # Small delay to ensure measurable duration
        time.sleep(0.01)
        manager.end_pipeline(n_steps=100, ignition_count=5)

        events = manager.audit.get_events()
        assert len(events) == 2
        assert events[1].event_type == "pipeline_complete"
        assert events[1].metadata["duration_ms"] > 0

    def test_end_pipeline_without_start(self):
        """Should handle end without start."""
        manager = ComplianceManager()
        manager.end_pipeline(n_steps=100, ignition_count=5)

        events = manager.audit.get_events()
        assert len(events) == 1
        assert events[0].metadata["duration_ms"] == 0

    def test_export_audit_trail(self):
        """Should export audit trail."""
        manager = ComplianceManager()
        manager.start_pipeline({}, n_steps=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            manager.export_audit_trail(filepath)
            assert os.path.exists(filepath)

            with open(filepath) as f:
                data = json.load(f)
            assert len(data) == 1
        finally:
            os.unlink(filepath)


class TestCreateComplianceConfig:
    """Tests for create_compliance_config function."""

    def test_strict_mode(self):
        """Should create strict config."""
        config = create_compliance_config(strict=True)
        assert config["audit_enabled"] is True
        assert config["data_classification"] == "confidential"
        assert config["retention_days"] == 30
        assert config["encryption_at_rest"] is True
        assert config["encryption_in_transit"] is True
        assert config["access_logging"] is True

    def test_non_strict_mode(self):
        """Should create non-strict config."""
        config = create_compliance_config(strict=False)
        assert config["data_classification"] == "internal"
        assert config["retention_days"] == 90
        assert config["encryption_at_rest"] is False
        assert config["encryption_in_transit"] is False
        assert config["access_logging"] is False

    def test_default_is_strict(self):
        """Should default to strict mode."""
        config = create_compliance_config()
        assert config["data_classification"] == "confidential"
