import pytest  # noqa: F401
import os
import json
import time
from core.compliance import (
    AuditEvent,
    AuditLogger,
    DataLifecycleManager,
    ComplianceManager,
    create_compliance_config,
)


def test_audit_event():
    event = AuditEvent(event_type="test_event", user_id="user1", operation="read")
    d = event.to_dict()
    assert d["event_type"] == "test_event"
    assert d["user_id"] == "user1"
    assert "event_id" in d


def test_audit_logger(tmp_path):
    logger = AuditLogger(enabled=True, data_classification="confidential")

    # Generic event
    event = logger.log_event(
        "type1", operation="op1", resource="res1", result="ok", metadata={"m": 1}
    )
    assert event.event_type == "type1"
    assert len(logger.get_events()) == 1

    # Config change
    logger.log_config_change({"a": 1}, {"a": 2}, reason="test")
    events = logger.get_events()
    assert events[1].event_type == "config_change"
    assert "a" in events[1].metadata["changed_keys"]

    # Pipeline events
    logger.log_pipeline_start({"hierarchical_mode": "on"}, 100)
    logger.log_pipeline_complete(100, 5, 50.0)

    # Data retention
    logger.log_data_retention("data1", 30, "delete")

    # Export
    filepath = tmp_path / "audit.json"
    logger.export_events(str(filepath))
    assert os.path.exists(filepath)
    with open(filepath) as f:
        data = json.load(f)
        assert len(data) == 5


def test_audit_logger_disabled():
    logger = AuditLogger(enabled=False)
    assert logger.log_event("test") is None


def test_data_lifecycle_manager():
    dlm = DataLifecycleManager(default_retention_days=30)

    # Register
    dlm.register_data("d1", "sim")
    assert "d1" in dlm._data_registry

    # Register with override
    dlm.register_data("d2", "sim", retention_days=-1)  # Immediate expiry
    expired = dlm.check_expired()
    assert "d2" in expired

    # Delete
    assert dlm.delete_data("d1") is True
    assert dlm._data_registry["d1"]["deleted"] is True
    assert dlm.delete_data("nonexistent") is False

    # Anonymize
    hist = {"s": [1.0, 2.0]}
    anon = dlm.anonymize_history(hist)
    assert anon["s"] == hist["s"]
    assert anon is not hist


def test_compliance_manager(tmp_path):
    cm = ComplianceManager(audit_enabled=True, retention_days=60)
    cm.start_pipeline({"mode": "test"}, 10)
    time.sleep(0.01)
    cm.end_pipeline(10, 2)

    events = cm.audit.get_events()
    assert len(events) == 2
    assert events[1].metadata["duration_ms"] > 0

    filepath = tmp_path / "compliance.json"
    cm.export_audit_trail(str(filepath))
    assert os.path.exists(filepath)


def test_create_compliance_config():
    conf_strict = create_compliance_config(strict=True)
    assert conf_strict["data_classification"] == "confidential"
    assert conf_strict["retention_days"] == 30

    conf_loose = create_compliance_config(strict=False)
    assert conf_loose["data_classification"] == "internal"
    assert conf_loose["retention_days"] == 90
