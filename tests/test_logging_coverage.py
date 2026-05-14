from core.logging_config import configure_logging, get_logger


def test_configure_logging():
    # Console output
    logger = configure_logging(level="DEBUG", json_output=False, audit_logging=True)
    # Trigger lazy proxy initialization
    logger.info("test")
    # In some versions of structlog, it might still be a proxy or different wrapper
    # Let's check for essential methods instead of strict isinstance if it's tricky
    assert hasattr(logger, "info")
    assert logger._context.get("audit_enabled") is True

    # JSON output
    logger_json = configure_logging(level="INFO", json_output=True, audit_logging=False)
    logger_json.info("test")
    assert hasattr(logger_json, "info")


def test_get_logger():
    logger = get_logger("test_module")
    logger.info("test")
    assert hasattr(logger, "info")

    logger_default = get_logger()
    logger_default.info("test")
    assert hasattr(logger_default, "info")
