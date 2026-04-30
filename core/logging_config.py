"""Structured logging configuration for APGI.

Provides ops-grade logging with JSON output support for production deployments.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    audit_logging: bool = False,
) -> structlog.BoundLogger:
    """Configure structured logging for APGI.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Enable JSON output for production/structured log aggregation
        audit_logging: Enable audit logging for compliance

    Returns:
        Configured structlog logger
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Configure structlog processors
    shared_processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        # Production: JSON output for log aggregation
        # format_exc_info needed for JSON to capture exception details
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: pretty console output
        # ConsoleRenderer handles exceptions prettily, so we skip format_exc_info
        # to avoid the "Remove format_exc_info" warning
        processors = shared_processors + [structlog.dev.ConsoleRenderer(colors=True)]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger("apgi")  # type: ignore[no-any-return]

    if audit_logging:
        logger = logger.bind(audit_enabled=True)  # type: ignore[assignment]

    return logger  # type: ignore[no-any-return]


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name (defaults to 'apgi')

    Returns:
        Configured structlog logger
    """
    logger_name = name or "apgi"
    return structlog.get_logger(logger_name)  # type: ignore[no-any-return]
