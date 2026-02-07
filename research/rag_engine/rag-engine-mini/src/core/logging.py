"""
Structured Logging Setup
========================
Uses structlog for JSON-formatted, context-rich logging.

إعداد السجلات المنظمة - يستخدم structlog للتنسيق JSON
"""

import logging
import sys
from typing import Literal

import structlog


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    json_format: bool = True,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Whether to output JSON (True for prod, False for dev)
    
    Design Decision: Using structlog for:
    - Structured JSON logs (easy to parse in log aggregators)
    - Context binding (request_id, tenant_id propagation)
    - Automatic exception formatting
    
    قرار التصميم: استخدام structlog للسجلات المنظمة
    """
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Shared processors
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,  # Merge context vars (request_id, etc.)
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if json_format:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Human-readable output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    
    # Also configure standard library logging for third-party libs
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Quiet noisy loggers
    for logger_name in ["httpx", "httpcore", "urllib3", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Optional logger name (usually module name)
    
    Returns:
        Bound logger instance
    
    Usage:
        log = get_logger(__name__)
        log.info("processing_document", doc_id="123", chunks=42)
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger=name)
    return logger


def bind_context(**kwargs) -> None:
    """
    Bind context variables that will be included in all subsequent logs.
    
    Usage:
        bind_context(request_id="abc-123", tenant_id="user-456")
        # All subsequent logs will include these fields
    
    ربط متغيرات السياق التي ستُضمّن في جميع السجلات اللاحقة
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """
    Clear all bound context variables.
    Should be called at the end of request handling.
    
    مسح جميع متغيرات السياق المربوطة
    """
    structlog.contextvars.clear_contextvars()
