"""
Structured Logging Configuration
================================
Production-ready structured logging with structlog.

Structured logging makes logs queryable and analyzable.
Unlike traditional logging, structured logs are key-value pairs
that can be indexed and searched efficiently.

تسجيل سجلات منظم باستخدام structlog
"""

import logging
import logging.config
import sys
from typing import Any, Callable
from contextvars import ContextVar
from datetime import datetime

import structlog
from structlog.types import Processor


# -----------------------------------------------------------------------------
# Context Variables for Distributed Tracing
# -----------------------------------------------------------------------------
REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
TENANT_ID: ContextVar[str | None] = ContextVar("tenant_id", default=None)
USER_ID: ContextVar[str | None] = ContextVar("user_id", default=None)


# -----------------------------------------------------------------------------
# Custom Processors
# -----------------------------------------------------------------------------


def add_request_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add request_id from context to log entry."""
    request_id = REQUEST_ID.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_tenant_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add tenant_id from context to log entry."""
    tenant_id = TENANT_ID.get()
    if tenant_id:
        event_dict["tenant_id"] = tenant_id
    return event_dict


def add_user_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add user_id from context to log entry."""
    user_id = USER_ID.get()
    if user_id:
        event_dict["user_id"] = user_id
    return event_dict


def drop_color_message_key(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Remove color-related keys for clean JSON output."""
    event_dict.pop("color_message", None)
    return event_dict


def filter_by_level(
    level: str = logging.INFO,
) -> Processor:
    """Filter logs below specified level."""

    def processor(
        logger: logging.Logger,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        if method_name.lower() == "debug" and level > logging.DEBUG:
            raise structlog.DropEvent
        if method_name.lower() == "info" and level > logging.INFO:
            raise structlog.DropEvent
        if method_name.lower() == "warning" and level > logging.WARNING:
            raise structlog.DropEvent
        return event_dict

    return processor


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configure structlog for production logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON; if False, pretty console output
        log_file: Optional file path to write logs to

    إعداد structlog للتسجيل في الإنتاج
    """
    shared_processors = [
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Add log level
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add context (request_id, tenant_id, user_id)
        add_request_id,
        add_tenant_id,
        add_user_id,
        # Add exception info if present
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        # Production: JSON output
        processors = shared_processors + [
            # Rename "event" to "message" for compatibility
            structlog.processors.EventRenamer("message"),
            # Clean up internal keys
            drop_color_message_key,
            # Render as JSON
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty console output
        processors = shared_processors + [
            # Format exception nicely
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # If log file specified, add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    الحصول على مثيل مسجل منظم
    """
    return structlog.get_logger(name)


# -----------------------------------------------------------------------------
# Context Managers for Request Context
# -----------------------------------------------------------------------------


class RequestContext:
    """
    Context manager for request-scoped logging context.

    مدير سياق لتسجيل السياق المستند إلى الطلب
    """

    def __init__(
        self,
        request_id: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ):
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.request_id_token = None
        self.tenant_id_token = None
        self.user_id_token = None

    def __enter__(self) -> "RequestContext":
        """Set context variables on entry."""
        if self.request_id:
            self.request_id_token = REQUEST_ID.set(self.request_id)
        if self.tenant_id:
            self.tenant_id_token = TENANT_ID.set(self.tenant_id)
        if self.user_id:
            self.user_id_token = USER_ID.set(self.user_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Reset context variables on exit."""
        if self.request_id_token:
            REQUEST_ID.reset(self.request_id_token)
        if self.tenant_id_token:
            TENANT_ID.reset(self.tenant_id_token)
        if self.user_id_token:
            USER_ID.reset(self.user_id_token)
        return None


# -----------------------------------------------------------------------------
# RAG-Specific Logging Helpers
# -----------------------------------------------------------------------------


class RAGLogger:
    """
    Specialized logger for RAG pipeline operations.

    مسجل متخصص لعمليات خط أنبوب RAG
    """

    def __init__(self, name: str = "rag"):
        self.logger = get_logger(name)

    def log_request(
        self,
        tenant_id: str,
        question: str,
        k: int,
        model: str,
    ) -> None:
        """Log incoming RAG request."""
        self.logger.info(
            "rag_request",
            tenant_id=tenant_id,
            question_length=len(question),
            k=k,
            model=model,
        )

    def log_retrieval(
        self,
        tenant_id: str,
        vector_results: int,
        keyword_results: int,
        fused_results: int,
        reranked_results: int,
    ) -> None:
        """Log retrieval metrics."""
        self.logger.info(
            "rag_retrieval",
            tenant_id=tenant_id,
            vector_results=vector_results,
            keyword_results=keyword_results,
            fused_results=fused_results,
            reranked_results=reranked_results,
        )

    def cache_hit(self, key: str, tier: str) -> None:
        """Log cache hit."""
        self.logger.debug(
            "cache_hit",
            key=key,
            tier=tier,
        )

    def cache_miss(self, key: str, tier: str) -> None:
        """Log cache miss."""
        self.logger.debug(
            "cache_miss",
            key=key,
            tier=tier,
        )

    def llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
    ) -> None:
        """Log LLM API call."""
        self.logger.info(
            "llm_call",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            duration_ms=duration_ms,
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log error with context."""
        self.logger.error(
            "error",
            error_type=error_type,
            error_message=error_message,
            **(context or {}),
        )


def get_rag_logger() -> RAGLogger:
    """Get a RAGLogger instance."""
    return RAGLogger()


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Configure logging
    configure_logging(level="DEBUG", json_output=False)

    # Get logger
    logger = get_logger("example")

    # Basic logging
    logger.info("application_started", version="1.0.0")

    # With request context
    with RequestContext(request_id="req-123", tenant_id="tenant-456", user_id="user-789"):
        logger.info("processing_request", path="/api/v1/ask", method="POST")
        logger.warning("cache_miss", key="embedding:abc123")
        logger.error(
            "api_error",
            status_code=500,
            message="Internal server error",
        )

    # RAG-specific logging
    rag_logger = get_rag_logger()
    rag_logger.log_request(
        tenant_id="tenant-123",
        question="What is RAG?",
        k=10,
        model="gpt-4",
    )
    rag_logger.log_retrieval(
        tenant_id="tenant-123",
        vector_results=30,
        keyword_results=15,
        fused_results=40,
        reranked_results=8,
    )
    rag_logger.cache_hit(key="embedding:abc", tier="redis")
    rag_logger.llm_call(
        model="gpt-4",
        prompt_tokens=500,
        completion_tokens=300,
        duration_ms=1500,
    )
