"""
Unified Logging Configuration for AI-Mastery-2026
==================================================

Provides consistent, structured logging across all modules.

Features:
---------
- Structured JSON logging for production
- Console logging with colors for development
- Log level configuration via environment
- Sensitive data filtering
- Performance logging decorators
- Request/response logging for APIs

Usage:
------
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    # Basic logging
    logger.info("Processing started", extra={"doc_id": "123"})
    logger.error("Processing failed", exc_info=True)

    # Structured logging
    logger.info(
        "RAG query completed",
        extra={
            "query_length": 50,
            "results_count": 5,
            "latency_ms": 125.3,
        },
    )

Configuration:
--------------
Set via environment variables:
    LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT=json         # json, text
    LOG_FILE=/var/log/app.log
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

# Try to import colorama for colored output
try:
    from colorama import Fore, Style
    from colorama import init as colorama_init

    colorama_init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


# Log level mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Default configuration
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # json or text
DEFAULT_LOG_FILE = os.getenv("LOG_FILE", None)


# Color mappings for console output
COLOR_MAP = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.RED + Style.BRIGHT,
}


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development.

    Provides color-coded log levels for easier visual scanning.
    """

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(
            fmt=fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt=datefmt or "%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        # Save original level
        original_level = record.levelno

        # Add color
        if COLORS_AVAILABLE:
            color = COLOR_MAP.get(original_level, "")
            reset = Style.RESET_ALL
            record.levelname = f"{color}{record.levelname}{reset}"

        # Format
        result = super().format(record)

        return result


class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter for production.

    Outputs logs as JSON for easy parsing by log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data["extra"] = record.extra_fields

        # Add process info
        log_data["process"] = {
            "pid": os.getpid(),
            "thread": record.thread,
        }

        return json.dumps(log_data, default=str)


class SensitiveDataFilter(logging.Filter):
    """
    Filter to redact sensitive data from logs.

    Prevents accidental logging of:
    - API keys
    - Passwords
    - Tokens
    - PII
    """

    SENSITIVE_PATTERNS = [
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "auth",
        "credential",
        "private_key",
        "access_token",
        "refresh_token",
    ]

    REDACTED_VALUE = "[REDACTED]"

    def filter(self, record: logging.LogRecord) -> bool:
        # Filter message
        record.msg = self._redact(str(record.msg))

        # Filter args
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact_value(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._redact_value(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )

        # Filter extra fields
        if hasattr(record, "extra_fields"):
            record.extra_fields = {
                k: self._redact_value(v) if isinstance(v, str) else v
                for k, v in record.extra_fields.items()
            }

        return True

    def _redact(self, value: str) -> str:
        """Redact sensitive patterns from a string."""
        result = value
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.lower() in result.lower():
                result = self.REDACTED_VALUE
                break
        return result

    def _redact_value(self, value: str) -> str:
        """Redact value if it looks sensitive."""
        # Check if value itself looks like a secret
        if len(value) > 20 and any(c.isdigit() for c in value):
            # Long string with digits might be a token/key
            return self.REDACTED_VALUE

        return self._redact(value)


def get_logger(
    name: str,
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
    structured: Optional[bool] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: from LOG_LEVEL env var)
        log_file: Optional file path for file logging
        structured: Use JSON format (default: from LOG_FORMAT env var)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing document", extra={"doc_id": "123"})
    """
    # Resolve configuration
    if level is None:
        level = LOG_LEVELS.get(DEFAULT_LOG_LEVEL.upper(), logging.INFO)

    if structured is None:
        structured = DEFAULT_LOG_FORMAT.lower() == "json"

    # Get or create logger
    logger = logging.getLogger(name)

    # Return if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Choose formatter
    if structured:
        formatter = JSONFormatter()
    else:
        formatter = ColoredFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add sensitive data filter
    logger.addFilter(SensitiveDataFilter())

    return logger


# Custom logging methods with extra field handling
class ExtraLogger(logging.Logger):
    """
    Custom logger class that handles extra fields properly.
    """

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        if extra:
            # Store extra fields for formatter
            if extra is None:
                extra = {}
            extra_copy = extra.copy()
            extra = {"extra_fields": extra_copy}

        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)


# Register custom logger class
logging.setLoggerClass(ExtraLogger)


# Performance logging decorator
F = TypeVar("F", bound=Callable)


def log_performance(
    logger_name: str = __name__,
    log_level: int = logging.INFO,
    include_args: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to log function performance.

    Args:
        logger_name: Name of logger to use
        log_level: Log level for performance info
        include_args: Whether to log function arguments

    Returns:
        Decorated function

    Example:
        >>> @log_performance(__name__, include_args=True)
        ... def process_documents(docs):
        ...     # Function body
        ...     pass
    """

    def decorator(func: F) -> F:
        logger = get_logger(logger_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()

            # Log start
            if log_level <= logging.DEBUG:
                log_args = ""
                if include_args:
                    log_args = f"args={args}, kwargs={kwargs}"
                logger.debug(f"Starting {func.__name__}({log_args})")

            try:
                result = func(*args, **kwargs)

                # Log success
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.log(
                    log_level,
                    f"{func.__name__} completed",
                    extra={
                        "function": func.__name__,
                        "duration_ms": round(duration, 2),
                        "status": "success",
                    },
                )

                return result

            except Exception as e:
                # Log error
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.error(
                    f"{func.__name__} failed",
                    extra={
                        "function": func.__name__,
                        "duration_ms": round(duration, 2),
                        "status": "error",
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# Request/Response logging for APIs
def log_request_response(logger_name: str = __name__):
    """
    Decorator to log API request/response.

    Example:
        >>> @log_request_response(__name__)
        ... async def query_endpoint(request: QueryRequest):
        ...     # Handler body
        ...     pass
    """

    def decorator(func: F) -> F:
        logger = get_logger(logger_name)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            request_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

            # Log request
            logger.info(
                "Request started",
                extra={
                    "request_id": request_id,
                    "function": func.__name__,
                },
            )

            try:
                result = await func(*args, **kwargs)

                # Log success
                logger.info(
                    "Request completed",
                    extra={
                        "request_id": request_id,
                        "function": func.__name__,
                        "status": "success",
                    },
                )

                return result

            except Exception as e:
                # Log error
                logger.error(
                    "Request failed",
                    extra={
                        "request_id": request_id,
                        "function": func.__name__,
                        "status": "error",
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# Context manager for timing code blocks
class log_duration:
    """
    Context manager to log duration of a code block.

    Example:
        >>> with log_duration(logger, "Processing documents"):
        ...     # Code to time
        ...     pass
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO,
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[datetime] = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds() * 1000

        if exc_type is not None:
            self.logger.error(
                f"Failed: {self.operation}",
                extra={
                    "operation": self.operation,
                    "duration_ms": round(duration, 2),
                    "error": str(exc_val),
                },
                exc_info=True,
            )
        else:
            self.logger.log(
                self.level,
                f"Completed: {self.operation}",
                extra={
                    "operation": self.operation,
                    "duration_ms": round(duration, 2),
                },
            )

        return False  # Don't suppress exceptions


# Convenience functions
def debug(logger: logging.Logger, message: str, **kwargs) -> None:
    """Log debug message with extra fields."""
    logger.debug(message, extra=kwargs)


def info(logger: logging.Logger, message: str, **kwargs) -> None:
    """Log info message with extra fields."""
    logger.info(message, extra=kwargs)


def warning(logger: logging.Logger, message: str, **kwargs) -> None:
    """Log warning message with extra fields."""
    logger.warning(message, extra=kwargs)


def error(
    logger: logging.Logger, message: str, exc_info: bool = False, **kwargs
) -> None:
    """Log error message with extra fields."""
    logger.error(message, extra=kwargs, exc_info=exc_info)


def critical(
    logger: logging.Logger, message: str, exc_info: bool = False, **kwargs
) -> None:
    """Log critical message with extra fields."""
    logger.critical(message, extra=kwargs, exc_info=exc_info)
