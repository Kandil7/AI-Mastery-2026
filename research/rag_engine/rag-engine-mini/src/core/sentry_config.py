"""
Error Tracking with Sentry
========================
Production error tracking and alerting.

Sentry provides real-time error tracking with:
- Stack traces and context
- Issue aggregation
- Release tracking
- Performance monitoring

تتبع الأخطاء في الإنتاج مع Sentry
"""

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from typing import Any, Callable
from functools import wraps


def setup_sentry(
    dsn: str,
    environment: str = "production",
    release: str | None = None,
    sample_rate: float = 1.0,
    traces_sample_rate: float = 0.1,
    before_send_callback: Callable | None = None,
) -> None:
    """
    Initialize Sentry for error tracking.

    Args:
        dsn: Sentry DSN (Data Source Name)
        environment: Environment name (development, staging, production)
        release: Release version (e.g., "rag-engine@1.0.0")
        sample_rate: Error sampling rate (1.0 = 100%, 0.1 = 10%)
        traces_sample_rate: Performance tracing sample rate
        before_send_callback: Custom callback to filter/modify events

    تهيئة Sentry لتتبع الأخطاء
    """

    def default_before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any] | None:
        """Default before_send callback."""

        # Filter out health check errors (too noisy)
        if event.get("request", {}).get("path") == "/health":
            return None

        # Add custom tags
        event.setdefault("tags", {})["environment"] = environment

        # Apply custom callback if provided
        if before_send_callback:
            return before_send_callback(event, hint)

        return event

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        sample_rate=sample_rate,
        traces_sample_rate=traces_sample_rate,
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
            RedisIntegration(),
        ],
        before_send=default_before_send,
        # Track breadcrumbs (log messages leading up to error)
        max_breadcrumbs=100,
        # Track performance
        profiles_sample_rate=0.1,
        # Track user feedback
        attach_stacktrace=True,
    )


def capture_exception(
    error: Exception,
    level: str = "error",
    extra: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """
    Manually capture an exception to Sentry.

    Args:
        error: Exception object to capture
        level: Error level (error, warning, info)
        extra: Additional context data
        tags: Key-value tags for filtering

    التقاط استثناء يدوياً إلى Sentry
    """
    with sentry_sdk.push_scope() as scope:
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        if extra:
            scope.set_context("custom", extra)

        sentry_sdk.capture_exception(error, level=level)


def capture_message(
    message: str,
    level: str = "info",
    extra: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """
    Send a custom message to Sentry.

    Args:
        message: Message to send
        level: Level (error, warning, info, debug)
        extra: Additional context data
        tags: Key-value tags for filtering

    إرسال رسالة مخصصة إلى Sentry
    """
    with sentry_sdk.push_scope() as scope:
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        if extra:
            scope.set_context("custom", extra)

        sentry_sdk.capture_message(message, level=level)


def set_user_context(
    user_id: str,
    email: str | None = None,
    username: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Set user context for all events.

    Args:
        user_id: Unique user identifier
        email: User email (optional)
        username: Username (optional)
        extra: Additional user metadata

    تعيين سياق المستخدم لجميع الأحداث
    """
    with sentry_sdk.push_scope() as scope:
        user = {"id": user_id}

        if email:
            user["email"] = email
        if username:
            user["username"] = username
        if extra:
            user.update(extra)

        scope.set_user(user)


def set_request_context(
    request_id: str,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
) -> None:
    """
    Set request context for all events.

    تعيين سياق الطلب لجميع الأحداث
    """
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("request_id", request_id)
        scope.set_tag("method", method)
        scope.set_tag("path", path)

        if headers:
            scope.set_context("request", {"headers": headers})


def set_transaction_context(
    transaction: str,
    operation: str | None = None,
) -> None:
    """
    Set transaction name for performance tracking.

    تعيين اسم المعاملة لتتبع الأداء
    """
    sentry_sdk.set_tag("transaction", transaction)
    if operation:
        sentry_sdk.set_tag("operation", operation)


def add_breadcrumb(
    message: str,
    category: str = "custom",
    level: str = "info",
    data: dict[str, Any] | None = None,
) -> None:
    """
    Add a breadcrumb (event leading up to error).

    Breadcrumbs show what happened before an error.

    إضافة فتات خبز (أحداث تسبق الخطأ)
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data,
    )


def track_performance(
    operation: str,
    duration_ms: float,
    tags: dict[str, str] | None = None,
) -> None:
    """
    Track custom performance metric.

    تتبع مقياس أداء مخصص
    """
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("operation", operation)
        scope.set_tag("duration_ms", int(duration_ms))

        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)


def track_rag_error(
    error_type: str,
    error_message: str,
    tenant_id: str,
    question: str,
    retrieval_count: int | None = None,
    llm_model: str | None = None,
) -> None:
    """
    Specialized error tracking for RAG pipeline.

    تتبع أخطاء متخصص لخط أنبوب RAG
    """
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("error_type", error_type)
        scope.set_tag("tenant_id", tenant_id)

        context = {
            "question": question,
            "question_length": len(question),
            "retrieval_count": retrieval_count,
            "llm_model": llm_model,
        }
        scope.set_context("rag_pipeline", context)

        sentry_sdk.capture_message(
            f"RAG Error: {error_type} - {error_message}",
            level="error",
        )


# -----------------------------------------------------------------------------
# Decorators
# -----------------------------------------------------------------------------


def sentry_trace(operation_name: str | None = None):
    """
    Decorator for Sentry performance tracing.

    Usage:
        @sentry_trace("embedding_generation")
        def generate_embedding(text):
            ...

    مصمم لتتبع أداء Sentry
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation_name or func.__name__

            # Set transaction
            set_transaction_context(op_name)

            # Add breadcrumb
            add_breadcrumb(
                f"Starting {op_name}",
                category="function",
            )

            try:
                # Start transaction
                with sentry_sdk.start_transaction(
                    op_name,
                    op="function",
                ) as transaction:
                    result = func(*args, **kwargs)

                    if transaction:
                        transaction.set_status("ok")

                    return result
            except Exception as e:
                # Capture exception
                capture_exception(
                    e,
                    extra={"function": func.__name__, "args": str(args)},
                    tags={"operation": op_name},
                )
                raise

        return wrapper

    return decorator


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Setup Sentry (development: mock DSN)
    # setup_sentry(
    #     dsn=os.getenv("SENTRY_DSN"),
    #     environment="production",
    #     release="rag-engine@1.0.0",
    #     sample_rate=1.0,
    #     traces_sample_rate=0.1,
    # )

    # Capture exception
    try:
        raise ValueError("Test error")
    except Exception as e:
        capture_exception(
            e,
            level="error",
            tags={"module": "test"},
            extra={"context": "test context"},
        )

    # Capture message
    capture_message(
        "Custom log message",
        level="info",
        tags={"component": "test"},
    )

    # Set user context
    set_user_context(
        user_id="user-123",
        email="test@example.com",
        username="testuser",
    )

    # Add breadcrumbs
    add_breadcrumb("User clicked button", category="ui")
    add_breadcrumb("API call started", category="api")

    # Track performance
    track_performance(
        operation="vector_search",
        duration_ms=150,
        tags={"model": "qdrant", "k": "10"},
    )

    # Track RAG error
    track_rag_error(
        error_type="RetrievalFailed",
        error_message="No results found",
        tenant_id="tenant-123",
        question="What is RAG?",
        retrieval_count=0,
        llm_model="gpt-4",
    )
