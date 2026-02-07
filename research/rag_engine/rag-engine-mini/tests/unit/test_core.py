"""
Comprehensive Unit Tests for Core Components
=========================================
Test suite for metrics, observability, and utilities.

اختبارات الوحدة للمكونات الأساسية
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.core.observability import (
    API_REQUEST_COUNT,
    API_REQUEST_LATENCY,
    TOKEN_USAGE,
    RETRIEVAL_SCORE,
    EMBEDDING_CACHE_HIT,
    RERANK_DURATION,
    QUERY_EXPANSION_COUNT,
    track_request_latency,
    track_token_usage,
    track_cache_hit,
    track_retrieval_score,
)


class TestMetrics:
    """Test Prometheus metrics."""

    def test_api_request_count(self):
        """Test API request counter increment."""
        initial = API_REQUEST_COUNT.collect()[0].samples[0].value

        API_REQUEST_COUNT.labels(method="POST", endpoint="/ask", status="200").inc()

        new_value = API_REQUEST_COUNT.collect()[0].samples[0].value
        assert new_value == initial + 1

    def test_token_usage(self):
        """Test token usage counter."""
        initial = TOKEN_USAGE.collect()[0].samples[0].value

        TOKEN_USAGE.labels(model="gpt-4", type="prompt").inc(100)
        TOKEN_USAGE.labels(model="gpt-4", type="completion").inc(50)

        new_value = TOKEN_USAGE.collect()[0].samples[0].value
        assert new_value == initial + 150

    def test_embedding_cache_metrics(self):
        """Test cache hit/miss metrics."""
        initial_hit = EMBEDDING_CACHE_HIT.collect()[0].samples[0].value
        initial_miss = EMBEDDING_CACHE_HIT.collect()[0].samples[1].value

        EMBEDDING_CACHE_HIT.labels(result="hit").inc()
        EMBEDDING_CACHE_HIT.labels(result="miss").inc()

        new_hit = EMBEDDING_CACHE_HIT.collect()[0].samples[0].value
        new_miss = EMBEDDING_CACHE_HIT.collect()[0].samples[1].value
        assert new_hit == initial_hit + 1
        assert new_miss == initial_miss + 1

    def test_retrieval_score_histogram(self):
        """Test retrieval score histogram."""
        RETRIEVAL_SCORE.labels(method="vector").observe(0.85)
        RETRIEVAL_SCORE.labels(method="vector").observe(0.92)
        RETRIEVAL_SCORE.labels(method="keyword").observe(1.0)

        samples = RETRIEVAL_SCORE.collect()[0].samples
        assert len(samples) == 2  # vector + keyword
        assert sum(s.value for s in samples) == 3

    def test_rerank_duration(self):
        """Test rerank duration histogram."""
        RERANK_DURATION.labels(method="cross_encoder").observe(0.5)
        RERANK_DURATION.labels(method="cross_encoder").observe(0.7)

        samples = RERANK_DURATION.collect()[0].samples
        assert samples[0].value == 2

    def test_query_expansion_count(self):
        """Test query expansion counter."""
        initial = QUERY_EXPANSION_COUNT.collect()[0].samples[0].value

        QUERY_EXPANSION_COUNT.labels(expanded="true").inc()
        QUERY_EXPANSION_COUNT.labels(expanded="false").inc()

        new_value = QUERY_EXPANSION_COUNT.collect()[0].samples[0].value
        assert new_value == initial + 2


class TestDecorators:
    """Test metric decorators."""

    def test_track_request_latency_decorator(self):
        """Test request latency decorator."""

        @track_request_latency(API_REQUEST_LATENCY, {"endpoint": "/test"})
        def dummy_function():
            return "result"

        result = dummy_function()

        samples = API_REQUEST_LATENCY.collect()[0].samples
        assert len(samples) > 0
        assert samples[0].labels["endpoint"] == "/test"

    def test_track_request_latency_exception(self):
        """Test latency decorator handles exceptions."""

        @track_request_latency(API_REQUEST_LATENCY, {"endpoint": "/test"})
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        # Should still track latency even on exception
        samples = API_REQUEST_LATENCY.collect()[0].samples
        assert len(samples) > 0


class TestObservabilityHelpers:
    """Test observability helper functions."""

    def test_metrics_endpoint_exists(self):
        """Test metrics endpoint function exists."""
        from src.core.observability import metrics_endpoint, generate_latest, CONTENT_TYPE_LATEST

        content = generate_latest()
        assert content is not None
        assert isinstance(content, bytes)

    @patch("src.core.observability.Response")
    def test_metrics_endpoint_response(self, mock_response):
        """Test metrics endpoint returns correct response."""
        from src.core.observability import metrics_endpoint, generate_latest, CONTENT_TYPE_LATEST

        response = metrics_endpoint()

        assert response.media_type == CONTENT_TYPE_LATEST


class TestTracing:
    """Test OpenTelemetry tracing."""

    @patch("src.core.tracing.trace")
    def test_get_tracer(self, mock_trace):
        """Test get_tracer function."""
        from src.core.tracing import get_tracer

        tracer = get_tracer("test")

        assert tracer is not None
        mock_trace.get_tracer.assert_called_once_with("test")

    def test_trace_operation_decorator(self):
        """Test trace_operation decorator."""
        from src.core.tracing import trace_operation

        @trace_operation("test_tracer", "test_operation")
        def dummy_function(x):
            return x * 2

        result = dummy_function(5)

        assert result == 10

    def test_with_span_context(self):
        """Test with_span_context context manager."""
        from src.core.tracing import with_span_context

        @with_span_context("test_tracer", "test_span")
        def dummy_function(ctx):
            ctx.add_attributes(test_attr="test_value")
            return "result"

        result = dummy_function()

        assert result == "result"


class TestStructuredLogging:
    """Test structured logging configuration."""

    def test_configure_logging_json(self, tmp_path):
        """Test JSON logging configuration."""
        from src.core.logging_config import configure_logging, get_logger

        log_file = tmp_path / "test.json.log"
        configure_logging(level="DEBUG", json_output=True, log_file=str(log_file))

        logger = get_logger("test")
        logger.info("test_message", key="value")

        # Verify log file exists and is valid JSON
        assert log_file.exists()
        content = log_file.read_text()
        assert '"message":"test_message"' in content
        assert '"key":"value"' in content

    def test_request_context_manager(self):
        """Test RequestContext context manager."""
        from src.core.logging_config import RequestContext, REQUEST_ID, TENANT_ID, USER_ID

        with RequestContext(request_id="req-123", tenant_id="tenant-456", user_id="user-789"):
            assert REQUEST_ID.get() == "req-123"
            assert TENANT_ID.get() == "tenant-456"
            assert USER_ID.get() == "user-789"

        # Context should be reset after exit
        assert REQUEST_ID.get() is None
        assert TENANT_ID.get() is None
        assert USER_ID.get() is None

    def test_rag_logger(self):
        """Test RAGLogger specialized logger."""
        from src.core.logging_config import get_rag_logger

        rag_logger = get_rag_logger()

        # These methods should not raise exceptions
        rag_logger.log_request("tenant-123", "What is RAG?", 10, "gpt-4")
        rag_logger.log_retrieval("tenant-123", 30, 15, 40, 8)
        rag_logger.cache_hit("key-123", "redis")
        rag_logger.cache_miss("key-456", "redis")
        rag_logger.llm_call("gpt-4", 500, 300, 1500)


class TestSentryConfig:
    """Test Sentry error tracking configuration."""

    def test_capture_exception(self):
        """Test capturing exceptions."""
        from src.core.sentry_config import capture_exception

        # This should not raise exceptions
        capture_exception(ValueError("Test error"))
        capture_exception(RuntimeError("Another error"))

    def test_capture_message(self):
        """Test capturing messages."""
        from src.core.sentry_config import capture_message

        # This should not raise exceptions
        capture_message("Test message", level="info")
        capture_message("Warning message", level="warning")

    def test_set_user_context(self):
        """Test setting user context."""
        from src.core.sentry_config import set_user_context

        # This should not raise exceptions
        set_user_context("user-123", email="test@example.com", username="testuser")
        set_user_context("user-456", username="anotheruser")

    def test_track_rag_error(self):
        """Test RAG error tracking."""
        from src.core.sentry_config import track_rag_error

        # This should not raise exceptions
        track_rag_error(
            error_type="RetrievalFailed",
            error_message="No results found",
            tenant_id="tenant-123",
            question="What is RAG?",
            retrieval_count=0,
            llm_model="gpt-4",
        )


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Fixture for temporary path."""
    return tmp_path_factory.mktemp("rag_engine_test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
