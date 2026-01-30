"""
OpenTelemetry Tracing Setup
============================
Distributed tracing setup for RAG pipeline.

OpenTelemetry is a vendor-neutral standard for observability data.
It provides automatic and manual instrumentation for tracing requests
across services.

تتبع OpenTelemetry للتوزيع
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import os
from typing import Callable, Any
from functools import wraps


def setup_tracing(
    service_name: str = "rag-engine",
    jaeger_host: str | None = None,
    otlp_endpoint: str | None = None,
    sample_rate: float = 0.1,
) -> TracerProvider:
    """
    Setup OpenTelemetry tracing for the application.

    Args:
        service_name: Name of the service (appears in traces)
        jaeger_host: Jaeger agent host (e.g., "localhost")
        otlp_endpoint: OTLP endpoint (e.g., "http://localhost:4317")
        sample_rate: Sampling rate (0.0 to 1.0). 0.1 = 10% of traces

    Returns:
        Configured TracerProvider

    إعداد تتبع OpenTelemetry للتطبيق
    """
    # Create resource with service name
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add exporter(s)
    if jaeger_host:
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=6831,
        )
        provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set as global default
    trace.set_tracer_provider(provider)

    return provider


def instrument_fastapi(app, service_name: str = "rag-engine") -> None:
    """
    Automatically instrument FastAPI for tracing.

    Includes:
    - Request/response timing
    - HTTP method and path
    - Status codes
    - Error handling

    أتمتة تتبع FastAPI
    """
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    app.state.tracer = trace.get_tracer(f"{service_name}.fastapi")


def instrument_sqlalchemy(engine) -> None:
    """
    Automatically instrument SQLAlchemy for tracing.

    Tracks:
    - Query execution time
    - SQL statements (sanitized)
    - Database connection info

    أتمتة تتبع SQLAlchemy
    """
    SQLAlchemyInstrumentor().instrument(engine=engine)


def instrument_requests() -> None:
    """
    Automatically instrument HTTP requests for tracing.

    Tracks outbound HTTP calls (e.g., to LLM APIs).

    أتمتة تتبع طلبات HTTP
    """
    RequestsInstrumentor().instrument()


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer instance for manual instrumentation.

    الحصول على مثيل Tracer للتتبع اليدوي
    """
    return trace.get_tracer(name)


def trace_operation(tracer_name: str, operation_name: str | None = None):
    """
    Decorator for tracing functions.

    Usage:
        @trace_operation("rag", "embedding_generation")
        def generate_embedding(text):
            ...

    مصمم لتتبع الدوال
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(tracer_name)
            op_name = operation_name or func.__name__

            with tracer.start_as_current_span(op_name) as span:
                # Add function name as attribute
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Try to execute and record result
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", "true")
                    return result
                except Exception as e:
                    span.set_attribute("function.success", "false")
                    span.set_attribute("function.error", str(e))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


class SpanContext:
    """
    Helper class for adding structured context to spans.

    فئة مساعدة لإضافة سياق منظم للـ spans
    """

    def __init__(self, span: trace.Span):
        self._span = span

    def add_attributes(self, **kwargs) -> "SpanContext":
        """Add multiple attributes at once."""
        for key, value in kwargs.items():
            self._span.set_attribute(key, str(value))
        return self

    def add_event(self, name: str, **kwargs) -> "SpanContext":
        """Add an event to the span."""
        self._span.add_event(name, kwargs)
        return self

    def set_status(self, status: str) -> "SpanContext":
        """Set the span status."""
        if status == "ok":
            self._span.set_status(trace.Status(trace.StatusCode.OK))
        elif status == "error":
            self._span.set_status(trace.Status(trace.StatusCode.ERROR))
        return self


def with_span_context(tracer_name: str, span_name: str):
    """
    Decorator that provides SpanContext to the function.

    Usage:
        @with_span_context("rag", "embedding_generation")
        def generate_embedding(text, ctx):
            ctx.add_attributes(text_length=len(text))
            ctx.add_event("embedding_started", model="openai")
            ...

    مصمم يوفر SpanContext للدالة
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(tracer_name)

            with tracer.start_as_current_span(span_name) as span:
                ctx = SpanContext(span)
                return func(*args, ctx=ctx, **kwargs)

        return wrapper

    return decorator


# -----------------------------------------------------------------------------
# RAG Pipeline Tracing Helpers
# -----------------------------------------------------------------------------


class RAGTracer:
    """
    Specialized tracer for RAG pipeline operations.

    متتبع متخصص لعمليات خط أنبوب RAG
    """

    def __init__(self, tracer: trace.Tracer):
        self._tracer = tracer

    def start_rag_pipeline(self, tenant_id: str, question: str) -> trace.Span:
        """Start the main RAG pipeline span."""
        span = self._tracer.start_as_current_span("rag_pipeline")
        span.set_attribute("tenant_id", tenant_id)
        span.set_attribute("question", question)
        span.set_attribute("question_length", len(question))
        return span

    def trace_embedding(self, text: str, cached: bool = False) -> trace.Span:
        """Trace embedding generation."""
        span = self._tracer.start_as_current_span("embedding_generation")
        span.set_attribute("text_length", len(text))
        span.set_attribute("cached", str(cached))
        return span

    def trace_vector_search(self, tenant_id: str, k: int, results_count: int) -> trace.Span:
        """Trace vector search operation."""
        span = self._tracer.start_as_current_span("vector_search")
        span.set_attribute("tenant_id", tenant_id)
        span.set_attribute("k", k)
        span.set_attribute("results_count", results_count)
        return span

    def trace_keyword_search(self, tenant_id: str, query: str, results_count: int) -> trace.Span:
        """Trace keyword search operation."""
        span = self._tracer.start_as_current_span("keyword_search")
        span.set_attribute("tenant_id", tenant_id)
        span.set_attribute("query", query)
        span.set_attribute("results_count", results_count)
        return span

    def trace_rerank(self, chunks_count: int, top_n: int) -> trace.Span:
        """Trace reranking operation."""
        span = self._tracer.start_as_current_span("rerank")
        span.set_attribute("chunks_count", chunks_count)
        span.set_attribute("top_n", top_n)
        return span

    def trace_llm_generation(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        temperature: float,
    ) -> trace.Span:
        """Trace LLM generation."""
        span = self._tracer.start_as_current_span("llm_generation")
        span.set_attribute("model", model)
        span.set_attribute("prompt_tokens", prompt_tokens)
        span.set_attribute("completion_tokens", completion_tokens)
        span.set_attribute("total_tokens", prompt_tokens + completion_tokens)
        span.set_attribute("temperature", temperature)
        return span


def get_rag_tracer() -> RAGTracer:
    """Get a RAGTracer instance."""
    return RAGTracer(trace.get_tracer("rag"))
