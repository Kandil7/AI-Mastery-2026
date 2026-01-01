"""
Observability for Production RAG Systems
=========================================

Implements production monitoring and observability for RAG systems.

Key Components:
- RAGMetrics: Prometheus-compatible metrics collection
- QualityMonitor: Automated quality assessment
- LatencyTracker: Percentile latency tracking

Production Pattern:
Observability for LLM systems extends beyond traditional APM to include:
- Quality metrics (faithfulness, relevance, hallucination rate)
- Cost metrics (tokens, cache hit rate, model usage)
- Safety metrics (PII detection, content policy violations)

Reference: "From Prototype to Production: Enterprise RAG Systems"
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from collections import defaultdict
import numpy as np


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class MetricPoint:
    """A single metric observation."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics."""
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class QualityScore:
    """Quality assessment result."""
    faithfulness: float
    relevance: float
    groundedness: float
    hallucination_detected: bool
    confidence: float
    issues: List[str] = field(default_factory=list)


# ============================================================
# LATENCY TRACKING
# ============================================================

class LatencyTracker:
    """
    Track and analyze latency percentiles.
    
    Production Pattern:
    LLM applications have highly variable latency due to:
    - Token count variation
    - Model load balancing
    - Network conditions
    
    Track P50, P90, P95, P99 to understand the distribution.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.observations: Dict[str, List[float]] = defaultdict(list)
    
    def record(self, operation: str, latency_ms: float) -> None:
        """Record a latency observation."""
        observations = self.observations[operation]
        observations.append(latency_ms)
        
        # Maintain window
        if len(observations) > self.window_size:
            self.observations[operation] = observations[-self.window_size:]
    
    def get_stats(self, operation: str) -> Optional[LatencyStats]:
        """Get latency statistics for an operation."""
        observations = self.observations.get(operation, [])
        
        if not observations:
            return None
        
        arr = np.array(observations)
        
        return LatencyStats(
            count=len(arr),
            mean_ms=float(np.mean(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr))
        )
    
    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get stats for all tracked operations."""
        return {
            op: self.get_stats(op)
            for op in self.observations.keys()
            if self.get_stats(op) is not None
        }


# ============================================================
# QUALITY MONITORING
# ============================================================

class QualityMonitor:
    """
    Automated quality assessment for RAG responses.
    
    Production Pattern:
    Continuously monitor response quality:
    - Faithfulness: Is the answer grounded in retrieved docs?
    - Relevance: Does the answer address the question?
    - Hallucination: Are there fabricated facts?
    
    Uses heuristics by default, LLM-as-judge for detailed analysis.
    """
    
    def __init__(
        self,
        llm_judge: Optional[Callable] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        self.llm_judge = llm_judge
        self.thresholds = thresholds or {
            "faithfulness": 0.7,
            "relevance": 0.6,
            "hallucination_rate": 0.1
        }
        
        # Track scores
        self.scores: List[QualityScore] = []
    
    def assess(
        self,
        query: str,
        response: str,
        context: List[str]
    ) -> QualityScore:
        """
        Assess response quality.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved documents used for generation
        """
        if self.llm_judge:
            return self._llm_assess(query, response, context)
        return self._heuristic_assess(query, response, context)
    
    def _llm_assess(
        self, 
        query: str, 
        response: str, 
        context: List[str]
    ) -> QualityScore:
        """Use LLM to assess quality (placeholder)."""
        # In production, call LLM with evaluation prompt
        result = self.llm_judge(query, response, context)
        self.scores.append(result)
        return result
    
    def _heuristic_assess(
        self,
        query: str,
        response: str,
        context: List[str]
    ) -> QualityScore:
        """Heuristic-based quality assessment."""
        issues = []
        
        # Faithfulness: Check if response terms appear in context
        context_text = " ".join(context).lower()
        response_words = set(response.lower().split())
        
        # Remove common words
        common = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "to", "of", "and", "or", "in", "on", "at", "for", "with"}
        response_words = response_words - common
        
        grounded_words = sum(1 for w in response_words if w in context_text)
        faithfulness = grounded_words / len(response_words) if response_words else 0
        
        if faithfulness < 0.5:
            issues.append("Low grounding in retrieved context")
        
        # Relevance: Check query-response overlap
        query_words = set(query.lower().split()) - common
        relevance_overlap = len(query_words & response_words)
        relevance = min(1.0, relevance_overlap / len(query_words)) if query_words else 0
        
        if relevance < 0.3:
            issues.append("Response may not address the query")
        
        # Hallucination detection (simple heuristics)
        hallucination_patterns = [
            "I cannot", "I don't have", "I'm not sure",
            "studies show", "research indicates", "according to experts"
        ]
        
        hallucination_detected = False
        response_lower = response.lower()
        
        # Check for broad claims without grounding
        for pattern in hallucination_patterns:
            if pattern in response_lower:
                if pattern not in context_text:
                    hallucination_detected = True
                    issues.append(f"Potential hallucination: '{pattern}'")
                    break
        
        # Check for specific claims (dates, numbers)
        import re
        numbers = re.findall(r'\b\d{4}\b|\b\d+%\b|\$\d+', response)
        for num in numbers:
            if num not in context_text:
                hallucination_detected = True
                issues.append(f"Ungrounded specific claim: {num}")
                break
        
        score = QualityScore(
            faithfulness=faithfulness,
            relevance=relevance,
            groundedness=(faithfulness + relevance) / 2,
            hallucination_detected=hallucination_detected,
            confidence=0.6,  # Lower for heuristic assessment
            issues=issues
        )
        
        self.scores.append(score)
        return score
    
    def get_summary(self) -> Dict[str, Any]:
        """Get quality summary statistics."""
        if not self.scores:
            return {"message": "No assessments recorded"}
        
        return {
            "total_assessments": len(self.scores),
            "avg_faithfulness": np.mean([s.faithfulness for s in self.scores]),
            "avg_relevance": np.mean([s.relevance for s in self.scores]),
            "avg_groundedness": np.mean([s.groundedness for s in self.scores]),
            "hallucination_rate": sum(1 for s in self.scores if s.hallucination_detected) / len(self.scores),
            "issues_count": sum(len(s.issues) for s in self.scores)
        }


# ============================================================
# METRICS COLLECTION
# ============================================================

class RAGMetrics:
    """
    Prometheus-compatible metrics collection.
    
    Production Pattern:
    Collect metrics in standard format for dashboards and alerts:
    
    - Counters: Total requests, errors, cache hits
    - Gauges: Active requests, cache size, model load
    - Histograms: Latency distributions, token counts
    
    Exports in Prometheus text format for scraping.
    """
    
    def __init__(self):
        # Counters
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Gauges (current values)
        self.gauges: Dict[str, float] = {}
        
        # Histograms (for distribution tracking)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Labels storage
        self.labeled_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Latency tracker
        self.latency = LatencyTracker()
    
    def inc(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        if labels:
            label_key = str(sorted(labels.items()))
            self.labeled_counters[name][label_key] += value
        else:
            self.counters[name] += value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self.gauges[name] = value
    
    def observe(self, name: str, value: float) -> None:
        """Add observation to histogram."""
        self.histograms[name].append(value)
        
        # Maintain size limit
        if len(self.histograms[name]) > 10000:
            self.histograms[name] = self.histograms[name][-5000:]
    
    def record_latency(self, operation: str, latency_ms: float) -> None:
        """Record latency for an operation."""
        self.latency.record(operation, latency_ms)
        self.observe(f"{operation}_latency_ms", latency_ms)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all metrics."""
        histogram_stats = {}
        for name, values in self.histograms.items():
            if values:
                arr = np.array(values)
                histogram_stats[name] = {
                    "count": len(arr),
                    "sum": float(np.sum(arr)),
                    "mean": float(np.mean(arr)),
                    "p50": float(np.percentile(arr, 50)),
                    "p99": float(np.percentile(arr, 99))
                }
        
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": histogram_stats,
            "latency": {k: v.__dict__ for k, v in self.latency.get_all_stats().items()}
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = ["# RAG System Metrics\n"]
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Histogram summaries
        for name, values in self.histograms.items():
            if values:
                arr = np.array(values)
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(arr)}")
                lines.append(f"{name}_sum {np.sum(arr)}")
                for p in [50, 90, 95, 99]:
                    lines.append(f'{name}{{quantile="{p/100}"}} {np.percentile(arr, p)}')
        
        return "\n".join(lines)


# ============================================================
# UNIFIED OBSERVABILITY
# ============================================================

class RAGObservability:
    """
    Unified observability layer for RAG systems.
    
    Combines metrics, quality monitoring, and latency tracking.
    
    Example:
        obs = RAGObservability()
        
        # Record a RAG request
        with obs.track_request("query"):
            result = rag_pipeline(query)
        
        # Assess quality  
        obs.assess_quality(query, result.answer, result.contexts)
        
        # Get dashboard data
        dashboard = obs.get_dashboard()
    """
    
    def __init__(
        self,
        llm_judge: Optional[Callable] = None,
        enable_quality_monitoring: bool = True
    ):
        self.metrics = RAGMetrics()
        self.quality = QualityMonitor(llm_judge=llm_judge) if enable_quality_monitoring else None
        self.latency = LatencyTracker()
        
        # Request tracking
        self._active_requests: Dict[str, float] = {}
    
    def track_request(self, request_type: str = "query"):
        """Context manager for tracking request latency."""
        return RequestTracker(self, request_type)
    
    def record_retrieval(
        self,
        query: str,
        num_docs: int,
        latency_ms: float
    ) -> None:
        """Record retrieval metrics."""
        self.metrics.inc("retrieval_total")
        self.metrics.observe("retrieval_docs_count", num_docs)
        self.metrics.record_latency("retrieval", latency_ms)
    
    def record_generation(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float
    ) -> None:
        """Record generation metrics."""
        self.metrics.inc("generation_total", labels={"model": model})
        self.metrics.observe("prompt_tokens", prompt_tokens)
        self.metrics.observe("completion_tokens", completion_tokens)
        self.metrics.record_latency("generation", latency_ms)
    
    def record_cache_hit(self, hit: bool) -> None:
        """Record cache hit/miss."""
        if hit:
            self.metrics.inc("cache_hits")
        else:
            self.metrics.inc("cache_misses")
    
    def record_error(self, error_type: str) -> None:
        """Record an error."""
        self.metrics.inc("errors_total", labels={"type": error_type})
    
    def assess_quality(
        self,
        query: str,
        response: str,
        context: List[str]
    ) -> Optional[QualityScore]:
        """Assess response quality."""
        if self.quality:
            return self.quality.assess(query, response, context)
        return None
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        data = {
            "metrics": self.metrics.get_all(),
            "latency": self.latency.get_all_stats()
        }
        
        if self.quality:
            data["quality"] = self.quality.get_summary()
        
        return data
    
    def get_prometheus_metrics(self) -> str:
        """Export Prometheus metrics."""
        return self.metrics.export_prometheus()


class RequestTracker:
    """Context manager for tracking request timing."""
    
    def __init__(self, obs: RAGObservability, request_type: str):
        self.obs = obs
        self.request_type = request_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.obs.metrics.inc(f"{self.request_type}_started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        self.obs.metrics.record_latency(self.request_type, latency_ms)
        
        if exc_type:
            self.obs.record_error(str(exc_type.__name__))
        else:
            self.obs.metrics.inc(f"{self.request_type}_completed")
        
        return False


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Models
    "MetricPoint",
    "LatencyStats",
    "QualityScore",
    # Components
    "LatencyTracker",
    "QualityMonitor",
    "RAGMetrics",
    # Unified
    "RAGObservability",
    "RequestTracker",
]
