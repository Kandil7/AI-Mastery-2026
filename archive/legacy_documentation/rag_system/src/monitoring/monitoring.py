"""
Monitoring and Cost Tracking for RAG System

Following RAG Pipeline Guide 2026 - Observability
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """Cost tracking configuration."""

    daily_budget_usd: float = 100.0
    monthly_budget_usd: float = 1000.0
    embedding_cost_per_1k: float = 0.0001  # Approximate
    llm_cost_per_1k_input: float = 0.01  # GPT-4o approximate
    llm_cost_per_1k_output: float = 0.03  # GPT-4o approximate


@dataclass
class QueryLog:
    """Log entry for a single query."""

    timestamp: str
    query: str
    latency_ms: float
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    retrieval_count: int = 0
    success: bool = True
    error: Optional[str] = None


class CostTracker:
    """
    Track costs for RAG operations.

    Tracks:
    - Embedding generation costs
    - LLM generation costs
    - Total spend vs budget
    """

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()
        self.daily_spend = 0.0
        self.monthly_spend = 0.0
        self.total_spend = 0.0
        self.query_count = 0

        self._load_history()

    def _load_history(self):
        """Load spending history from file."""

        history_file = "data/cost_history.json"

        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    self.daily_spend = data.get("daily_spend", 0)
                    self.monthly_spend = data.get("monthly_spend", 0)
                    self.total_spend = data.get("total_spend", 0)
                    self.query_count = data.get("query_count", 0)
            except Exception as e:
                logger.warning(f"Could not load cost history: {e}")

    def _save_history(self):
        """Save spending history."""

        os.makedirs("data", exist_ok=True)

        with open("data/cost_history.json", "w") as f:
            json.dump(
                {
                    "daily_spend": self.daily_spend,
                    "monthly_spend": self.monthly_spend,
                    "total_spend": self.total_spend,
                    "query_count": self.query_count,
                    "last_updated": datetime.now().isoformat(),
                },
                f,
            )

    def calculate_embedding_cost(self, num_texts: int) -> float:
        """Calculate cost for embedding generation."""

        # Approximate: $0.0001 per 1k texts (multilingual MPNet)
        cost = (num_texts / 1000) * self.config.embedding_cost_per_1k

        return cost

    def calculate_llm_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Calculate cost for LLM generation."""

        input_cost = (tokens_input / 1000) * self.config.llm_cost_per_1k_input
        output_cost = (tokens_output / 1000) * self.config.llm_cost_per_1k_output

        return input_cost + output_cost

    def log_query(
        self,
        tokens_input: int = 0,
        tokens_output: int = 0,
        num_retrieved: int = 0,
    ):
        """Log a query and update costs."""

        # Calculate costs
        embedding_cost = self.calculate_embedding_cost(num_retrieved)
        llm_cost = self.calculate_llm_cost(tokens_input, tokens_output)

        total_cost = embedding_cost + llm_cost

        # Update tracking
        self.daily_spend += total_cost
        self.monthly_spend += total_cost
        self.total_spend += total_cost
        self.query_count += 1

        # Save periodically
        if self.query_count % 10 == 0:
            self._save_history()

        return total_cost

    def check_budget(self) -> Dict[str, Any]:
        """Check budget status."""

        daily_remaining = max(0, self.config.daily_budget_usd - self.daily_spend)
        monthly_remaining = max(0, self.config.monthly_budget_usd - self.monthly_spend)

        daily_alert = self.daily_spend > self.config.daily_budget_usd * 0.9
        monthly_alert = self.monthly_spend > self.config.monthly_budget_usd * 0.9

        return {
            "daily": {
                "spent": self.daily_spend,
                "budget": self.config.daily_budget_usd,
                "remaining": daily_remaining,
                "alert": daily_alert,
            },
            "monthly": {
                "spent": self.monthly_spend,
                "budget": self.config.monthly_budget_usd,
                "remaining": monthly_remaining,
                "alert": monthly_alert,
            },
            "total_queries": self.query_count,
            "avg_cost_per_query": (
                self.total_spend / self.query_count if self.query_count > 0 else 0
            ),
        }

    def reset_daily(self):
        """Reset daily spend counter."""

        self.daily_spend = 0
        self._save_history()

    def reset_monthly(self):
        """Reset monthly spend counter."""

        self.monthly_spend = 0
        self._save_history()


class QueryLogger:
    """
    Log and analyze query patterns.
    """

    def __init__(self, max_logs: int = 10000):
        self.max_logs = max_logs
        self.logs: List[QueryLog] = []
        self._load_logs()

    def _load_logs(self):
        """Load logs from file."""

        log_file = "data/query_logs.json"

        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)
                    self.logs = [
                        QueryLog(
                            timestamp=item["timestamp"],
                            query=item["query"],
                            latency_ms=item["latency_ms"],
                            tokens_input=item.get("tokens_input", 0),
                            tokens_output=item.get("tokens_output", 0),
                            cost_usd=item.get("cost_usd", 0),
                            retrieval_count=item.get("retrieval_count", 0),
                            success=item.get("success", True),
                            error=item.get("error"),
                        )
                        for item in data
                    ]
            except Exception as e:
                logger.warning(f"Could not load query logs: {e}")

    def _save_logs(self):
        """Save logs to file."""

        os.makedirs("data", exist_ok=True)

        data = [
            {
                "timestamp": log.timestamp,
                "query": log.query,
                "latency_ms": log.latency_ms,
                "tokens_input": log.tokens_input,
                "tokens_output": log.tokens_output,
                "cost_usd": log.cost_usd,
                "retrieval_count": log.retrieval_count,
                "success": log.success,
                "error": log.error,
            }
            for log in self.logs
        ]

        with open("data/query_logs.json", "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def log(
        self,
        query: str,
        latency_ms: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
        retrieval_count: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Log a query."""

        log_entry = QueryLog(
            timestamp=datetime.now().isoformat(),
            query=query,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            retrieval_count=retrieval_count,
            success=success,
            error=error,
        )

        self.logs.append(log_entry)

        # Trim if needed
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs :]

        # Save periodically
        if len(self.logs) % 100 == 0:
            self._save_logs()

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get query statistics for the last N hours."""

        cutoff = datetime.now() - timedelta(hours=hours)

        recent_logs = [
            log for log in self.logs if datetime.fromisoformat(log.timestamp) > cutoff
        ]

        if not recent_logs:
            return {
                "query_count": 0,
                "avg_latency_ms": 0,
                "success_rate": 0,
            }

        # Calculate metrics
        latencies = [log.latency_ms for log in recent_logs]
        successes = sum(1 for log in recent_logs if log.success)

        return {
            "query_count": len(recent_logs),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "success_rate": successes / len(recent_logs),
            "total_cost_usd": sum(log.cost_usd for log in recent_logs),
            "avg_cost_per_query": (
                sum(log.cost_usd for log in recent_logs) / len(recent_logs)
            ),
        }

    def get_top_queries(self, hours: int = 24, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent queries."""

        cutoff = datetime.now() - timedelta(hours=hours)

        recent_logs = [
            log for log in self.logs if datetime.fromisoformat(log.timestamp) > cutoff
        ]

        # Count queries
        query_counts = defaultdict(int)
        for log in recent_logs:
            query_counts[log.query] += 1

        # Sort and return top
        sorted_queries = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"query": query, "count": count} for query, count in sorted_queries[:limit]
        ]


class RAGMonitor:
    """
    Complete monitoring system for RAG.
    """

    def __init__(self):
        self.cost_tracker = CostTracker()
        self.query_logger = QueryLogger()

    def log_query(
        self,
        query: str,
        latency_ms: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        retrieval_count: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Log a query with cost tracking."""

        # Calculate cost
        cost = self.cost_tracker.log_query(
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            num_retrieved=retrieval_count,
        )

        # Log query
        self.query_logger.log(
            query=query,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost,
            retrieval_count=retrieval_count,
            success=success,
            error=error,
        )

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for monitoring dashboard."""

        return {
            "cost": self.cost_tracker.check_budget(),
            "queries_last_24h": self.query_logger.get_statistics(hours=24),
            "queries_last_hour": self.query_logger.get_statistics(hours=1),
            "top_queries": self.query_logger.get_top_queries(hours=24),
        }


# Singleton instance
_monitor: Optional[RAGMonitor] = None


def get_monitor() -> RAGMonitor:
    """Get the global monitor instance."""

    global _monitor

    if _monitor is None:
        _monitor = RAGMonitor()

    return _monitor
