"""
Cost Optimization for Production RAG Systems
=============================================

Implements Pillar 5: Managing cost and performance at scale.

Key Components:
- SemanticCache: Vector-based response caching
- ModelRouter: Intelligent tiered model selection  
- CostTracker: Token usage metering

Production Pattern:
Traditional exact-match caching fails for LLMs because semantically 
identical queries are rarely phrased identically. Semantic caching
uses vector similarity to identify cache hits.

Reference: "From Prototype to Production: Enterprise RAG Systems"
"""

import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class CacheEntry:
    """A cached response with metadata."""
    query: str
    query_embedding: np.ndarray
    response: str
    created_at: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > ttl_seconds


@dataclass
class QueryClassification:
    """Classification of query complexity."""
    tier: int  # 1=simple, 2=standard, 3=complex, 4=specialized
    confidence: float
    reasoning: str
    estimated_tokens: int


@dataclass
class CostReport:
    """Cost tracking report."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    queries_processed: int
    cache_hits: int
    cache_hit_rate: float
    model_usage: Dict[str, int]
    period_start: datetime
    period_end: datetime


# ============================================================
# SEMANTIC CACHING
# ============================================================

class SemanticCache:
    """
    Vector-based semantic cache for LLM responses.
    
    Production Pattern:
    Uses embedding similarity to match semantically equivalent queries,
    even if phrased differently.
    
    Example:
        "What is the return policy?" 
        Cache hit for: "How can I return an item?"
    
    Benefits:
    - 60-90% cost reduction on recurring queries
    - Sub-millisecond response for cache hits
    - Reduces LLM API latency
    
    Configuration:
        similarity_threshold: 0.85-0.95 (higher = stricter matching)
        ttl_seconds: Cache entry lifetime
        max_size: Maximum cache entries
    """
    
    def __init__(
        self,
        embedder: Optional[Callable] = None,
        similarity_threshold: float = 0.90,
        ttl_seconds: int = 3600,
        max_size: int = 10000
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embeddings: List[np.ndarray] = []
        self._keys: List[str] = []
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _compute_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        cached_embedding: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings."""
        dot = np.dot(query_embedding, cached_embedding)
        norm_q = np.linalg.norm(query_embedding)
        norm_c = np.linalg.norm(cached_embedding)
        
        if norm_q == 0 or norm_c == 0:
            return 0.0
        
        return dot / (norm_q * norm_c)
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for query.
        
        Uses semantic similarity if embedder available,
        falls back to exact match otherwise.
        
        Returns:
            Cached response or None if no match
        """
        # First try exact match
        key = self._compute_key(query)
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired(self.ttl_seconds):
                entry.hit_count += 1
                entry.last_accessed = datetime.now()
                self._cache.move_to_end(key)
                self.hits += 1
                return entry.response
            else:
                # Remove expired entry
                del self._cache[key]
        
        # Try semantic matching
        if self.embedder and self._embeddings:
            query_embedding = self.embedder(query)
            
            best_similarity = 0.0
            best_entry = None
            
            for i, cached_embedding in enumerate(self._embeddings):
                similarity = self._compute_similarity(
                    query_embedding, cached_embedding
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    cache_key = self._keys[i]
                    if cache_key in self._cache:
                        best_entry = self._cache[cache_key]
            
            if best_similarity >= self.similarity_threshold and best_entry:
                if not best_entry.is_expired(self.ttl_seconds):
                    best_entry.hit_count += 1
                    best_entry.last_accessed = datetime.now()
                    self.hits += 1
                    return best_entry.response
        
        self.misses += 1
        return None
    
    def set(self, query: str, response: str) -> None:
        """
        Cache a query-response pair.
        
        Args:
            query: The input query
            response: The generated response
        """
        # Evict if at capacity (LRU)
        while len(self._cache) >= self.max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            # Remove embedding
            if oldest_key in self._keys:
                idx = self._keys.index(oldest_key)
                self._keys.pop(idx)
                self._embeddings.pop(idx)
        
        key = self._compute_key(query)
        
        # Compute embedding if available
        query_embedding = None
        if self.embedder:
            query_embedding = self.embedder(query)
            self._embeddings.append(query_embedding)
            self._keys.append(key)
        
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding if query_embedding is not None else np.array([]),
            response=response,
            created_at=datetime.now()
        )
        
        self._cache[key] = entry
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional substring to match in queries
            
        Returns:
            Number of entries removed
        """
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            self._embeddings.clear()
            self._keys.clear()
            return count
        
        # Remove matching entries
        to_remove = []
        for key, entry in self._cache.items():
            if pattern.lower() in entry.query.lower():
                to_remove.append(key)
        
        for key in to_remove:
            del self._cache[key]
            if key in self._keys:
                idx = self._keys.index(key)
                self._keys.pop(idx)
                self._embeddings.pop(idx)
        
        return len(to_remove)
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "embeddings_count": len(self._embeddings)
        }


# ============================================================
# MODEL ROUTING
# ============================================================

class ModelRouter:
    """
    Intelligent model routing based on query complexity.
    
    Production Pattern:
    Not every query needs the most powerful (expensive) model.
    Route queries to appropriate model tiers:
    
    - Tier 1: Simple lookups, FAQs → Fast/cheap model or cache
    - Tier 2: Standard retrieval → Mid-tier model
    - Tier 3: Complex reasoning → Large model  
    - Tier 4: Specialized domain → Fine-tuned model
    
    Benefits:
    - 40-70% cost reduction
    - Lower latency for simple queries
    - Better quality for complex queries
    """
    
    def __init__(
        self,
        models: Optional[Dict[int, str]] = None,
        classifier: Optional[Callable] = None
    ):
        """
        Args:
            models: Dict mapping tier (1-4) to model name
            classifier: Optional LLM-based query classifier
        """
        self.models = models or {
            1: "gpt-3.5-turbo",       # Fast, cheap
            2: "gpt-4o-mini",           # Balanced
            3: "gpt-4o",                # Powerful
            4: "ft:gpt-4o:custom",      # Fine-tuned
        }
        self.classifier = classifier
        
        # Usage tracking
        self.usage: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    
    def classify(self, query: str) -> QueryClassification:
        """
        Classify query complexity.
        
        Uses LLM classifier if available, otherwise rule-based.
        """
        if self.classifier:
            return self._llm_classify(query)
        return self._rule_based_classify(query)
    
    def _llm_classify(self, query: str) -> QueryClassification:
        """LLM-based query classification."""
        # In production, call LLM with classification prompt
        result = self.classifier(query)
        return result
    
    def _rule_based_classify(self, query: str) -> QueryClassification:
        """Rule-based query classification."""
        query_lower = query.lower()
        words = query.split()
        
        # Tier 1: Simple queries
        simple_patterns = [
            "what is", "define", "who is", "when was",
            "how many", "list", "name"
        ]
        if any(query_lower.startswith(p) for p in simple_patterns):
            if len(words) < 8:
                return QueryClassification(
                    tier=1,
                    confidence=0.8,
                    reasoning="Simple factual query",
                    estimated_tokens=100
                )
        
        # Tier 4: Domain-specific
        domain_keywords = [
            "legal", "contract", "compliance", "regulation",
            "medical", "diagnosis", "pharmaceutical",
            "financial", "trading", "investment"
        ]
        if any(kw in query_lower for kw in domain_keywords):
            return QueryClassification(
                tier=4,
                confidence=0.7,
                reasoning="Domain-specific terminology detected",
                estimated_tokens=500
            )
        
        # Tier 3: Complex reasoning
        complex_indicators = [
            "compare", "analyze", "evaluate", "synthesize",
            "implications", "relationship between", "how does",
            "why", "explain the reasoning", "step by step"
        ]
        complex_count = sum(1 for ind in complex_indicators if ind in query_lower)
        
        if complex_count >= 2 or len(words) > 20:
            return QueryClassification(
                tier=3,
                confidence=0.75,
                reasoning="Complex reasoning required",
                estimated_tokens=800
            )
        
        # Tier 2: Standard (default)
        return QueryClassification(
            tier=2,
            confidence=0.6,
            reasoning="Standard information retrieval",
            estimated_tokens=300
        )
    
    def route(self, query: str) -> Tuple[str, QueryClassification]:
        """
        Route query to appropriate model.
        
        Returns:
            (model_name, classification)
        """
        classification = self.classify(query)
        model = self.models.get(classification.tier, self.models[2])
        
        # Track usage
        self.usage[classification.tier] = self.usage.get(classification.tier, 0) + 1
        
        return model, classification
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = sum(self.usage.values())
        distribution = {
            f"tier_{k}": v / total if total > 0 else 0 
            for k, v in self.usage.items()
        }
        
        return {
            "total_queries": total,
            "distribution": distribution,
            "usage_by_tier": self.usage.copy()
        }


# ============================================================
# COST TRACKING
# ============================================================

class CostTracker:
    """
    Token usage and cost tracking.
    
    Production Pattern:
    Track every LLM call for:
    - Billing and chargeback
    - Cost optimization analysis
    - Quota management
    - Anomaly detection
    
    Pricing (example, configure as needed):
        gpt-3.5-turbo: $0.50/$1.50 per 1M tokens (input/output)
        gpt-4o-mini:   $0.15/$0.60 per 1M tokens
        gpt-4o:        $2.50/$10.00 per 1M tokens
    """
    
    # Default pricing per 1M tokens (input, output)
    DEFAULT_PRICING = {
        "gpt-3.5-turbo": (0.50, 1.50),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "claude-3-haiku": (0.25, 1.25),
        "claude-3-sonnet": (3.00, 15.00),
        "claude-3-opus": (15.00, 75.00),
    }
    
    def __init__(self, pricing: Optional[Dict[str, Tuple[float, float]]] = None):
        self.pricing = pricing or self.DEFAULT_PRICING
        
        # Tracking data
        self.records: List[Dict[str, Any]] = []
        self.period_start = datetime.now()
    
    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        query: Optional[str] = None,
        cached: bool = False
    ) -> float:
        """
        Record a model invocation.
        
        Returns:
            Cost in USD
        """
        # Calculate cost
        input_price, output_price = self.pricing.get(model, (1.0, 3.0))
        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        
        record = {
            "timestamp": datetime.now(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost,
            "cached": cached,
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8] if query else None
        }
        
        self.records.append(record)
        
        return cost
    
    def get_report(
        self, 
        since: Optional[datetime] = None
    ) -> CostReport:
        """
        Generate cost report.
        
        Args:
            since: Start time for report (default: period start)
        """
        since = since or self.period_start
        
        # Filter records
        filtered = [r for r in self.records if r["timestamp"] >= since]
        
        # Aggregate
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0.0
        cache_hits = 0
        model_usage: Dict[str, int] = {}
        
        for record in filtered:
            total_tokens += record["total_tokens"]
            prompt_tokens += record["prompt_tokens"]
            completion_tokens += record["completion_tokens"]
            total_cost += record["cost_usd"]
            
            if record["cached"]:
                cache_hits += 1
            
            model = record["model"]
            model_usage[model] = model_usage.get(model, 0) + 1
        
        queries = len(filtered)
        hit_rate = cache_hits / queries if queries > 0 else 0.0
        
        return CostReport(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=total_cost,
            queries_processed=queries,
            cache_hits=cache_hits,
            cache_hit_rate=hit_rate,
            model_usage=model_usage,
            period_start=since,
            period_end=datetime.now()
        )
    
    def reset_period(self) -> CostReport:
        """Reset tracking period and return final report."""
        report = self.get_report()
        self.records.clear()
        self.period_start = datetime.now()
        return report


# ============================================================
# UNIFIED COST OPTIMIZATION LAYER
# ============================================================

class CostOptimizer:
    """
    Unified cost optimization layer.
    
    Combines caching, routing, and tracking for production use.
    
    Example:
        optimizer = CostOptimizer(embedder=embed_fn)
        
        # Before calling LLM
        cached = optimizer.get_cached(query)
        if cached:
            return cached
        
        # Route to appropriate model
        model, classification = optimizer.route(query)
        
        # After LLM call  
        optimizer.cache_and_track(query, response, model, tokens)
    """
    
    def __init__(
        self,
        embedder: Optional[Callable] = None,
        models: Optional[Dict[int, str]] = None,
        cache_threshold: float = 0.90,
        cache_ttl: int = 3600
    ):
        self.cache = SemanticCache(
            embedder=embedder,
            similarity_threshold=cache_threshold,
            ttl_seconds=cache_ttl
        )
        self.router = ModelRouter(models=models)
        self.tracker = CostTracker()
    
    def get_cached(self, query: str) -> Optional[str]:
        """Check cache for query."""
        return self.cache.get(query)
    
    def route(self, query: str) -> Tuple[str, QueryClassification]:
        """Route query to model."""
        return self.router.route(query)
    
    def cache_and_track(
        self,
        query: str,
        response: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_response: bool = True
    ) -> float:
        """
        Cache response and track cost.
        
        Returns:
            Cost in USD
        """
        if cache_response:
            self.cache.set(query, response)
        
        return self.tracker.record(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            query=query,
            cached=False
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats."""
        return {
            "cache": self.cache.get_stats(),
            "routing": self.router.get_usage_stats(),
            "cost": self.tracker.get_report().__dict__
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Models
    "CacheEntry",
    "QueryClassification",
    "CostReport",
    # Components
    "SemanticCache",
    "ModelRouter",
    "CostTracker",
    # Unified
    "CostOptimizer",
]