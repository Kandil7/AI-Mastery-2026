# Theory 5: Production Considerations

## 5.1 Latency Optimization

### 5.1.1 Parallel Retrieval

```python
async def parallel_retrieve(query: str, indexes: List[str]) -> Dict[str, List[dict]]:
    """Execute retrieval across multiple indexes in parallel."""
    tasks = {idx: search_index(idx, query) for idx in indexes}
    return await asyncio.gather(*tasks.values())
```

### 5.1.2 Caching Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    CACHING ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Query ──▶ [L1: Query Cache] ──▶ Cache Hit? ──Yes─▶ Return│
│       │                              │                      │
│       │                              No                     │
│       ▼                              │                      │
│   [L2: Embedding Cache]             │                      │
│       │                              │                      │
│       ▼                              │                      │
│   [L3: Result Cache] ───────────────┘                      │
│       │                                                     │
│       ▼                                                     │
│   [Vector DB Search]                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.1.3 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| P50 Latency | <100ms | Median response time |
| P95 Latency | <500ms | 95th percentile |
| P99 Latency | <1000ms | 99th percentile |
| Cache Hit Rate | >60% | Query cache effectiveness |
| Throughput | >1000 QPS | Queries per second |

## 5.2 Monitoring & Observability

### 5.2.1 Key Metrics

```python
from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class RetrievalMetrics:
    query_id: str
    latency_ms: float
    indexes_searched: int
    results_returned: int
    cache_hit: bool
    modality_distribution: Dict[str, int]
    rerank_applied: bool

class MetricsCollector:
    def __init__(self):
        self.metrics_log = []
    
    def record(self, metrics: RetrievalMetrics):
        self.metrics_log.append(metrics)
        # Also send to monitoring system
        self._send_to_prometheus(metrics)
    
    def get_summary(self, window_hours: int = 24) -> Dict:
        recent = [m for m in self.metrics_log 
                  if time.time() - m.timestamp < window_hours * 3600]
        
        return {
            'avg_latency': sum(m.latency_ms for m in recent) / len(recent),
            'cache_hit_rate': sum(m.cache_hit for m in recent) / len(recent),
            'avg_results': sum(m.results_returned for m in recent) / len(recent),
        }
```

### 5.2.2 Tracing

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("adaptive_retrieval")
async def retrieve_with_tracing(query: str):
    # Span automatically tracks timing and context
    return await adaptive_retrieve(query)
```

## 5.3 Quality Assurance

### 5.3.1 A/B Testing Framework

```python
class ABTestRouter:
    """Route queries to different retrieval strategies for A/B testing."""
    
    def __init__(self, experiment_config: dict):
        self.config = experiment_config
        self.results = {'control': [], 'treatment': []}
    
    def get_variant(self, query_id: str) -> str:
        """Deterministically assign query to variant."""
        hash_val = hash(query_id) % 100
        if hash_val < self.config['treatment_percentage']:
            return 'treatment'
        return 'control'
    
    def record_feedback(self, variant: str, relevance_score: float):
        """Record user feedback for analysis."""
        self.results[variant].append(relevance_score)
    
    def get_results(self) -> dict:
        """Calculate statistical significance."""
        from scipy import stats
        
        control = self.results['control']
        treatment = self.results['treatment']
        
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        return {
            'control_mean': np.mean(control),
            'treatment_mean': np.mean(treatment),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

### 5.3.2 Quality Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Recall@K | Relevant in top K / Total relevant | >0.85 |
| Precision@K | Relevant in top K / K | >0.70 |
| MRR | Average of 1/rank of first relevant | >0.75 |
| NDCG@K | Normalized discounted cumulative gain | >0.80 |

## 5.4 Scaling Considerations

### 5.4.1 Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────┐
│                 HORIZONTAL SCALING PATTERN                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Load Balancer                                             │
│       │                                                     │
│   ┌───┴───┬─────────┬─────────┐                            │
│   ▼       ▼         ▼         ▼                            │
│ [Node 1] [Node 2] [Node 3] [Node N]                        │
│   │       │         │         │                            │
│   └───┬───┴─────────┴─────────┘                            │
│       │                                                     │
│   [Shared Vector DB Cluster]                                │
│       │                                                     │
│   [Shared Cache (Redis Cluster)]                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4.2 Index Partitioning

```python
class PartitionedIndex:
    """Partition index by domain/modality for scalability."""
    
    def __init__(self, partition_key: str, num_partitions: int):
        self.partition_key = partition_key
        self.num_partitions = num_partitions
        self.partitions = [VectorIndex() for _ in range(num_partitions)]
    
    def _get_partition(self, doc_id: str) -> int:
        return hash(doc_id) % self.num_partitions
    
    def add(self, doc_id: str, embedding: np.ndarray, metadata: dict):
        partition = self._get_partition(doc_id)
        self.partitions[partition].add(doc_id, embedding, metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[dict]:
        # Search all partitions and merge
        all_results = []
        for partition in self.partitions:
            results = partition.search(query_embedding, top_k)
            all_results.extend(results)
        
        return sorted(all_results, key=lambda x: x['score'], reverse=True)[:top_k]
```

## 5.5 Security Considerations

### 5.5.1 Access Control

```python
from functools import wraps

def require_permission(permission: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('user')
            if not user.has_permission(permission):
                raise PermissionError(f"Missing {permission} permission")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@require_permission('read_multimodal')
async def retrieve_multimodal(query: str, user):
    return await adaptive_retrieve(query)
```

### 5.5.2 Content Filtering

```python
class ContentFilter:
    """Filter inappropriate or sensitive content from results."""
    
    def __init__(self, sensitivity_level: str = 'medium'):
        self.sensitivity_level = sensitivity_level
        self.filtered_terms = self._load_filtered_terms()
    
    def filter_results(self, results: List[dict]) -> List[dict]:
        filtered = []
        for doc in results:
            if self._is_safe(doc):
                filtered.append(doc)
        return filtered
    
    def _is_safe(self, doc: dict) -> bool:
        content = doc.get('content', '')
        return not any(term in content.lower() for term in self.filtered_terms)
```

## 5.6 Summary

Production considerations:
- Optimize latency with parallel retrieval and caching
- Monitor key metrics for quality and performance
- Implement A/B testing for continuous improvement
- Scale horizontally with partitioning
- Apply security controls for access and content
