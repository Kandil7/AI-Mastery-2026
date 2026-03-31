# Coding Challenges - Module 1: Adaptive Multimodal RAG

## Challenge Overview

Complete these coding challenges to demonstrate your mastery of adaptive multimodal RAG concepts. Challenges are organized by difficulty level.

---

## 🟢 Easy Challenge: Modality-Aware Query Router

### Problem

Build a simple query router that routes queries to appropriate indexes based on detected modality.

### Requirements

1. Implement a `SimpleModalityRouter` class with:
   - A method to detect query modality (text, image, code, table)
   - A method to route queries to appropriate index
   - Support for at least 4 different indexes

2. The router should:
   - Detect modality from query keywords
   - Return the appropriate index name
   - Include a confidence score

### Starter Code

```python
from enum import Enum
from typing import Tuple

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    TABLE = "table"

class SimpleModalityRouter:
    def __init__(self):
        # Define your keyword mappings here
        pass
    
    def detect_modality(self, query: str) -> Tuple[ModalityType, float]:
        """
        Detect modality from query.
        
        Returns:
            Tuple of (modality, confidence)
        """
        # Your implementation here
        pass
    
    def route(self, query: str) -> dict:
        """
        Route query to appropriate index.
        
        Returns:
            Dict with 'index', 'modality', 'confidence'
        """
        # Your implementation here
        pass

# Test your implementation
if __name__ == "__main__":
    router = SimpleModalityRouter()
    
    test_queries = [
        "Show me the diagram",
        "Write a function",
        "Display the data table",
        "Explain the concept"
    ]
    
    for query in test_queries:
        result = router.route(query)
        print(f"Query: {query}")
        print(f"  Index: {result['index']}")
        print(f"  Modality: {result['modality']}")
        print(f"  Confidence: {result['confidence']}")
```

### Expected Output

```
Query: Show me the diagram
  Index: image_index
  Modality: image
  Confidence: 0.85

Query: Write a function
  Index: code_index
  Modality: code
  Confidence: 0.80

Query: Display the data table
  Index: table_index
  Modality: table
  Confidence: 0.90

Query: Explain the concept
  Index: text_index
  Modality: text
  Confidence: 0.75
```

### Evaluation Criteria

- ✅ Correctly detects all 4 modalities
- ✅ Returns appropriate index names
- ✅ Confidence scores are reasonable (0.5-1.0)
- ✅ Code is clean and well-documented

---

## 🟡 Medium Challenge: Adaptive Retrieval with RRF Fusion

### Problem

Build an adaptive retrieval system that:
1. Routes queries to multiple indexes based on analysis
2. Retrieves results from each index
3. Fuses results using Reciprocal Rank Fusion

### Requirements

1. Implement an `AdaptiveRetriever` class with:
   - Query analysis (modality, intent, complexity)
   - Multi-index retrieval (mock data is fine)
   - RRF-based fusion
   - Final ranking

2. The system should:
   - Analyze queries to determine which indexes to search
   - Retrieve mock results from each selected index
   - Apply RRF to combine results
   - Return top-K fused results

### Starter Code

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class SearchResult:
    id: str
    content: str
    score: float
    modality: str
    index: str

class AdaptiveRetriever:
    def __init__(self):
        # Initialize mock indexes with sample data
        self.indexes = self._create_mock_indexes()
    
    def _create_mock_indexes(self) -> Dict[str, List[SearchResult]]:
        """Create mock indexes with sample data."""
        return {
            'text_index': [
                SearchResult('t1', 'Text document 1', 0.9, 'text', 'text_index'),
                SearchResult('t2', 'Text document 2', 0.8, 'text', 'text_index'),
            ],
            'image_index': [
                SearchResult('i1', 'Architecture diagram', 0.85, 'image', 'image_index'),
                SearchResult('i2', 'Flow chart', 0.75, 'image', 'image_index'),
            ],
            'code_index': [
                SearchResult('c1', 'Python function', 0.88, 'code', 'code_index'),
                SearchResult('c2', 'Java class', 0.72, 'code', 'code_index'),
            ],
            'table_index': [
                SearchResult('tb1', 'Metrics table', 0.82, 'table', 'table_index'),
            ]
        }
    
    def analyze_query(self, query: str) -> List[str]:
        """
        Analyze query and determine which indexes to search.
        
        Returns:
            List of index names to search
        """
        # Your implementation here
        pass
    
    def retrieve_from_index(self, index: str, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve results from a specific index.
        
        Returns:
            List of search results
        """
        # Your implementation here
        pass
    
    def reciprocal_rank_fusion(self, results_by_index: Dict[str, List[SearchResult]], 
                                k: int = 60) -> List[SearchResult]:
        """
        Apply RRF to combine results from multiple indexes.
        
        Returns:
            Fused and ranked results
        """
        # Your implementation here
        pass
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Main retrieval method.
        
        Returns:
            Top-K fused results
        """
        # Your implementation here
        pass

# Test your implementation
if __name__ == "__main__":
    retriever = AdaptiveRetriever()
    
    queries = [
        "Show me the architecture diagram",
        "Write a function to calculate sum",
        "Compare the metrics",
        "Explain the system"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for i, result in enumerate(results):
            print(f"  {i+1}. [{result.modality}] {result.content} (score: {result.score:.3f})")
```

### Evaluation Criteria

- ✅ Query analysis correctly selects indexes
- ✅ RRF implementation is correct
- ✅ Results are properly fused and ranked
- ✅ System handles edge cases (no results, single index)
- ✅ Code is modular and testable

---

## 🔴 Hard Challenge: Production-Ready Adaptive Multimodal RAG

### Problem

Build a production-ready adaptive multimodal RAG system with:
1. Complete query analysis pipeline
2. Hybrid routing (rules + ML-based)
3. Multi-modal retrieval with fusion
4. Caching and metrics collection
5. Comprehensive error handling

### Requirements

1. Implement a `ProductionRAGSystem` class with:
   - **Query Analysis**: Modality, intent, complexity, domain detection
   - **Hybrid Router**: Rule-based + ML-based routing with confidence thresholds
   - **Multi-Modal Retrieval**: Parallel retrieval from multiple indexes
   - **Fusion Pipeline**: RRF + cross-modal scoring + diversity optimization
   - **Caching**: Query and embedding cache with Redis-like interface
   - **Metrics**: Latency, cache hit rate, result quality tracking

2. The system must:
   - Handle concurrent requests (async/await)
   - Implement proper error handling and fallbacks
   - Log all routing decisions for debugging
   - Support configuration via dataclass
   - Include unit tests

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ProductionRAGSystem                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query ──▶ [Analysis] ──▶ [Hybrid Router] ──▶ [Retrieval]  │
│              │                 │                  │         │
│              ▼                 ▼                  ▼         │
│        [Modality]        [Rules + ML]      [Parallel]       │
│        [Intent]          [Confidence]      [Indexes]        │
│        [Complexity]                         │                │
│        [Domain]                             ▼                │
│                                        [Fusion]              │
│                                            │                 │
│                                            ▼                 │
│                                      [Results]               │
│                                                             │
│  [Cache Layer] ◀────────────────────────────────▶ [Metrics] │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Starter Code

```python
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    TABLE = "table"
    MIXED = "mixed"

class QueryIntent(Enum):
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"

@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    rrf_k: int = 60
    diversity_factor: float = 0.3
    ml_confidence_threshold: float = 0.7
    cache_ttl_seconds: int = 3600
    max_concurrent_retrievals: int = 10
    top_k_default: int = 10
    enable_cross_modal_scoring: bool = True
    enable_diversity_optimization: bool = True

@dataclass
class QueryContext:
    """Parsed query context."""
    query: str
    modality: ModalityType
    modality_confidence: float
    intent: QueryIntent
    intent_confidence: float
    complexity: float
    domain: str
    entities: List[str]

@dataclass
class RetrievalMetrics:
    """Metrics for a retrieval operation."""
    query_id: str
    latency_ms: float
    indexes_searched: int
    results_returned: int
    cache_hit: bool
    routing_rules_applied: List[str]

class ProductionRAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.cache = {}  # Simple in-memory cache (use Redis in production)
        self.metrics_log = []
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all system components."""
        # Your implementation here
        pass
    
    async def analyze_query(self, query: str) -> QueryContext:
        """
        Complete query analysis pipeline.
        
        Returns:
            QueryContext with all analysis results
        """
        # Your implementation here
        pass
    
    async def route_query(self, context: QueryContext) -> Dict[str, Any]:
        """
        Hybrid routing: rules + ML with confidence threshold.
        
        Returns:
            Routing decision with indexes and parameters
        """
        # Your implementation here
        pass
    
    async def retrieve_from_indexes(self, query: str, 
                                     routing: Dict[str, Any]) -> Dict[str, List[dict]]:
        """
        Parallel retrieval from multiple indexes.
        
        Returns:
            Results grouped by index
        """
        # Your implementation here
        pass
    
    async def fuse_results(self, results_by_index: Dict[str, List[dict]], 
                           context: QueryContext) -> List[dict]:
        """
        Complete fusion pipeline: RRF + cross-modal + diversity.
        
        Returns:
            Fused and ranked results
        """
        # Your implementation here
        pass
    
    async def retrieve(self, query: str, 
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main retrieval entry point.
        
        Args:
            query: User query
            user_id: Optional user ID for personalization
            
        Returns:
            Complete retrieval result with metadata
        """
        start_time = time.time()
        query_id = f"{user_id}_{int(time.time())}" if user_id else str(int(time.time()))
        
        # Check cache
        cache_key = f"query:{hash(query)}"
        if cache_key in self.cache:
            # Cache hit
            cached_result = self.cache[cache_key]
            self._record_metrics(RetrievalMetrics(
                query_id=query_id,
                latency_ms=(time.time() - start_time) * 1000,
                indexes_searched=0,
                results_returned=len(cached_result.get('results', [])),
                cache_hit=True,
                routing_rules_applied=[]
            ))
            return cached_result
        
        # Cache miss - full pipeline
        try:
            # Step 1: Analyze query
            context = await self.analyze_query(query)
            
            # Step 2: Route query
            routing = await self.route_query(context)
            
            # Step 3: Retrieve from indexes
            results_by_index = await self.retrieve_from_indexes(query, routing)
            
            # Step 4: Fuse results
            fused_results = await self.fuse_results(results_by_index, context)
            
            # Build response
            result = {
                'query': query,
                'query_id': query_id,
                'context': {
                    'modality': context.modality.value,
                    'intent': context.intent.value,
                    'complexity': context.complexity,
                },
                'routing': routing,
                'results': fused_results[:self.config.top_k_default],
                'metadata': {
                    'total_results': len(fused_results),
                    'indexes_searched': len(results_by_index),
                }
            }
            
            # Cache result
            self.cache[cache_key] = result
            
            # Record metrics
            self._record_metrics(RetrievalMetrics(
                query_id=query_id,
                latency_ms=(time.time() - start_time) * 1000,
                indexes_searched=len(results_by_index),
                results_returned=len(fused_results),
                cache_hit=False,
                routing_rules_applied=routing.get('applied_rules', [])
            ))
            
            return result
            
        except Exception as e:
            # Fallback on error
            return self._fallback_response(query, str(e))
    
    def _record_metrics(self, metrics: RetrievalMetrics):
        """Record retrieval metrics."""
        self.metrics_log.append(metrics)
    
    def _fallback_response(self, query: str, error: str) -> dict:
        """Return fallback response on error."""
        return {
            'query': query,
            'error': error,
            'results': [],
            'fallback': True
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of retrieval metrics."""
        if not self.metrics_log:
            return {'error': 'No metrics recorded'}
        
        return {
            'total_queries': len(self.metrics_log),
            'avg_latency_ms': sum(m.latency_ms for m in self.metrics_log) / len(self.metrics_log),
            'cache_hit_rate': sum(m.cache_hit for m in self.metrics_log) / len(self.metrics_log),
            'avg_results': sum(m.results_returned for m in self.metrics_log) / len(self.metrics_log),
        }

# Test your implementation
async def test_production_system():
    config = RAGConfig(
        rrf_k=60,
        diversity_factor=0.3,
        ml_confidence_threshold=0.7,
        top_k_default=5
    )
    
    system = ProductionRAGSystem(config)
    
    test_queries = [
        "Show me the architecture diagram",
        "Write a function to calculate sum",
        "Compare Q1 and Q2 revenue metrics",
        "Fix the database connection error",
        "Explain how the system works",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        
        result = await system.retrieve(query, user_id="test_user")
        
        if result.get('fallback'):
            print(f"Error: {result.get('error')}")
            continue
        
        print(f"Modality: {result['context']['modality']}")
        print(f"Intent: {result['context']['intent']}")
        print(f"Indexes: {result['routing'].get('target_indexes', [])}")
        print(f"Results: {len(result['results'])}")
        
        for i, doc in enumerate(result['results'][:3]):
            print(f"  {i+1}. [{doc.get('modality', 'unknown')}] {doc.get('content', '')[:50]}...")
    
    print(f"\n{'='*60}")
    print("Metrics Summary:")
    print("=" * 60)
    metrics = system.get_metrics_summary()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_production_system())
```

### Evaluation Criteria

- ✅ Complete query analysis pipeline
- ✅ Hybrid routing with confidence thresholds
- ✅ Parallel retrieval implementation
- ✅ RRF + cross-modal + diversity fusion
- ✅ Caching with TTL
- ✅ Metrics collection and reporting
- ✅ Error handling and fallbacks
- ✅ Async/await for concurrency
- ✅ Clean, production-ready code structure
- ✅ Comprehensive documentation

### Bonus Points

- [ ] Add unit tests for all components
- [ ] Implement Redis-based caching
- [ ] Add OpenTelemetry tracing
- [ ] Implement query expansion for complex queries
- [ ] Add personalization based on user history
- [ ] Implement A/B testing framework

---

## Submission Guidelines

1. Create a file named `challenge_solution.py` with your complete implementation
2. Include docstrings for all public methods
3. Include example usage and test cases
4. Submit via the designated channel

## Grading Rubric

| Criterion | Easy | Medium | Hard |
|-----------|------|--------|------|
| Functionality | 30% | 25% | 20% |
| Code Quality | 30% | 25% | 20% |
| Completeness | 20% | 25% | 20% |
| Documentation | 10% | 15% | 15% |
| Testing | 10% | 10% | 15% |
| Bonus | - | - | 10% |

---

*Good luck! Reach out if you need clarification on any requirements.*
