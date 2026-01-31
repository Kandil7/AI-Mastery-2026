# Multi-Layer Cache Implementation Guide

## Overview

This document provides a comprehensive guide to the multi-layer caching functionality implementation in the RAG Engine Mini. The multi-layer cache system provides hierarchical caching with L1 (in-memory), L2 (Redis), and L3 (persistent) layers, which was marked as pending in the project completion checklist.

## Architecture

### Component Structure

The multi-layer caching functionality follows the same architectural patterns as the rest of the RAG Engine:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer     │────│  Application     │────│   Domain/       │
│   (routes)      │    │  Services/       │    │   Ports/        │
│                 │    │  Use Cases       │    │   Adapters      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │ HTTP Requests          │ Business Logic        │ Interfaces &
         │                        │                       │ Implementations
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Multi-Layer    │    │ MultiLayer      │
│   Endpoints     │    │  Cache Service  │    │ Cache Port      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **Multi-Layer Cache Service** (`src/application/services/multi_layer_cache_service.py`): Core caching logic
2. **Cache Layers**: Three-tier cache system (L1, L2, L3)
3. **Cache Entry**: Data structure for cached values with metadata
4. **Dependency Injection**: Service registration and wiring

## Implementation Details

### 1. Multi-Layer Cache Service

The `MultiLayerCacheService` implements the `MultiLayerCachePort` interface and provides:

- **Three-Tier Hierarchy**: L1 (memory), L2 (Redis), L3 (persistent) caching
- **Automatic Promotion**: Values move up the hierarchy when accessed
- **Size Management**: Eviction policies for memory constraints
- **TTL Support**: Time-to-live for cache entries
- **Batch Operations**: Bulk loading and invalidation
- **Statistics**: Performance metrics for each layer

Key methods:
```python
async def get(key: str, layer: Optional[CacheLayer] = None) -> Optional[Any]
async def set(key: str, value: Any, ttl: Optional[timedelta] = None, target_layers: Optional[List[CacheLayer]] = None) -> bool
async def delete(key: str) -> bool
async def invalidate_by_prefix(prefix: str) -> int
async def get_stats() -> Dict[CacheLayer, Dict[str, Any]]
async def warm_up(data: Dict[str, Any], ttl: Optional[timedelta] = None) -> bool
async def clear_layer(layer: CacheLayer) -> bool
```

### 2. Cache Layers

The system implements three distinct cache layers:

#### L1: Memory Cache
- **Purpose**: Fastest access for frequently used data
- **Technology**: Python dictionary with size limits
- **Characteristics**: 
  - Sub-millisecond access times
  - Limited by available memory
  - LRU-style eviction policy

#### L2: Redis Cache
- **Purpose**: Shared cache across application instances
- **Technology**: Redis server
- **Characteristics**:
  - Millisecond access times
  - Shared across multiple processes/servers
  - Persistence options available

#### L3: Persistent Cache
- **Purpose**: Long-term storage for expensive computations
- **Technology**: Database or file system
- **Characteristics**:
  - Slower access times but reliable
  - Survives application restarts
  - Suitable for infrequently accessed data

### 3. Cache Entry Structure

Each cached value includes metadata:

```python
@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    layer: Optional[CacheLayer] = None
    size_bytes: Optional[int] = None
```

## API Usage

### Setting Values in Cache

```python
from src.application.services.multi_layer_cache_service import MultiLayerCacheService, CacheLayer
from datetime import timedelta

cache_service = MultiLayerCacheService(redis_client=redis_conn)

# Set a value in all layers with 1-hour TTL
await cache_service.set(
    "document_embedding:123", 
    embedding_vector, 
    ttl=timedelta(hours=1)
)

# Set a value in specific layers only
await cache_service.set(
    "query_result:456", 
    result_data, 
    ttl=timedelta(minutes=30),
    target_layers=[CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS]
)
```

### Getting Values from Cache

```python
# Get from any layer (follows hierarchy: L1 -> L2 -> L3)
value = await cache_service.get("document_embedding:123")

# Get from specific layer only
value = await cache_service.get("document_embedding:123", layer=CacheLayer.L1_MEMORY)
```

### Cache Statistics

```python
# Get statistics for all cache layers
stats = await cache_service.get_stats()

for layer, layer_stats in stats.items():
    print(f"{layer.value}:")
    print(f"  Size: {layer_stats['size']}")
    print(f"  Hit Rate: {layer_stats['hit_rate']:.2%}")
    print(f"  Memory Usage: {layer_stats['memory_usage_bytes']} bytes")
```

### Bulk Operations

```python
# Warm up cache with multiple values
data = {
    "doc_1": {"title": "Document 1", "content": "..."},
    "doc_2": {"title": "Document 2", "content": "..."},
    # ... more documents
}

await cache_service.warm_up(data, ttl=timedelta(hours=2))

# Invalidate all keys with a prefix
invalidated_count = await cache_service.invalidate_by_prefix("category:tech:")
```

## Integration Points

### Dependency Injection

The cache service can be integrated into the existing DI container:

```python
# In the bootstrap file
multi_layer_cache = MultiLayerCacheService(
    redis_client=redis_cache,  # From existing Redis adapter
    persistent_store=db_repo   # From existing DB repository
)

return {
    # ... other services
    "multi_layer_cache": multi_layer_cache,
}
```

## Use Cases in RAG Systems

Multi-layer caching is essential for RAG systems:

1. **Embedding Caching**: Store computed embeddings to avoid recomputation
2. **Document Chunk Caching**: Cache processed document segments
3. **Query Result Caching**: Store frequent query results
4. **LLM Response Caching**: Cache deterministic LLM outputs
5. **Search Result Caching**: Store ranked document lists
6. **Precomputed Aggregations**: Cache expensive analytical queries

## Performance Considerations

1. **L1 Cache**: Keep frequently accessed small objects (e.g., embeddings)
2. **L2 Cache**: Share common data across application instances
3. **L3 Cache**: Store expensive-to-compute results with longer TTLs
4. **Size Limits**: Monitor and tune cache sizes based on available resources
5. **Hit Rates**: Aim for >80% hit rates for effective caching

## Memory Management

The L1 cache implements automatic memory management:

- **Size Tracking**: Approximates object sizes to enforce limits
- **Eviction Policy**: Removes oldest entries when size limits are reached
- **Expiration**: Automatically removes expired entries

## Security Considerations

1. **Data Sensitivity**: Cache only non-sensitive data or encrypt sensitive data
2. **Namespace Isolation**: Use prefixes to separate tenants/data types
3. **Access Controls**: Ensure cache access respects application permissions
4. **Injection Prevention**: Sanitize keys to prevent cache poisoning

## Educational Value

This implementation demonstrates:

1. **Clean Architecture**: Clear separation of concerns
2. **Port/Adapter Pattern**: Interface-based design
3. **Performance Optimization**: Hierarchical caching strategies
4. **Resource Management**: Memory and size constraints
5. **Real-world Application**: Practical caching for RAG systems

## Testing

The multi-layer cache functionality includes comprehensive tests in `tests/unit/test_multi_layer_cache_service.py`:

- Cache hierarchy behavior
- Size management and eviction
- TTL handling and expiration
- Bulk operations
- Statistics tracking
- Error handling

## Conclusion

The multi-layer cache functionality completes a critical feature that was marked as pending in the project completion checklist. It follows the same architectural patterns as the rest of the RAG Engine Mini, ensuring consistency and maintainability. The implementation provides comprehensive tools for managing data across multiple storage tiers, dramatically improving performance for expensive operations in RAG systems.

This addition brings the RAG Engine Mini significantly closer to full completion, providing users with essential performance optimizations for production deployments.