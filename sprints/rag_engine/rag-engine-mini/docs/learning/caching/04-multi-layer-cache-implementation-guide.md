# Multi-Layer Caching Implementation Guide

## Overview

This document provides a comprehensive guide to the multi-layer caching functionality implementation in the RAG Engine Mini. The multi-layer cache system provides hierarchical caching with L1 (in-memory), L2 (Redis), and L3 (persistent) layers to optimize performance across different data access patterns, which was marked as pending in the project completion checklist.

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
│   FastAPI       │    │  Multi-Layer    │    │ MultiLayerCache │
│   Endpoints     │    │  Cache Service  │    │ Port Interface  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **Multi-Layer Cache Service** (`src/application/services/multi_layer_cache_service.py`): Core caching logic
2. **Cache Layers**: L1 (memory), L2 (Redis), L3 (persistent) cache layers
3. **Dependency Injection** (`src/core/bootstrap.py`): Service registration and wiring

## Implementation Details

### 1. Multi-Layer Cache Service

The `MultiLayerCacheService` implements the `MultiLayerCachePort` interface and provides:

- **Hierarchical Caching**: Three-tier cache hierarchy (L1, L2, L3)
- **Cache Operations**: Get, set, delete, invalidate operations
- **Statistics Tracking**: Hit rates, memory usage, and performance metrics
- **Cache Warming**: Batch loading of initial data
- **Size Management**: Eviction policies and size limits

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

#### L1 (Memory Cache)
- **Location**: In-application memory
- **Speed**: Fastest access (< 1μs)
- **Size**: Limited by available RAM (default: 10MB, 1000 items)
- **Persistence**: Volatile, lost on restart
- **Use Case**: Frequently accessed embeddings, prompts, query results

#### L2 (Redis Cache)
- **Location**: External Redis server
- **Speed**: Fast access (~1ms)
- **Size**: Limited by Redis configuration
- **Persistence**: Volatile with optional persistence
- **Use Case**: Shared cache across application instances

#### L3 (Persistent Cache)
- **Location**: Database, file system, or other persistent store
- **Speed**: Slower access (~10ms+)
- **Size**: Virtually unlimited
- **Persistence**: Persistent across restarts
- **Use Case**: Long-term storage of computed results

### 3. Cache Entry Model

The `CacheEntry` dataclass contains metadata for cached items:
- Key and value
- Creation and expiration timestamps
- Size in bytes
- Origin cache layer

## API Usage

### Basic Operations

```python
from src.application.services.multi_layer_cache_service import MultiLayerCacheService, CacheLayer

cache_service = MultiLayerCacheService(redis_client=your_redis_client)

# Set a value in all layers with TTL
await cache_service.set("my_key", "my_value", ttl=timedelta(hours=1))

# Get a value (will check L1 → L2 → L3 hierarchy)
value = await cache_service.get("my_key")

# Get from specific layer only
value = await cache_service.get("my_key", layer=CacheLayer.L1_MEMORY)

# Delete from all layers
await cache_service.delete("my_key")
```

### Advanced Operations

```python
# Invalidate all keys with a prefix
invalidated_count = await cache_service.invalidate_by_prefix("user:session:")

# Warm up cache with batch data
initial_data = {
    "embedding:doc1": [0.1, 0.2, 0.3],
    "prompt:qa": "Answer based on context..."
}
await cache_service.warm_up(initial_data, ttl=timedelta(hours=24))

# Get cache statistics
stats = await cache_service.get_stats()
print(f"L1 Hit Rate: {stats[CacheLayer.L1_MEMORY]['hit_rate']}")

# Clear a specific layer
await cache_service.clear_layer(CacheLayer.L1_MEMORY)
```

## Integration Points

### Dependency Injection

The multi-layer cache service would be registered in `src/core/bootstrap.py`:

```python
# Initialize cache with Redis client
cache = MultiLayerCacheService(redis_client=redis_client)

return {
    # ... other services
    "cache": cache,  # This replaces the original Redis cache
}
```

## Use Cases in RAG Systems

Multi-layer caching is essential for RAG performance:

1. **Embedding Caching**: Store computed embeddings to avoid recomputation
2. **Query Results**: Cache frequent query responses
3. **Prompt Templates**: Store commonly used prompt structures
4. **Chunk Summaries**: Cache document summaries and metadata
5. **Entity Extraction**: Store extracted entities and relationships
6. **LLM Responses**: Cache deterministic LLM outputs

## Performance Considerations

1. **Tiered Access**: Fastest access for hottest data (L1), slower for cold data (L3)
2. **Memory Management**: Eviction policies prevent memory overflow
3. **Serialization Overhead**: Efficient serialization between layers
4. **Network Latency**: Minimize cross-network cache access where possible
5. **Cache Hit Ratios**: Monitor and optimize hit rates for each tier

## Eviction Policies

### L1 Memory Cache
- **Size-based Eviction**: Maintains max item count and byte size
- **LRU-like Behavior**: Removes oldest entries when limits exceeded
- **TTL Enforcement**: Automatically removes expired entries

### L2 Redis Cache
- **Redis Policies**: Uses Redis's built-in eviction policies
- **TTL Propagation**: TTLs applied at set time
- **Memory Optimization**: Uses Redis's memory-efficient structures

### L3 Persistent Cache
- **Application Managed**: Custom eviction based on access patterns
- **Size Limits**: Configurable retention policies
- **TTL Handling**: Time-based cleanup jobs

## Security Considerations

1. **Data Isolation**: Proper separation between tenants in shared caches
2. **Cache Poisoning**: Validate data before caching
3. **Key Sanitization**: Prevent special characters in cache keys
4. **Access Control**: Restrict direct cache access in multi-tenant systems
5. **Encryption**: Encrypt sensitive cached data when necessary

## Educational Value

This implementation demonstrates:

1. **Clean Architecture**: Clear separation of concerns
2. **Performance Optimization**: Multi-tier caching strategy
3. **Resource Management**: Memory and size constraints
4. **Scalability**: Hierarchical caching for different access patterns
5. **System Design**: Trade-offs between speed, cost, and persistence

## Testing

The multi-layer caching functionality includes comprehensive tests in `tests/unit/test_multi_layer_cache_service.py`:

- Cache hierarchy operations
- TTL enforcement
- Size limit management
- Cross-layer promotion
- Statistics tracking
- Error handling

## Conclusion

The multi-layer caching functionality completes a critical feature that was marked as pending in the project completion checklist. It follows the same architectural patterns as the rest of the RAG Engine Mini, ensuring consistency and maintainability. The implementation provides comprehensive tools for managing hierarchical caching in RAG applications, optimizing performance across different data access patterns.

This addition brings the RAG Engine Mini significantly closer to full completion, providing users with the ability to achieve optimal performance through strategic caching across multiple storage tiers.