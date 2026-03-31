---

# Case Study 27: Real-Time Recommender System - Multi-Model Database Architecture

## Executive Summary

**Problem**: Deliver personalized recommendations to 450M+ users with sub-100ms latency while handling 100K+ requests/sec and maintaining 99.99% availability.

**Solution**: Implemented multi-model architecture using PostgreSQL for metadata, Cassandra for user activity, Neo4j for relationship graphs, and Qdrant for embedding-based recommendations.

**Impact**: Achieved 95%+ click-through rate, 22% improvement in engagement, sub-50ms P99 latency, and 35% reduction in infrastructure costs vs previous architecture.

**System design snapshot**:
- SLOs: p99 <100ms; 99.99% availability; CTR >90%
- Scale: 450M+ users, 10B+ items, 100K+ QPS at peak
- Cost efficiency: 35% reduction in infrastructure costs
- Data quality: Real-time feedback loop for recommendation optimization
- Reliability: Multi-region deployment with automatic failover

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PostgreSQL     │    │   Cassandra     │    │   Neo4j Graph   │
│  • Item metadata │    │  • User activity│    │  • Relationships │
│  • User profiles │    │  • Session data │    │  • Collaborative │
└────────┬──────────┘    └────────┬──────────┘    └────────┬──────────┘
         │                         │                           │
         └───────────┬─────────────┴───────────┬───────────────┘
                     │                         │
             ┌───────▼─────────┐     ┌─────────▼─────────┐
             │   Qdrant Vector │     │   Recommendation  │
             │  • Item embeddings│     │  • Hybrid ranking │
             └─────────────────┘     └─────────────────────┘
```

## Implementation Details

### PostgreSQL Configuration
- **Schema design**: 
  - `items` table: id, title, category, metadata (JSONB)
  - `users` table: id, preferences, demographics, segments
  - `interactions` table: user_id, item_id, timestamp, type, score
- **Indexing strategy**: 
  - GIN indexes on JSONB metadata for faceted search
  - BRIN indexes on timestamp for time-range queries
  - Partial indexes for active items only
- **Extensions**: pgvector for hybrid search capabilities

### Cassandra Implementation
- **Table design**: 
  - `user_activity` (partition key: user_id, clustering: timestamp DESC)
  - `session_data` (partition key: session_id, clustering: event_time)
  - `realtime_metrics` (partition key: metric_type, clustering: timestamp)
- **Consistency**: QUORUM for writes, ONE for reads (optimized for speed)
- **Compaction**: Size-tiered compaction for write-heavy workloads
- **Caching**: Row cache enabled for hot user sessions

### Neo4j Graph Database
- **Node types**: `User`, `Item`, `Category`, `Tag`, `Session`
- **Relationship types**: `VIEWED`, `PURCHASED`, `RATED`, `FOLLOWS`, `RELATED_TO`
- **Query patterns**:
  - Collaborative filtering: `MATCH (u:User)-[:VIEWED]->(i:Item)<-[:VIEWED]-(u2:User)-[:VIEWED]->(i2:Item)`
  - Content-based: `MATCH (i:Item)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $tags`
  - Session-based: `MATCH (s:Session)-[:CONTAINS]->(i:Item) WHERE s.timestamp > $window`
- **Performance**: Indexes on `User.id`, `Item.id`, `Tag.name`; materialized views

### Qdrant Vector Database
- **Collection design**: 
  - `items`: vector embeddings + metadata (category, popularity, recency)
  - `users`: user preference embeddings for collaborative filtering
- **HNSW configuration**: `m=32`, `ef_construction=200`, `ef_search=150`
- **Quantization**: PQ with 8-bit quantization for memory efficiency
- **Hybrid search**: Vector similarity + metadata filtering + keyword search

## Performance Optimization

### Query Routing Strategy
- **Real-time recommendations**: Qdrant + Redis cache (sub-50ms)
- **Personalized feeds**: PostgreSQL + Cassandra join (50-100ms)
- **Graph-based recommendations**: Neo4j for complex relationship queries (100-200ms)
- **Fallback mechanism**: Keyword search when vector search fails

### Caching Strategy
- **Redis tier 1**: Hot recommendations (TTL: 5 minutes)
- **Redis tier 2**: User session state (TTL: 30 minutes)
- **Application cache**: Pre-computed popular items (TTL: 1 hour)
- **CDN cache**: Static recommendation widgets (TTL: 1 hour)

## Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Click-through Rate | 95%+ | >90% |
| P99 Latency | <50ms | <100ms |
| Throughput | 100K+ QPS | 80K QPS |
| Infrastructure Cost | 35% reduction | >30% reduction |
| Recommendation Freshness | <5 minutes | <10 minutes |
| Model Update Frequency | Real-time | <15 minutes |

## Key Lessons Learned

1. **Multi-model architecture is essential for complex recommenders** - no single database can handle all requirements
2. **Vector databases transform recommendation quality** - semantic similarity outperforms traditional collaborative filtering
3. **Real-time feedback loops are critical** - immediate user feedback improves recommendation relevance
4. **Caching strategy must be multi-layered** - different TTLs for different data types
5. **Query routing intelligence improves performance** - directing queries to optimal systems

## Technical Challenges and Solutions

- **Challenge**: Cold start problem for new users/items
  - **Solution**: Hybrid approach using content-based + popularity-based recommendations

- **Challenge**: Real-time updates for rapidly changing user preferences
  - **Solution**: Change data capture (CDC) with Kafka for event-driven updates

- **Challenge**: Balancing exploration vs exploitation
  - **Solution**: Multi-armed bandit algorithms with database-backed state storage

- **Challenge**: Data consistency across heterogeneous systems
  - **Solution**: Event sourcing with transactional outbox pattern

## Integration with ML Systems

### Real-time Personalization Pipeline
1. **User request**: API gateway receives recommendation request
2. **Context extraction**: Extract user ID, session, device, location
3. **Query routing**: Determine optimal database combination
4. **Parallel execution**: Execute queries across multiple databases
5. **Result merging**: Combine and rank results using learned weights
6. **Feedback collection**: Log impression and click events
7. **Model update**: Real-time feature updates for next request

### Offline Training Integration
- **Feature engineering**: Extract graph features, temporal patterns, user behavior
- **Model training**: Daily batch training with fresh data
- **A/B testing**: Database-backed experiment tracking
- **Rollout strategy**: Canary releases with gradual traffic routing

> 💡 **Pro Tip**: For recommender systems, the most valuable improvements come from better data and feature engineering, not just better algorithms. Invest in database architecture that enables rich feature extraction and real-time feedback.