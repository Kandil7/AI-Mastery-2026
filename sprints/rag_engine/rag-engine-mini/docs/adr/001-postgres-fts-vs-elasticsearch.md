# ADR-001: Postgres FTS vs Elasticsearch for Keyword Search

## Status
Accepted

## Context
To implement **Hybrid Search**, we need a strong Keyword Search (Lexical) component. The industry standard is typically Elasticsearch (or OpenSearch), but our stack already relies on Postgres for relational data.

For our RAG system, we need to decide between:
- **PostgreSQL Full-Text Search (FTS)**: Leverage existing database
- **Elasticsearch**: Industry-standard search engine

## Decision
Use PostgreSQL Full-Text Search (FTS) as the keyword search component.

## Alternatives Considered
### Elasticsearch
- Pros: Industry standard, advanced features, superior performance at scale
- Cons: Additional infrastructure, operational complexity, licensing costs

### SQLite with FTS
- Pros: Lightweight, simple deployment
- Cons: Limited concurrent access, less robust than Postgres

### PostgreSQL FTS (Chosen)
- Pros: Leverages existing infrastructure, ACID compliance, good performance for medium datasets
- Cons: Less advanced features than Elasticsearch, scaling limitations

## Rationale
1. **Operational Simplicity**: Using existing Postgres infrastructure reduces operational overhead
2. **Consistency**: Single source of truth for both structured and unstructured data
3. **Cost-Effectiveness**: No additional licensing or infrastructure costs
4. **Sufficient Performance**: Adequate for most RAG use cases (< 1M documents)

## Consequences
### Positive
- Reduced infrastructure complexity
- Lower operational costs
- Easier deployment and maintenance
- Consistent backup and recovery procedures

### Negative
- Scaling limitations compared to Elasticsearch
- Fewer advanced search features
- Potential performance bottlenecks at high scale

## Validation
This approach has been validated in production systems with 10k-100k documents where the simplicity outweighs the advanced features of Elasticsearch.