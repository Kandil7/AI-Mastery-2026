# Runbook (ops)

## SLO targets
- P95 query latency: < 2.5s
- Availability: 99.9%
- Freshness: ingestion to query <= 10 minutes

## Alerts
- Provider error rate > 5%
- Vector DB latency P95 > 500ms
- Cost per 1k queries exceeds threshold

## Playbooks
1) Provider outage
   - Switch routing policy to backup provider.
   - Reduce model size or enable cache.

2) Vector DB degradation
   - Enable BM25-only fallback.
   - Rebuild or re-shard index if corrupt.

3) Cost spike
   - Lower top_k, disable reranker, or increase cache TTL.

## Backups
- Snapshot vector indexes daily.
- Store raw documents in object storage with versioning.
