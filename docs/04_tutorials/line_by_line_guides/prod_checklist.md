# Production Readiness Checklist

Use this checklist before enabling the pipeline in production.

## Data and Indexing
- [ ] Chunking mode is set (`structured` recommended)
- [ ] Re-indexing done after chunking or embedding changes
- [ ] BM25 corpus exists at `bm25_index_path` if hybrid mode is used

## Retrieval and Ranking
- [ ] `retrieval.top_k` tuned for recall
- [ ] RRF fusion enabled if using hybrid retrieval
- [ ] Reranker enabled and `top_k` set

## Answer Safety
- [ ] `verification.enabled=true`
- [ ] `answer.max_context_words` set to prevent prompt bloat
- [ ] Strict fallback tested on low-context queries

## Security and Governance
- [ ] Tenant ID always provided to `run_query_pipeline`
- [ ] Filters enforce ACL/tenant boundaries
- [ ] Tooling config doesn’t expose unrestricted web/SQL access

## Performance
- [ ] Measure latency for retrieval, rerank, generation
- [ ] Cache layer considered for hot queries
- [ ] Model routing validated for cost control

## Observability
- [ ] Trace ID logged per request
- [ ] Errors captured with enough metadata
- [ ] Citation coverage monitored
