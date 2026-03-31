# Operation Guide: RAG Pipeline

This guide explains how the online and offline paths work together, and how to operate the system safely.

## Online path (query)

High-level flow:

1) Accept request (question, tenant_id, filters, mode)
2) Load settings
3) Apply tenant filter
4) Retrieve candidates (vector + optional BM25)
5) Optional query rewrite
6) Optional rerank
7) Generate answer with context budget
8) Verify and strict-fallback if needed
9) Return answer + citations + trace

### Modes

- `hybrid` -> vector + BM25 with fusion and reranking
- `vector` -> vector only
- `agentic` -> tool plan execution + synthesis

### Practical recommendations

- Keep `verification.enabled=true` in production
- Use `reranker.enabled=true` if quality matters more than latency
- Use `query_rewrite` only if queries are long or messy

## Offline path (index)

High-level flow:

1) Load settings
2) Build chunker (structured or simple)
3) Read text sources
4) Chunk, embed, and upsert into vector store
5) Save BM25 corpus for hybrid retrieval

### When to re-index

- After changing `chunking.mode`, `max_tokens`, or `overlap`
- After major content updates
- After changes to embedding model

## Common issues and fixes

- Empty or low-quality results:
  - Increase `retrieval.top_k`
  - Enable `query_rewrite`
  - Verify you re-indexed after chunking changes

- Hallucinations:
  - Keep `verification.enabled=true`
  - Keep `reranker.enabled=true`
  - Lower `answer.max_context_words` to force tighter context

- Slow responses:
  - Lower `retrieval.top_k`
  - Disable `query_rewrite`
  - Use smaller model for QA

## Suggested production defaults

```yaml
chunking:
  mode: "structured"
  max_tokens: 400
  overlap: 40
retrieval:
  top_k: 12
  fusion:
    use_rrf: true
reranker:
  enabled: true
  top_k: 8
query_rewrite:
  enabled: false
answer:
  max_context_words: 1200
verification:
  enabled: true
```
