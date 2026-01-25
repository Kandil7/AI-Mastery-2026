# Quick Start: RAG Pipeline (Week5 Backend)

This guide shows the fastest path to enable the new RAG pipeline controls.

## 1) Configure settings

Edit your active config (usually `research/week5-backend/week5_backend/config/settings.yaml`) and set the key knobs below.

```yaml
chunking:
  mode: "structured"
  max_tokens: 400
  overlap: 40
retrieval:
  top_k: 12
  fusion:
    use_rrf: true
    rrf_k: 60
    vector_weight: 1.0
    bm25_weight: 1.0
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

## 2) Rebuild the index (recommended)

If you changed chunking or BM25 settings, re-index your data:

```bash
python research/week5-backend/week5_backend/pipelines/offline_index.py --source <PATH>
```

## 3) Run a query

Use the API or call the pipeline directly from code:

```python
from pipelines.online_query import run_query_pipeline

response = run_query_pipeline(
    tenant_id="tenant-123",
    question="What is our on-call policy?",
    filters={},
    top_k=8,
    mode="hybrid",
)
```

## 4) Tuning checklist

- If results are weak: raise `retrieval.top_k` and/or enable `query_rewrite`.
- If answers are verbose: lower `answer.max_context_words`.
- If hallucinations appear: keep `verification.enabled=true` and `reranker.enabled=true`.

## 5) Minimal safe defaults

If you want the simplest safe setup, use:

```yaml
chunking:
  mode: "structured"
retrieval:
  top_k: 10
reranker:
  enabled: true
verification:
  enabled: true
```
