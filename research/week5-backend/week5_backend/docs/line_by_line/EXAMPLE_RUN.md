# Example Run: RAG Pipeline

This file shows concrete, minimal examples of how to run the pipeline and what to expect.

## Example 1: Hybrid retrieval

```python
from pipelines.online_query import run_query_pipeline

response = run_query_pipeline(
    tenant_id="tenant-123",
    question="What is our on-call policy?",
    filters={},
    top_k=8,
    mode="hybrid",
)

print(response["answer"])
print(response["citations"])
```

Expected behavior:
- Retrieval uses vector + BM25 with RRF fusion
- Reranker trims to `reranker.top_k`
- Answer is built from context (respecting `answer.max_context_words`)
- Verification runs and may trigger strict fallback

## Example 2: Agentic mode

```python
from pipelines.online_query import run_query_pipeline

response = run_query_pipeline(
    tenant_id="tenant-123",
    question="Find relevant data and summarize from web: incident runbook",
    filters={},
    top_k=6,
    mode="agentic",
)

print(response["answer"])
```

Expected behavior:
- Planner selects tools (rag, web, sql if configured)
- Tool outputs are synthesized into a final answer
- Verification runs; RAG fallback happens on failure

## Example 3: Vector-only mode

```python
from pipelines.online_query import run_query_pipeline

response = run_query_pipeline(
    tenant_id="tenant-123",
    question="Explain deployment rollback steps",
    filters={},
    top_k=5,
    mode="vector",
)
```

Expected behavior:
- BM25 is skipped
- Retrieval is purely vector-based
