# Architecture

## Goals
- Low-latency retrieval and generation with explicit citations.
- Multi-model routing by task (QA, summarization, extraction).
- Multi-tenant isolation and governance controls.
- Scale ingestion and indexing independently of online query.
- Observability: traceability, cost attribution, and evaluation hooks.

## High-level components
1) Ingestion pipeline
   - Sources: docs, tickets, PDFs, code repos, wikis, databases.
   - Steps: parsing -> cleaning -> structured chunking -> embedding -> indexing.
2) Retrieval stack
   - Hybrid retrieval: BM25 + vector search fused with RRF.
   - Optional query rewriting for better recall.
   - Reranking: cross-encoder or LLM reranker.
   - Filtering: tenant, ACL, recency, domain.
3) Answering stack
   - Prompt assembly with retrieved context.
   - Strict fallback when verification fails.
   - Provider selection: cost/latency/quality policies.
   - Citation formatting and hallucination checks.
4) Agentic RAG
   - Planner -> tools -> verifier loop.
   - Tools: RAG, SQL, web, calculators, internal APIs.
5) Evaluation and feedback
   - Offline datasets + online feedback signals.
   - Regression detection and A/B testing.

## Data flow (online)
Client -> API (query) -> planner -> retriever -> reranker -> answer -> verifier -> response

```mermaid
flowchart LR
    A[Client] --> B[API: /query]
    B --> C[Planner]
    C --> D[Retriever]
    D --> E[Reranker]
    E --> F[Answer Generator]
    F --> G[Verifier]
    G --> H[Response + Citations]
```

## Data flow (offline)
Source -> parser -> chunker -> embeddings -> vector store -> metadata store -> eval dataset

```mermaid
flowchart LR
    S[Source Docs] --> P[Parser]
    P --> C[Chunker]
    C --> E[Embeddings]
    E --> V[Vector Store]
    E --> B[BM25 Corpus]
    V --> M[Metadata Store]
    M --> D[Eval Dataset]
```

## Agentic RAG flow
```mermaid
flowchart LR
    Q[Question] --> P[Planner]
    P --> T[Tool Registry]
    T --> R1[RAG Tool]
    T --> R2[SQL Tool]
    T --> R3[Web Tool]
    R1 --> S[Tool Outputs]
    R2 --> S
    R3 --> S
    S --> A[Synthesis Answer]
    A --> V[Verifier]
    V --> O[Final Response]
```

## Scaling patterns
- Separate ingestion workers from query service.
- Use async tasks for IO (vector DB, model calls).
- Cache embeddings and retrieval results by content hash.
- Shard vector indexes by tenant or domain.

## Failure modes and mitigations
- Provider outage: automatic failover to backup model.
- Vector store latency: fall back to BM25 and cached answers.
- Cost spikes: enforce per-tenant budgets and rate limits.
