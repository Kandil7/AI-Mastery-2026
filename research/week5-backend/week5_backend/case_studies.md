# Real-Case Projects (Full Examples)

## 1) Customer Support RAG (Multi-Tenant SaaS)
**Goal:** Respond to support tickets with citations from tenant-specific docs.

### Data sources
- Zendesk/Intercom tickets
- Help center docs (HTML/PDF)
- Product release notes

### Pipeline
1) Ingest tickets nightly per tenant.
2) Chunk by section + ticket resolution.
3) Embed and index with `tenant_id` + `plan_tier` filters.
4) Online query uses `tenant_id` filter, reranker on top 50, and strict citations.

### API flow
- `POST /ingest` with `{tenant_id, source_type: "web", uri}`
- `POST /query` with `{tenant_id, question, top_k: 8, mode: "rag"}`

### Example output
Answer includes citations:
- chunk_id, doc_id, snippet, score.


## 2) Compliance & Legal RAG
**Goal:** Verified answers with a verifier agent. Every claim must be cited.

### Data sources
- Policies, contracts, regulatory PDFs
- Internal memos and audits

### Pipeline
1) OCR and parse PDFs.
2) Chunk by section headers.
3) Rerank with a legal domain model.
4) Verifier checks citations and rejects unsupported claims.

### Agentic pattern
- Planner chooses: RAG -> verifier
- Verifier forces re-query if missing citations


## 3) Engineering Knowledge Base (Code-Aware)
**Goal:** Answer engineering questions referencing code and RFCs.

### Data sources
- Git repos, ADRs, RFCs, API docs

### Pipeline
1) Parse repo and separate code vs prose.
2) Chunk code by function/module.
3) Use hybrid retrieval (BM25 + vector).
4) Rerank for exact API usage.

### Agentic extensions
- Tool: SQL for build metrics
- Tool: RAG for docs
- Tool: Web for external RFCs (optional)
