# 🎓 The RAG Engineering Blueprint

> A high-level guide to how RAG Engine Mini was constructed and why.

---

## 🏗️ The Foundations

Building a RAG engine isn't about connecting an API; it's about building a **robust data pipeline**. We chose **Clean Architecture** as our foundation to ensure that:
1.  Our code is independent of libraries (LangChain/LlamaIndex free).
2.  Our business logic is easily testable.
3.  Our storage can be swapped without rewriting the UI.

---

## 🔍 The Retrieval Philosophy

We believe in **"Semantic + Lexical"** complementarity.

1.  **Vector Search** is the "Fast Brain" (Intuition). It finds things that *feel* similar.
2.  **Keyword Search** is the "Logical Brain" (Fact). It finds things that *match* exactly.

By using **RRF (Reciprocal Rank Fusion)**, we merge these two brains without the need for manual weight tuning (which is usually a nightmare).

---

## ⚡ The Production Reality

A notebook RAG script dies when it hits 1,000 documents. To avoid this, we implemented:

### 1. Idempotency (The "I've seen this before" rule)
We hash every file. If you upload the same file 10 times, we only index it once. This saves **$$$** and processing time.

### 2. Async Workers
Heavy lifting (OCR, Embedding, Indexing) happens in the background via **Celery**. This keeps the API responsive for the user.

### 3. Observability
"What you can't measure, you can't improve." We added **Prometheus** metrics to track LLM costs and search quality in real-time.

---

## 🚀 The Next Frontier

Want to go deeper? Here is what you should learn next:

1.  **Agentic RAG**: Giving the LLM tools to search the web or run code.
2.  **GraphRAG**: Storing relationships between entities, not just chunks.
3.  **Knowledge Distillation**: Training smaller models to perform indexing tasks at lower costs.

---

## 📚 Educational Path

1.  Read the [Architecture Docs](docs/architecture.md).
2.  Follow the [Notebooks](notebooks/).
3.  Deep dive into [RRF Fusion](docs/deep-dives/hybrid-search-rrf.md).
4.  Run the [Evaluation Script](scripts/eval_retrieval.py).

*Happy Engineering!*
