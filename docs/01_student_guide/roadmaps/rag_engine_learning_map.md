# 🧩 Concept → Code Map (Educational Layer)

This document bridges the gap between **Theory** (AI Concepts) and **Practice** (Production Code). Use this map to understand *where* specific concepts live in the codebase and *why* they were implemented that way.

---

## 🏗️ Core RAG Architecture

### 1. Embeddings & Vectorization
*   **Concept**: Converting text into dense vector representations.
*   **Code Location**:
    *   `src/application/ports/embeddings.py`: The abstract interface (Port).
    *   `src/adapters/embeddings/openai.py`: The specific implementation.
*   **Design Rationale**:
    *   **Port Pattern**: Allows swapping OpenAI for HuggingFace/Ollama without touching core logic.
    *   **Pydantic Models**: Ensures strict typing for inputs/outputs.

### 2. Hybrid Search (The Retrieval Engine)
*   **Concept**: Combining Keyword Search (Lexical) with Semantic Search (Vector) to maximize Recall.
*   **Code Location**:
    *   `src/application/services/fusion.py`: Implementation of RRF (Reciprocal Rank Fusion).
    *   `src/adapters/postgres/keyword_store.py`: Postgres `tsvector` implementation.
    *   `src/adapters/qdrant/vector_store.py`: Qdrant vector retrieval.
*   **Key Insight**: Semantic search misses exact keywords (dates, SKUs); Keyword search misses synonyms. Hybrid covers both.

### 3. Chunking Strategies
*   **Concept**: Splitting reliable text into manageable pieces for the LLM.
*   **Code Location**:
    *   `src/application/services/chunking_service.py`: Logic for fixed-size and semantic windowing.
*   **Trade-offs**:
    *   **Small Chunks**: Better retrieval precision, missing context.
    *   **Large Chunks**: Better context, diluted vector meaning.

---

## 🧠 Advanced Intelligence (Level 10+)

### 4. Reranking (Cross-Encoder)
*   **Concept**: Scoring the retrieved documents to strictly filter out noise.
*   **Code Location**:
    *   `src/application/services/reranker.py`: Using a `CrossEncoder` model locally.
*   **Performance Note**: Reranking is computationally expensive (slow) but critically increases precision. It acts as the "Quality Gate".

### 5. Multi-Agent Orchestration
*   **Concept**: Specialized agents coordinating to solve a problem.
*   **Code Location**:
    *   `notebooks/14_multi_agent_swarm_orchestration.ipynb`: The Supervisor/Worker implementation.
*   **Why Code-First?**: Swarms are logic flows, not just static architecture.

---

## 🛡️ Security & Production

### 6. PII Redaction
*   **Concept**: Removing sensitive data before it hits the vector DB or LLM.
*   **Code Location**:
    *   `src/application/services/pii_service.py`: Regex and NLP-based redaction.

### 7. Evaluation (RAGAS)
*   **Concept**: Quantifying "Goodness" using metrics like Faithfulness and Answer Relevance.
*   **Code Location**:
    *   `scripts/evaluate_ragas.py`: The pipeline that runs test questions against the RAG.

---

## 🧪 Educational "Why Not?"

### Why Postgres for Keywords?
*   **Alternative**: Elasticsearch / Opensearch.
*   **Decision**: For a typical RAG app (<10M Docs), managing a separate ES cluster is complex engineering overhead. Postgres `tsvector` is "Good Enough" and 10x simpler to deploy.

### Why Qdrant?
*   **Alternative**: pgvector / Pinecone.
*   **Decision**: Qdrant is built for high-performance Rust-based filtering and has a great local-mode (Docker) that matches cloud behavior perfectly.
