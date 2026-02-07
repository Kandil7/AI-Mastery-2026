# 🎓 The Educational Layer

This repository is designed not just as a tool, but as a complete **AI Engineering Curriculum**.

## 🗺️ Navigation

### 1. Concept ↔ Code Map
Understanding *where* theory lives in the codebase.
👉 [Read the Concept Map](learning-map.md)

### 2. Failure Modes
Learn from broken systems. What happens when RAG fails?
👉 [Low Recall (Retrieval Failures)](failure-modes/01_low_recall.md)

### 3. Architecture Decision Records (ADRs)
Why did we build it this way? Decisions from the Architect's desk.
👉 [ADR-001: Postgres FTS vs Elastic](adr/001-postgres-fts-vs-elasticsearch.md)

---

## 🏗️ Structure
*   `docs/learning-map.md`: The central index.
*   `docs/adr/`: Architectural decisions.
*   `docs/failure-modes/`: Debugging guides.
*   `docs/exercises/`: Practical challenges.
    *   [Level 1: Semantic Embeddings](exercises/level1_embeddings.md)
*   `docs/code-walkthroughs/`: Step-by-step code tours.
    *   [Use Case: Ask Hybrid](code-walkthroughs/ask-hybrid-usecase.md)
*   **Evaluation Suites**:
    *   [Basics: Recall, Precision, MRR](../notebooks/90_evaluation_basics.ipynb)
