# 🧠 Stage 3: Intelligence & Knowledge (GraphRAG)

> Moving from "Search" to "Reasoning" with Self-Correction and Knowledge Graphs.

---

## 1. Self-Corrective RAG (Self-RAG)

RAG systems often fail because they retrieve the wrong documents or hallucinate answers. Stage 3 implements a **Self-Correction loop**.

### How it works:
1.  **Retrieve**: Fetch chunks using hybrid search.
2.  **Grade Retrieval**: An "LLM Evaluator" checks if chunks are relevant to the question.
    *   *If irrelevant*: Rewrite the query and search again (Attempt 2).
3.  **Generate**: Produce an answer.
4.  **Grade Answer**: The evaluator checks if the answer is grounded in facts.
    *   *If hallucinated*: Regenerate with stricter "Grounding" instructions.

**Benefit**: Significantly higher precision. The system "knows" when it doesn't know.

---

## 2. Knowledge Graph Foundations (GraphRAG)

Standard RAG struggles with "Global" questions like: *"How are Company A and Company B related?"* if that information is spread across 5 documents.

### Implementation:
1.  **Extraction**: During indexing, we use the LLM to extract **Knowledge Triplets**: `(Subject) - [Relation] -> (Object)`.
2.  **Storage**: These are stored in a structural table (`graph_triplets`).
3.  **Graph Search**: You can now search for an entity (e.g., "AI Engineering") and see all its relationships across the entire database, regardless of which document they came from.

---

## 🛠️ New Features in Action

### 1. Self-Correction
This is enabled automatically in the `AskQuestionHybridUseCase`. You will see system logs like `retrieval_graded` and `answer_graded`.

### 2. Graph Search
Use the new endpoint to explore relationships:
```bash
GET /api/v1/queries/graph-search?entity=Transformers
```

---

## 🚀 The Future: Full GraphRAG
In the next evolution, we will implement **Traversals**, allowing the system to follow the "path" between entities to answer multi-hop questions automatically.

---

## 📚 Further Learning
- [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)
- [Microsoft: GraphRAG Implementation Guide](https://microsoft.github.io/graphrag/)
