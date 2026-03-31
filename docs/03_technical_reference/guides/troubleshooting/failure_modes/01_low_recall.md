# ❌ Failure Mode: Low Recall (The "I Don't Know" Problem)

## 🤕 Symptoms
*   The user asks a question where the answer **definitely exists** in the documents.
*   The model replies: *"I'm sorry, I don't have that information"* or gives a generic answer.
*   The Retrieval Score (if logged) is low.

## 🔍 Root Causes Analysis

### 1. The "Keyword Mismatch" Trap
*   **Scenario**: User searches for "Q3 Financials", document contains "Third Quarter Earnings".
*   **Vector Failure**: Sometimes embeddings drift strictly on semantic meaning and miss exact terminologies if the dense vector space is not fine-tuned.
*   **Fix**: This is why we implemented `Hybrid Search`. The Keyword element would likely catch "Financials" if synonyms were expanded, but raw vectors might miss the exact phrasing connection without context.

### 2. The "Chunking" Window
*   **Scenario**: The answer spans across two chunks. Chunk A has the setup, Chunk B has the conclusion. Neither chunk *alone* looks relevant enough to be retrieved.
*   **Fix**: **Contextual Retrieval** or **Parent-Child Chunking**. We implemented Parent-Child in Level 8 to return the larger context when a small snippet matches.

### 3. "Top-K" Starvation
*   **Scenario**: We retrieve top-3 chunks. The correct answer is in chunk #5.
*   **Fix**: Increase retrieval to top-10 or top-20, then use a **Reranker** (Cross-Encoder) to filter them down. Reranking allows "Broad Search, Strict Filter".

## 🛠️ How We Fixed It in RAG Engine Mini

| Strategy | Component | Implementation |
| :--- | :--- | :--- |
| **Hybrid Search** | `fusion.py` | Combines Vector + Keyword search to catch both meaning & exact terms. |
| **Reranking** | `reranker.py` | We fetch Top-25 and rerank to Top-5 to avoid starvation. |
| **Query Expansion** | `query_service.py` | We rewrite the user query to 3-4 variations to cast a wider net. |
