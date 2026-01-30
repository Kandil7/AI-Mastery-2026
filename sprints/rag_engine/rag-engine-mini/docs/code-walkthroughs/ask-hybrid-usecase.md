# 🚶 Code Walkthrough: The Hybrid Search Flow

## 🗺️ The Path of a Query
This guide follows a user's question from API to Answer.

### 1. The Entry Point: `src/api/routes/query.py`
The request hits `POST /v1/query`.
*   **Validation**: The `QueryRequest` Pydantic model ensures `text` is not empty.
*   **Dependency Injection**: The route asks for `AskHybridUseCase` from `bootstrap.py`.

### 2. The Use Case: `src/application/use_cases/ask_hybrid.py`
This is the "Brain" or "Orchestrator".
*   It does **NOT** know about OpenAI or Qdrant. It only knows interfaces (Ports).
*   **Parallel Execution**: It calls `retrieve_vectors` and `retrieve_keywords` often in parallel (or sequentially if simple).

### 3. The Retrieval
*   **Vector**: `src/adapters/qdrant/vector_store.py` -> Embeds query -> KNN Search.
*   **Keyword**: `src/adapters/postgres/keyword_store.py` -> Stems query -> TSvector matching.

### 4. The Fusion: `src/application/services/fusion.py`
We get 2 lists of results:
*   List A (Vector): [Doc1 (0.9), Doc2 (0.8)]
*   List B (Keyword): [Doc2 (10.0), Doc3 (5.0)]
*   **RRF Algorithm**: We merge them. Doc2 appears in both, so its score skyrockets.

### 5. The Answer: `src/adapters/llm/openai.py`
*   The top chunks are pasted into a `PROMPT_TEMPLATE`.
*   The LLM generates the answer.
*   The `Answer` object is returned with `sources`.

## 🧠 Key Design Pattern
Notice how the **Logic** (Fusion) is separated from the **Tools** (Qdrant). This is Clean Architecture.
