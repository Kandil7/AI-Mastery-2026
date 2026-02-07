# 🚀 Stage 2: Advanced RAG Strategies

> Deep dive into Streaming, Parent-Child Retrieval, and Contextual Retrieval.

---

## 1. Streaming Responses (SSE)

Waiting 5-10 seconds for a full LLM response feels slow. Stage 2 adds **Streaming**, where the user sees the answer "typing" in real-time.

### How it works:
1.  **LLM Layer**: Use `stream=True` in OpenAI/Ollama APIs.
2.  **Use Case**: Yields chunks as they arrive.
3.  **FASTAPI**: Uses `StreamingResponse` with `media_type="text/plain"`.
4.  **Frontend**: The Gradio UI uses the generator to update the markdown box continuously.

---

## 2. Parent-Child Retrieval

**The Problem**: Large chunks are slow to search and expensive to embed. Small chunks are fast but lack context (e.g., a chunk might say "it costs $50" without mentioning the product name).

**The Solution**:
1.  **Indexing**: We split the document into **Parents** (2048 tokens) and **Children** (512 tokens).
2.  **Storage**: Both are stored in Postgres, but only **Children** are indexed in Qdrant.
3.  **Retrieval**: We search for the best Child, but we fetch the **Parent Text** to send to the LLM.

**Benefit**: The LLM gets the rich surrounding context of the Parent, while the search stays laser-focused on the precise Child match.

---

## 3. Contextual Retrieval (Anthropic Style)

Even parent chunks can lose context if the document is very technical or the subject is only mentioned in the first page.

**Implementation**:
1.  During indexing, the LLM generates a **1-2 sentence summary** of the entire document.
2.  This summary is stored as `chunk_context`.
3.  During retrieval, we prepend this summary to every chunk: `[Context: Summary...] \n\n {Chunk Text}`.

**Benefit**: Every chunk now "remembers" what the document is about, significantly reducing hallucinations.

---

## 🛠️ How to use these features?

These features are enabled by default in Stage 2 indexing.

1.  **Upload a document**: It will automatically use hierarchical chunking and summary generation.
2.  **Ask a question**: Use the new `/api/v1/queries/ask-hybrid-stream` endpoint or run `make demo`.

---

## 📚 Further Learning
- [Anthropic: Contextual Retrieval Guide](https://www.anthropic.com/news/contextual-retrieval)
- [Hierarchical Chunking Explained](https://cookbook.openai.com/examples/recursive_retriever_nodes)
