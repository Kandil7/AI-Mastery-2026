# 🏗️ Deep Dive: Scaling RAG Pipelines

> Why simple scripts fail and how we designed RAG Engine Mini for production scaling.

---

## 1. The Async Bottleneck

In a basic RAG script, when a user uploads a 100-page PDF, the API hangs while it chunks and embeds. This causes **Timeouts** and a terrible user experience.

### The Solution: Task Queues
In RAG Engine Mini, we use **Celery + Redis**:
1.  API receives the file and saves it to the `file_store`.
2.  API pushes a task ID to Redis.
3.  A background **Worker** picks up the task and does the heavy lifting (OCR, Chunking, Embedding).
4.  The user can poll the `/status` endpoint to see progress.

---

## 2. Infrastructure as Code (Docker)

Running Postgres, Redis, Qdrant, and Celery manually is a nightmare. We use **Docker Compose** to ensure:
*   **Environment Parity**: Your local setup matches production perfectly.
*   **Isolation**: No "it works on my machine" issues.

---

## 3. Embedding Cache (The Money Saver)

Embeddings are expensive. If you re-index the same document (or similar sentences), you shouldn't pay twice.
Inside `CachedEmbeddings`:
1.  We hash the text chunk.
2.  Check **Redis** if the vector exists.
3.  If yes, return it in 1ms.
4.  If no, call OpenAI/Local and store it for next time.

---

## 4. Multi-Tenancy (B2B Ready)

Most RAG engines forget about security. In RAG Engine Mini, every row in the database and every point in the vector store has a `user_id` (Tenant).
When you search:
```python
vector_store.search(..., filter={"user_id": current_user.id})
```
This ensures User A can NEVER see User B's documents, even if they share the same physical database.

---

## 5. Pro-Tip: Horizontal Scaling

Because our architecture is **Stateless**, you can run 10 copies of the API and 50 copies of the Worker on different servers. They all talk to the same Postgres/Qdrant, and the system scales linearly.
