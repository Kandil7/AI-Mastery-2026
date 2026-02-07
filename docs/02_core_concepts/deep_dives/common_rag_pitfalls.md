# 🚧 Common RAG Pitfalls & Solutions

> A "War Story" guide to what goes wrong in RAG systems and how we fixed it in RAG Engine Mini.

---

## 1. The "Needle in a Haystack" problem
**The Pitfall**: You have 10,000 document chunks, but the answer is hidden in a sentence that doesn't use the same keywords as the question.
**Our Solution**: **Hybrid Search**. We use Vector embeddings (for meaning) and Keyword FTS (for exact naming). By combining them with **RRF Fusion**, we ensure the "needle" stays at the top.

---

## 2. The "Loss of Context" problem
**The Pitfall**: You break a book into 500-word chunks. One chunk says "The CEO decided to resigned." If you search for "Who resigned?", you find the chunk, but it doesn't say who the CEO is (it was mentioned 3 pages ago).
**Our Solution**: 
1.  **Parent-Child Retrieval**: We search small chunks but send large parent-context chunks to the LLM.
2.  **Contextual Retrieval**: We prepend a document summary to every chunk so the LLM always knows which "CEO" we are talking about.

---

## 3. The "AI Confidence" problem (Hallucinations)
**The Pitfall**: The user asks about "Quantum Coffee Machines." Your database has nothing about this. The LLM gets confused and invents a story about how Quantum Coffee is brewed.
**Our Solution**: **Self-RAG (LLM-as-a-Judge)**. We have a grader that explicitly checks if the retrieved info is relevant. If the relevance score is 0, we tell the user "I don't know" instead of inventing facts.

---

## 4. The "Performance Wall" problem
**The Pitfall**: Your RAG works great in a notebook. You deploy it. 10 users upload files at once. Your API crashes because of the heavy embedding workload.
**Our Solution**: **Celery Worker Architecture**. API handles requests; Workers handle the "heavy lifting." This keeps the UI snappy even if the backend is processing 1,000 pages.

---

## 5. The "Re-Indexing Burn" (Cost)
**The Pitfall**: Every time you test, you pay OpenAI $0.01. After 1,000 tests, you've spent $10 unnecessarily on the same sentences.
**Our Solution**: **Redis Embedding Cache**. We save every vector. If the system sees the same text again, the cost is $0.00.

---

> [!CAUTION]
> Avoid "Library Lock-in" (e.g. over-relying on LangChain). We built this repo with **Vanilla Python** logic to give you total control over these solutions.
