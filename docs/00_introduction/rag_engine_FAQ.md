# ❓ Frequently Asked Questions (FAQ)

> Addressing the most common "Advanced" questions from Stage 3 and 4.

---

### Q1: Why use GraphRAG instead of standard Vector RAG?
**A**: Standard RAG finds specific "pages". GraphRAG finds "relationships" across pages. If you ask "How are Project X and Finance Team Y linked?", standard RAG might find finance team docs OR project docs, but GraphRAG finds the triplet `(Project X) - [ManagedBy] -> (Finance Team Y)`.

---

### Q2: How does the Self-Correction loop handle infinite loops?
**A**: We implement a max_retry of 1. If the first expansion/regeneration fails to improve the grade, we return the best effort with a warning. This prevents burning LLM tokens in a feedback loop.

---

### Q3: Why convert tables to Markdown?
**A**: LLMs understand structured text (Markdown/CSV) much better than tabular pixel positions. By explicit conversion, we ensure the LLM "sees" the hierarchy of rows and columns as structured data.

---

### Q4: Can I use local models for vision-sensing?
**A**: Yes! Ollama supports vision models like `llava`. Simply update the `OLLAMA_CHAT_MODEL` in your `.env` to a vision-capable model.

---

### Q5: Is the Knowledge Graph real-time?
**A**: Almost. Triplets are extracted during the indexing background task. As soon as a document is "indexed," its relationships are available for graph search.

---

### Q6: How do I scale the vision processing?
**A**: Since vision is slow, we recommend running a separate Celery queue for Multi-Modal tasks with more workers on GPU-enabled nodes.
