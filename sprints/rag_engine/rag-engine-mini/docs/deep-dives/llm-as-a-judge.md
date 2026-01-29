# ⚖️ Deep Dive: LLM-as-a-Judge (Self-Correction)

> Moving from "Trusting LLMs" to "Verifying LLMs".

---

## The Core Concept

Self-Corrective RAG (Self-RAG) uses a second LLM pass (or a cheaper model) to critique the outputs of the retrieval and generation phases. In this project, we implement this in `SelfCritiqueService`.

---

## 1. Grade Retrieval (Relevance Grader)

Most RAG systems fail when the search returns "junk". The LLM then tries to answer using that junk, leading to hallucinations.

*   **Process**: Before answering, the judge looks at the query and the top-N chunks.
*   **Prompt**: "Is this document relevant to the user's question?"
*   **Action**: If "Irrelevant", we don't even try to answer. We perform **Query Expansion** to find better context.

---

## 2. Grade Answer (Hallucination Grader)

Is the answer supported by the facts (Context)? Or did the model make it up from its internal memory?

*   **Process**: After generating the answer, the judge compares the answer against the retrieved chunks.
*   **Action**: If the judge finds "Hallucination", we discard the answer and regenerate it with a **Strict Grounding** prompt.

---

## 3. Why LLM-as-a-Judge?

1.  **Metric Proxy**: Traditional metrics like ROUGE or BLEU don't work for semantic meaning. LLM judges are closer to human evaluation.
2.  **Autonomous Loops**: It allows the system to realize it failed and try again without user intervention.
3.  **Audit Trail**: You can store the grades in Postgres to see which documents are causing poor performance.

---

## 🛠️ Implementation in RAG Engine Mini

We use a high-prompt-engineering approach in `src/application/services/self_critique.py`.

```python
def grade_answer(self, question, answer, context):
    # Sends Answer + Context to LLM
    # Expects binary choice: 'grounded' | 'hallucination'
```

---

## 📚 Further Reading
- [G-Eval: NLG Evaluation using GPT-4](https://arxiv.org/abs/2303.16634)
- [Prometheus: Inducing Fine-grained Evaluation Capability in LLMs](https://arxiv.org/abs/2310.08491)
