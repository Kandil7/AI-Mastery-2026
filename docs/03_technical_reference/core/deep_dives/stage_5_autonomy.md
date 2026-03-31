# 🚀 Stage 5: Autonomy & Web Intelligence

> From "Search Engine" to "Autonomous AI Agent".

---

## 1. Semantic Routing (The Efficiency Brain)

**The Problem**: Sending every "Hello" or "How are you?" to a complex Hybrid RAG pipeline is a waste of time and money (LLM tokens).

**Our Solution**:
1.  **Classification**: We use a fast, low-temperature LLM pass to classify the user's intent.
2.  **Routes**:
    *   `CHITCHAT`: Responds immediately without searching any database.
    *   `DOCUMENT_QNA`: Standard RAG flow for local data.
    *   `LIVE_INFO`: Triggers the Web Search tool.

**Benefit**: Faster response times and up to 40% reduction in API costs.

---

## 2. Web Search Fallback (Universal Knowledge)

**The Problem**: RAG systems only know what you upload. If the answer isn't in your files, the system fails.

**Our Solution**:
1.  **Fallback Logic**: If the `Self-RAG` grader decides retrieval is "irrelevant" for the local docs, the system automatically pivots.
2.  **Tavily/Searxng Tool**: The system performs a real-time web search and formats the results as if they were document chunks.

**Benefit**: Your AI becomes "all-knowing," merging your private data with the live world.

---

## 3. Privacy Guard (PII Redaction)

**The Problem**: Sending sensitive data (Emails, Phone numbers) to external APIs like OpenAI is a compliance risk.

**Our Solution**:
1.  **Redaction**: Before the question reaches the LLM, we replace sensitive info with placeholders (e.g., `<EMAIL_0>`).
2.  **Restoration**: After the LLM generates the answer, we "pop" the original values back into the text before showing it to the user.

---

## 🛠️ How to use Autonomous Features

These features are enabled by default in `AskQuestionHybridUseCase`. You can configure the Web Search API key in your `.env` file:
```env
TAVILY_API_KEY=your_key_here
```

---

## 📚 Further Learning
- [Semantic Router: High-speed classification](https://github.com/aurelio-labs/semantic-router)
- [Autonomous Agents and Tool-Use](https://blog.langchain.dev/agents/)
