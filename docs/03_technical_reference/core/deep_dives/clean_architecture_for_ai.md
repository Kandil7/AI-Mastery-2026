# 🏗️ Deep Dive: Clean Architecture for AI

> Why we use Ports & Adapters for RAG instead of just a script.

---

## The "Dirty Script" Problem

Most AI prototypes look like this:
```python
# main.py
import openai
import qdrant_client

def ask(query):
    # Logic + API calls + Database calls all mixed together
    q_client = qdrant_client.QdrantClient(...)
    results = q_client.search(...)
    ...
```

**Why is this bad?**
1.  **Vendor Lock-in**: If you want to switch from Qdrant to Pinecone, you have to rewrite everything.
2.  **Hard to Test**: You can't test "Logic" without actually calling the "API".
3.  **Fragile**: A change in the database schema breaks the UI logic.

---

## 🏛️ The Solution: Ports & Adapters

In RAG Engine Mini, we follow **Clean Architecture (Hexagonal Architecture)**.

### 1. The Domain Layer (The Core)
Pure entities (math and data). No knowledge of OpenAI, SQL, or the Web.
*   *Example*: `Chunk`, `TenantId`, `Answer`.

### 2. The Application Layer (The Logic)
Contains "Use Cases" (workflows) and "Ports" (interfaces).
*   **Port**: A Protocol (Interface) that says "I need someone who can `embed_text`". It doesn't care if it's OpenAI or a local model.
*   **Use Case**: The orchestrator. "Get question → Call Embedder Port → Call Vector Port → Return Answer".

### 3. The Adapters Layer (The Implementation)
This is where the real work happens.
*   `OpenAIAdapter`: Implements the `EmbedderPort`.
*   `LocalModelAdapter`: ALSO implements the `EmbedderPort`.

---

## 🚀 The Result: Total Flexibility

Because we use **Dependency Injection**, we can swap components in `bootstrap.py` without touching a single line of business logic.

| Component | Standard | Choice B | Choice C |
|-----------|----------|----------|----------|
| **LLM** | OpenAI | Ollama | Anthropic |
| **Vector** | Qdrant | Milvus | Pinecone |
| **Embeddings** | OpenAI | HuggingFace | Cohere |
| **Persistence** | Postgres | MongoDB | CSV (Mock) |

---

## 🛠️ Implementation Example

In `src/application/ports/llm.py`:
```python
class LLMPort(Protocol):
    def generate(self, prompt: str) -> str: ...
```

In `src/adapters/llm/openai_llm.py`:
```python
class OpenAILLM:
    def generate(self, prompt: str) -> str:
        # call openai.chat...
```

In `src/application/use_cases/ask_question.py`:
```python
class AskQuestionUseCase:
    def __init__(self, llm: LLMPort): # Injected!
        self._llm = llm
```

---

## 🌟 Why this matters for AI Engineering
AI models evolve every week. Models get deprecated, better ones come out, and costs change. A **Clean Architecture** allows your codebase to survive the constant shifts of the AI landscape without needing a full rewrite.

---

## 📚 Further Reading
- [Clean Architecture by Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Hexagonal Architecture (Ports & Adapters)](https://alistair.cockburn.us/hexagonal-architecture/)
