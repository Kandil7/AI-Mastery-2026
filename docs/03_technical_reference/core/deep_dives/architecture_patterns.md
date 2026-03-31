# 🏗️ Deep Dive: Architecture Patterns in AI

> Why "Clean Architecture" is the secret to future-proof AI systems.

---

## 1. The "LangChain Trap"
Many beginners start by import LangChain and using its `Chain` objects. While great for prototypes, this creates **High Coupling**:
*   If you want to switch from Qdrant to Chroma, you have to rewrite your chains.
*   Testing internal logic becomes hard because it's wrapped in library objects.

**Our Solution**: RAG Engine Mini uses **Vanilla Python** with **Ports & Adapters**.

---

## 2. Ports & Adapters (Hexagonal Architecture)

### The Port (Interface)
A Port defines *what* needs to be done. It's a `Protocol` (abstract interface).
Example: `LLMPort` says "I need a function that takes a string and returns a string."

### The Adapter (Implementation)
An Adapter defines *how* it's done for a specific vendor.
Example: `OpenAILLM` is an adapter. `OllamaLLM` is another.

**Benefit**: You can swap `OpenAI` for `Ollama` by changing ONE line in `bootstrap.py`. Your business logic (`Use Cases`) never changes.

---

## 3. The Dependency Injection (DI) Container

In this project, `src/core/bootstrap.py` is the **Brain**. It:
1.  Reads configuration.
2.  Instantiates all Adapters.
3.  Hooks them into the Use Cases.
4.  Returns a `container` dictionary.

This makes the system **Unit Testable**. You can "Mock" any adapter easily.

---

## 4. Separation of Concerns

| Directory | Responsibility |
|-----------|----------------|
| `src/domain` | Plain Data Objects. NO dependencies allowed. |
| `src/application` | Business Rules (e.g. "How to grade an answer"). |
| `src/adapters` | Talking to the outside world (DBs, APIs, Files). |
| `src/api` | Translating HTTP requests into Use Case calls. |

---

## 5. Senior Wisdom: Think in "Contracts"

When building Stage 5 or beyond, don't ask "How do I use this new AI library?". Ask "What is the **Contract** (Port) I need to fulfill?". 

This approach ensures your code survives for years, regardless of which AI model is currently the trend.
