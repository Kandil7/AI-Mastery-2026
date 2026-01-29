# 🖼️ Visual Guide: Data Flow & Architecture

> A visual representation of how RAG Engine Mini processes information.

---

## 🏗️ 1. Global Architecture (Clean Design)

The system is built on **Ports & Adapters**, ensuring the business logic remains "pure" and decoupled from databases or LLM providers.

```mermaid
graph TD
    API[API Layer / FastAPI] --> UC[Use Cases / Application]
    UC --> Ports[Interfaces / Ports]
    Ports --> Adapters[Implementation / Adapters]
    
    subgraph Adapters Layer
        Adapters --> Postgres[(Postgres DB)]
        Adapters --> Qdrant[(Qdrant Vector)]
        Adapters --> Redis[(Redis Cache)]
        Adapters --> AI[LLM / OpenAI / Ollama]
    end
```

---

## 📥 2. The Indexing Pipeline (Async)

Using **Celery** to prevent API blocking.

```mermaid
sequenceDiagram
    User->>API: Upload File
    API->>Store: Save to Disk
    API->>Redis: Push Task
    API-->>User: HTTP 202 (Processing)
    
    Worker->>Redis: Pop Task
    Worker->>Store: Load File
    Worker->>Extractor: Extract Text (PyMuPDF)
    Worker->>LLM: Generate Doc Summary
    Worker->>Chunker: Hierarchical Chunking
    Worker->>Cache: Embed Child Chunks
    Worker->>Postgres: Save Parents & Groups
    Worker->>Qdrant: Save Child Vectors
    Worker->>Postgres: Update Doc Status (Indexed)
```

---

## 🔍 3. The Retrieval Flow (Hybrid + Self-RAG)

How we find the best answer with **Self-Correction**.

```mermaid
graph LR
    Query[User Query] --> Expand[Query Expansion]
    Expand --> Vector[Vector Search]
    Expand --> Keyword[Keyword Search]
    
    Vector --> Hydrate[Hydrate Parent Text]
    Keyword --> Hydrate
    
    Hydrate --> Fusion[RRF Fusion]
    Fusion --> Rerank[Cross-Encoder Rerank]
    
    Rerank --> Grade{Judge: Relevant?}
    Grade -- No --> Rewrite[Rewrite Question]
    Rewrite --> Vector
    
    Grade -- Yes --> Prompt[Build Contextual Prompt]
    Prompt --> LLM[Generate Answer]
    
    LLM --> Grounded{Judge: Hallucination?}
    Grounded -- Yes --> Strict[Regenerate Stricter]
    Strict --> Final[Final Answer]
    Grounded -- No --> Final
```

---

## 🕸️ 4. Graph Construction

Turning messy docs into structured knowledge.

```mermaid
graph LR
    Chunk[Text Chunk] --> LLM[LLM Extractor]
    LLM --> T1[Subject: AI]
    LLM --> T2[Relation: PartOf]
    LLM --> T3[Object: Computer Science]
    
    T1 & T2 & T3 --> DB[(Graph Triplets Table)]
```

---

> [!TIP]
> Use this guide alongside the [Architecture Docs](./architecture.md) to understand the high-level design patterns used in the codebase.
