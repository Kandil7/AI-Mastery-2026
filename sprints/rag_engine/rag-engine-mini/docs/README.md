# Docs Index

Welcome to the RAG Engine Mini documentation.

## 📚 Documentation Structure

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design and data flow diagrams |
| [Modules](modules.md) | Module-by-module technical reference |
| [API Reference](api-reference.md) | Complete REST API documentation |
| [Configuration](configuration.md) | All environment variables and settings |
| [Deployment](deployment.md) | Local, Docker, K8s, and cloud deployment |
| [Developer Guide](developer-guide.md) | Extending and customizing the project |
| [Prompt Engineering](prompt-engineering.md) | LLM prompts and guardrails |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |

## 🧠 Technical Deep Dives

For those who want to understand the "Why" behind the architecture:

- [**Hybrid Search & RRF Fusion**](deep-dives/hybrid-search-rrf.md): Combining vector and keyword search.
- [**Chunking Strategies**](deep-dives/chunking-strategies.md): Splitting documents optimally for LLMs.
- [**Clean Architecture for AI**](deep-dives/clean-architecture-for-ai.md): Using Ports & Adapters for RAG.
- [**Advanced RAG (Stage 2)**](deep-dives/stage-2-advanced-rag.md): Streaming, Parent-Child, and Contextual Retrieval.
- [**Intelligence & Graph (Stage 3)**](deep-dives/stage-3-intelligence.md): Self-Corrective RAG and Knowledge Graphs.
- [**Multi-Modal & Tables (Stage 4)**](deep-dives/stage-4-multimodal.md): Tables-to-Markdown and Image Vision.
- [**Autonomous Agent (Stage 5)**](deep-dives/stage-5-autonomy.md): Routing, Web-Search and Privacy.
- [**Scaling RAG Pipes**](deep-dives/scaling-rag-pipelines.md): Async workers, Caching, and Redis.
- [**LLM-as-a-Judge**](deep-dives/llm-as-a-judge.md): Verification and Grounding.
- [**Common RAG Pitfalls**](deep-dives/common-rag-pitfalls.md): Troubleshooting common failures.
- [**Architecture Patterns**](deep-dives/architecture-patterns.md): Ports, Adapters, and Modularity.

## 🎓 Developer & Educational

- [**Developer Guide**](../docs/developer-guide.md): How to extend the system.
- [**Visual Guide**](./VISUAL_GUIDE.md): Flowcharts of Architecture and Data.

- [**Notebooks Index**](../notebooks/): Step-by-step interactive tutorials.

## 📖 Global Resources

- [**Glossary of Terms**](../GLOSSARY.md): English/Arabic technical reference.

## 🚀 Quick Links

- [Getting Started](../README.md#-quick-start)
- [Project Structure](../STRUCTURE.md)
- [Educational Notebooks](../notebooks/)

## 🔧 Key Concepts

### Clean Architecture
- **Domain**: Pure business logic (entities, errors)
- **Application**: Use cases and ports (interfaces)
- **Adapters**: External implementations (databases, APIs)
- **API**: HTTP layer (routes, validation)

### Hybrid Search
- Vector search (semantic similarity via Qdrant)
- Keyword search (lexical matching via Postgres FTS)
- RRF fusion (combining results without score calibration)
- Cross-Encoder reranking (precision improvement)

### Multi-Tenancy
- API key-based authentication
- Tenant isolation in all queries
- Per-tenant chunk deduplication

## 📝 Contributing to Docs

When adding documentation:
1. Use clear, concise language
2. Include code examples
3. Add bilingual support (English + Arabic) where helpful
4. Update this index file
