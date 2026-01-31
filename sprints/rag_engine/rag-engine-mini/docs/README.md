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
- [**Privacy & Compliance**](deep-dives/privacy-and-compliance.md): PII Redaction and Security.
- [**Production Readiness**](deployment.md#production-readiness-audit): Infrastructure, Health Checks, and Hardening.
- [**LLM Provider Strategy & Adapters**](./notebooks/08_llm_provider_strategy.ipynb): Adapters for different LLM services.
- [**Semantic Chunking**](./notebooks/09_semantic_chunking.ipynb): High-precision embedding-based splitting.
- [**Vector Visualization**](./notebooks/10_vector_visualization.ipynb): 3D semantic cluster visualization (PCA).
- [**Agentic RAG Workflows**](./notebooks/11_agentic_rag_workflows.ipynb): **[PINNACLE]** Autonomous ReAct planning.
- [**Synthetic Data Flywheel**](./notebooks/12_synthetic_data_flywheel.ipynb): **[LEGEND]** Self-improving evaluation datasets.
- [**Advanced Evaluation (RAGAS)**](./scripts/evaluate_ragas.py): Evaluating RAG pipelines with RAGAS.
- [**Scaling RAG Pipes**](deep-dives/scaling-rag-pipes.md): Async workers, Caching, and Redis.
- [**LLM-as-a-Judge**](deep-dives/llm-as-a-judge.md): Verification and Grounding.
- [**Common RAG Pitfalls**](deep-dives/common-rag-pitfalls.md): Troubleshooting common failures.
- [**Architecture Patterns**](deep-dives/architecture-patterns.md): Ports, Adapters, and Modularity.

## 🎓 Educational & Curriculum 🆕

### Core Educational Resources
- [**AI Engineering Curriculum**](AI_ENGINEERING_CURRICULUM.md): Complete curriculum from beginner to expert in production-ready RAG systems.
- [**Educational Implementation Guide**](EDUCATIONAL_IMPLEMENTATION_GUIDE.md): Comprehensive walkthrough connecting theory to implementation.
- [**Educational Layer Guide**](EDUCATIONAL.md): Navigation map for educational content.

### Learning Pathways
- [**Complete Learning Pathway**](educational/complete_learning_pathway_guide.md): Structured learning journey from RAG beginner to AI architect.
- [**Mastery Journey**](MASTERY_JOURNEY.md): Skill tree and graduation requirements for RAG engineering mastery.
- [**Roadmap**](ROADMAP.md): Journey from "Hello World" to "AI Lead Engineer".

### Layer-Specific Guides
- [**Domain Layer Guide**](educational/domain_layer_guide.md): Pure business logic and entities.
- [**Application Layer Guide**](educational/application_layer_guide.md): Use cases and services.
- [**Adapters Layer Guide**](educational/adapters_layer_guide.md): Concrete implementations of external dependencies.
- [**API Layer Guide**](educational/api_layer_guide.md): FastAPI routes and controllers.
- [**Workers Layer Guide**](educational/workers_layer_guide.md): Background processing and task queues.
- [**Complete RAG Pipeline Guide**](educational/complete_rag_pipeline_guide.md): End-to-end system walkthrough.

### Extension and Development Guides
- [**Extension Development Guide**](educational/extension_development_guide.md): How to extend the RAG Engine while maintaining educational quality.
- [**Atomic Commit Practices**](educational/atomic_commit_practices_guide.md): Following senior engineer standards for commits.

### Hands-On Learning
- [**Comprehensive Hands-On Notebook**](../notebooks/educational/rag_engine_mini_comprehensive_guide.ipynb): Interactive learning notebook covering all system aspects.
- [**Practical Exercises Guide**](educational/practical_exercises_guide.md): Hands-on exercises to reinforce learning.

### Code Walkthroughs
- [**Ask Hybrid Use Case**](code-walkthroughs/ask-hybrid-usecase.md): Step-by-step tour of the main use case.
- [**Document Ingestion Process**](code-walkthroughs/document-ingestion.md): Complete walkthrough of document processing pipeline.
- [**Embedding Generation & Caching**](code-walkthroughs/embedding-generation-caching.md): How embeddings are created and cached efficiently.

### Architecture Decision Records (ADRs)
- [**PostgreSQL FTS vs Elasticsearch**](adr/001-postgres-fts-vs-elasticsearch.md): Search backend selection rationale.
- [**RRF vs Weighted Fusion**](adr/002-rrf-vs-weighted-fusion.md): Ranking algorithm decision.
- [**Cross-Encoder vs LLM Reranking**](adr/003-cross-encoder-vs-llm-rerank.md): Reranking approach comparison.
- [**Minimal Vector Payload Design**](adr/004-minimal-vector-payload.md): Storage architecture decision.
- [**Chunk Deduplication Strategy**](adr/005-chunk-dedup-design.md): Approach to handling duplicate content.

### Failure Modes & Troubleshooting
- [**Low Recall Issues**](failure_modes/low-recall.md): Diagnosing and resolving retrieval problems.
- [**Poor Re-Ranking**](failure_modes/bad-re-ranking.md): Identifying and fixing ranking issues.
- [**Hallucinations**](failure_modes/hallucinations.md): Preventing and mitigating hallucinations.
- [**Slow Latency**](failure_modes/slow-latency.md): Performance troubleshooting guide.
- [**Poor Chunking**](failure_modes/poor-chunking.md): Optimizing document segmentation.

### Exercises & Assessments
- [**Level 1: Semantic Embeddings**](exercises/level1_embeddings.md): Foundational embedding concepts.
- [**Level 2: Chunking Strategies**](exercises/level2_chunking.md): Document segmentation techniques.
- [**Level 3: Hybrid Search**](exercises/level3_hybrid_search.md): Combining different retrieval methods.
- [**Level 4: Re-Ranking**](exercises/level4_reranking.md): Improving result quality.
- [**Level 5: Evaluation**](exercises/level5_evaluation.md): Measuring system performance.
