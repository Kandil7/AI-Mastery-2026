# 🧩 Concept → Code Map

This document maps key RAG concepts to their corresponding implementations in the codebase, helping learners understand the relationship between theory and practice.

## Core Concepts & Code Locations

### 1. Embeddings
- **Concept**: Vector representation of text for semantic similarity
- **Code Files**:
  - `src/core/embeddings.py` - Core embedding interfaces and implementations
  - `src/app/application/ports/embeddings.py` - Embedding provider abstraction
  - `src/app/application/services/embedding_cache.py` - Caching mechanism to reduce costs
- **Design Rationale**:
  - Port abstraction allows easy switching between providers (OpenAI, HuggingFace, etc.)
  - Caching reduces API calls and improves performance

### 2. Document Ingestion & Processing
- **Concept**: Loading, parsing, and preparing documents for retrieval
- **Code Files**:
  - `src/core/document_loader.py` - Document loading utilities
  - `src/core/chunking.py` - Text splitting strategies (recursive, semantic)
  - `src/app/domain/models/document.py` - Document domain model
- **Design Rationale**:
  - Separation of loading, parsing, and chunking concerns
  - Configurable chunk sizes and overlap for different use cases

### 3. Hybrid Search
- **Concept**: Combining lexical (keyword) and semantic (vector) search for improved recall
- **Code Files**:
  - `src/app/infrastructure/repositories/keyword_store.py` - PostgreSQL FTS implementation
  - `src/app/infrastructure/repositories/qdrant_store.py` - Vector store implementation
  - `src/app/application/services/fusion.py` - Result fusion algorithms (RRF, weighted)
- **Design Rationale**:
  - Multiple retrieval strategies improve coverage
  - Reciprocal Rank Fusion (RRF) combines results effectively without parameter tuning

### 4. Re-ranking
- **Concept**: Improving result relevance by re-ordering retrieved documents
- **Code Files**:
  - `src/app/application/services/reranker.py` - Re-ranking service
  - `src/app/infrastructure/rerankers/cross_encoder_reranker.py` - Local cross-encoder implementation
  - `src/app/infrastructure/rerankers/llm_reranker.py` - LLM-based re-ranking
- **Design Rationale**:
  - Local cross-encoders offer better latency/cost trade-off than LLM re-ranking
  - Pluggable architecture allows experimentation with different re-ranking approaches

### 5. Retrieval-Augmented Generation (RAG)
- **Concept**: Using retrieved documents to augment LLM responses
- **Code Files**:
  - `src/app/application/use_cases/ask_hybrid.py` - Main RAG orchestration
  - `src/app/application/services/prompt_builder.py` - Context-aware prompt construction
  - `src/app/domain/models/query.py` - Query domain model
- **Design Rationale**:
  - Clear separation between retrieval and generation phases
  - Context-aware prompting for better answer quality

### 6. Evaluation & Metrics
- **Concept**: Measuring RAG system effectiveness
- **Code Files**:
  - `src/app/application/services/evaluation.py` - Evaluation framework
  - `src/app/domain/models/evaluation.py` - Evaluation metrics and results
  - `notebooks/evaluation_metrics.ipynb` - Interactive evaluation notebook
- **Design Rationale**:
  - Quantitative measurement of system performance
  - Iterative improvement based on metrics

### 7. Caching & Optimization
- **Concept**: Improving performance and reducing costs
- **Code Files**:
  - `src/app/infrastructure/cache/redis_cache.py` - Redis-based caching
  - `src/app/application/services/query_cache.py` - Query result caching
  - `src/app/infrastructure/repositories/embedding_cache.py` - Embedding caching
- **Design Rationale**:
  - Multiple layers of caching for optimal performance
  - Cost reduction through result and embedding reuse

## Learning Path Suggestions

### Beginner Path
1. Start with `src/core/embeddings.py` to understand vector representations
2. Explore `src/core/chunking.py` to learn about document preprocessing
3. Review `notebooks/basic_rag_example.ipynb` for end-to-end flow

### Intermediate Path
1. Study `src/app/application/use_cases/ask_hybrid.py` for orchestration patterns
2. Examine `src/app/infrastructure/repositories/keyword_store.py` and `qdrant_store.py` for retrieval implementations
3. Understand `src/app/application/services/fusion.py` for result combination

### Advanced Path
1. Analyze `src/app/application/services/reranker.py` for relevance optimization
2. Review `src/app/application/services/evaluation.py` for performance measurement
3. Investigate `src/app/infrastructure/cache/redis_cache.py` for system optimization

## Key Design Principles

1. **Separation of Concerns**: Clear boundaries between domain, application, and infrastructure layers
2. **Port/Adapter Architecture**: Ports define contracts, adapters implement specific technologies
3. **Configurability**: System behavior adjustable through configuration files
4. **Observability**: Built-in logging, metrics, and monitoring capabilities
5. **Testability**: Modular design enables comprehensive unit and integration testing