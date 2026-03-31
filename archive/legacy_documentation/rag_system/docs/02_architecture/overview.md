# Architecture Overview

**Version**: 1.0.0  
**Last Updated**: March 27, 2026

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  RAG PIPELINE 2026                          │
├─────────────────────────────────────────────────────────────┤
│  Data Sources → Ingestion → Processing → Vector Store      │
│       ↓                                                      │
│  Query → Transform → Retrieve → Rerank → Generate         │
│       ↓                                                      │
│  Evaluate → Monitor → Log                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Core Components

### 1. Data Ingestion Layer

**Purpose**: Load and process documents from multiple sources

**Components**:
- `MultiSourceIngestionPipeline` - Main ingestion orchestrator
- `FileConnector` - File system ingestion
- `APIConnector` - REST API ingestion
- `DatabaseConnector` - Database ingestion
- `WebhookConnector` - Real-time webhook ingestion

**Features**:
- Incremental updates (delta sync)
- Checksum validation
- Multiple format support (PDF, DOCX, TXT, MD, HTML, JSON)
- Metadata extraction

**File**: `src/data/multi_source_ingestion.py`

---

### 2. Processing Layer

**Purpose**: Transform raw text into searchable chunks

**Components**:
- `AdvancedChunker` - 6 chunking strategies
- `EmbeddingPipeline` - Multi-provider embeddings
- `ArabicProcessor` - Arabic text optimization

**Chunking Strategies**:
1. **Fixed** - Fast, predictable
2. **Recursive** - Structure-preserving (recommended)
3. **Semantic** - Context-aware
4. **Late** - Global context
5. **Agentic** - Query-aware
6. **Islamic** - Domain-specific (Quran, Hadith, Fiqh)

**File**: `src/processing/`

---

### 3. Storage Layer

**Purpose**: Store and index embeddings for fast retrieval

**Components**:
- `VectorStore` - Vector database abstraction
- `BM25Index` - Keyword search index
- `MemoryVectorStore` - In-memory storage
- `QdrantVectorStore` - Production vector DB
- `ChromaDBVectorStore` - Lightweight vector DB

**Features**:
- HNSW indexing for fast search
- Metadata filtering
- Batch operations
- Persistence

**File**: `src/retrieval/vector_store.py`

---

### 4. Retrieval Layer

**Purpose**: Find relevant documents for queries

**Components**:
- `HybridRetriever` - Semantic + BM25 search
- `QueryTransformer` - Query enhancement
- `Reranker` - Cross-encoder reranking

**Retrieval Strategies**:
- **Hybrid Search** - Semantic (60%) + BM25 (40%)
- **Reciprocal Rank Fusion** - Combine multiple retrievers
- **Adaptive Retrieval** - Adjust based on query type

**File**: `src/retrieval/`

---

### 5. Generation Layer

**Purpose**: Generate answers from retrieved context

**Components**:
- `LLMClient` - Multi-provider LLM abstraction
- `RAGGenerator` - RAG-specific generation
- `ResponseGuardrails` - Quality and safety checks

**Supported Providers**:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (local models)
- Mock (for testing)

**File**: `src/generation/generator.py`

---

### 6. Specialization Layer

**Purpose**: Domain-specific expertise

**Components**:
- `IslamicScholar` - 8 domain specialists
- `ComparativeFiqhScholar` - 4 madhhab comparison
- `AuthorityRanker` - Source authority ranking
- `CrossReferenceSystem` - Concept relationships
- `MultiHopReasoning` - Multi-step reasoning

**Domain Specialists**:
1. **Quran/Tafsir** - Quranic exegesis
2. **Hadith** - Prophetic traditions
3. **Fiqh** - Islamic jurisprudence
4. **Aqeedah** - Islamic theology
5. **Arabic Language** - Linguistics
6. **History** - Islamic history
7. **Spirituality** - Islamic spirituality
8. **Literature** - Arabic literature

**File**: `src/specialists/`

---

### 7. Agent Layer

**Purpose**: Specialized agents for complex tasks

**Components**:
- `IslamicRAGAgent` - Base agent class
- `EnhancedIslamicRAGAgent` - Advanced agents
- `AgentTeam` - Multi-agent collaboration

**Agent Roles**:
1. **Muhaqqiq** - Researcher
2. **Mufti** - Fiqh researcher
3. **Mufassir** - Tafsir specialist
4. **Muhaddith** - Hadith specialist
5. **Lughawi** - Linguist
6. **Muarrikh** - Historian
7. **Murabbi** - Educator
8. **Muqarin** - Comparative scholar

**File**: `src/agents/`

---

### 8. Evaluation Layer

**Purpose**: Measure system quality

**Components**:
- `RAGEvaluator` - Standard RAG metrics
- `IslamicRAGEvaluator` - Islamic-specific metrics
- `ArabicTestDataset` - Pre-built test sets

**Metrics**:
- **Retrieval**: Precision@K, Recall@K, MRR, NDCG
- **Generation**: Faithfulness, Relevance
- **Islamic**: Authority score, Source authenticity, Evidence presence

**File**: `src/evaluation/`

---

### 9. Monitoring Layer

**Purpose**: Track system performance and costs

**Components**:
- `RAGMonitor` - Central monitoring
- `CostTracker` - Cost tracking and budgeting
- `QueryLogger` - Query logging and analytics

**Tracked Metrics**:
- Query latency (P50, P95, P99)
- Token usage
- Cost per query
- Success rate
- Top queries

**File**: `src/monitoring/monitoring.py`

---

### 10. API Layer

**Purpose**: REST API for external access

**Components**:
- `FastAPI` application
- Query endpoints
- Indexing endpoints
- Streaming support

**Endpoints**:
- `POST /api/v1/query` - Query the system
- `POST /api/v1/query/stream` - Streaming queries
- `POST /api/v1/index` - Start indexing
- `GET /api/v1/stats` - Get statistics
- `GET /health` - Health check

**File**: `src/api/service.py`

---

## 🔄 Data Flow

### Indexing Flow

```
1. Documents loaded from sources
         ↓
2. Parsed and cleaned
         ↓
3. Chunked using strategy
         ↓
4. Embeddings generated
         ↓
5. Stored in vector DB + BM25 index
         ↓
6. Metadata indexed
```

### Query Flow

```
1. User submits query
         ↓
2. Query transformed (rewritten, expanded)
         ↓
3. Hybrid retrieval (semantic + BM25)
         ↓
4. Reranking with cross-encoder
         ↓
5. Top-K results selected
         ↓
6. LLM generates answer with context
         ↓
7. Response with citations returned
         ↓
8. Query logged and evaluated
```

---

## 📊 Component Interaction

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Data      │────▶│ Processing  │────▶│   Storage   │
│  Ingestion  │     │   Layer     │     │    Layer    │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   API       │◀────│ Generation  │◀────│  Retrieval  │
│   Layer     │     │   Layer     │     │    Layer    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Users     │     │ Specialists │     │ Monitoring  │
│             │     │   & Agents  │     │ & Evaluation│
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## 🎯 Design Principles

### 1. Modularity

Each component is independent and replaceable:
- Swap embedding providers without code changes
- Change vector databases via configuration
- Add new chunking strategies easily

### 2. Extensibility

Easy to add new features:
- New data sources via connectors
- New LLM providers via adapter pattern
- New agent roles via inheritance

### 3. Performance

Optimized for speed:
- Caching at multiple levels
- Batch processing
- Async operations
- HNSW indexing

### 4. Observability

Full visibility:
- Comprehensive logging
- Metrics collection
- Cost tracking
- Query analytics

### 5. Islamic Domain Optimization

Specialized for Islamic texts:
- Verse and hadith preservation
- Madhhab-aware retrieval
- Authority-based ranking
- Cross-reference system

---

## 📁 File Structure

```
rag_system/
├── src/
│   ├── data/              # Ingestion layer
│   ├── processing/        # Processing layer
│   ├── retrieval/         # Retrieval + Storage
│   ├── generation/        # Generation layer
│   ├── specialists/       # Specialization layer
│   ├── agents/            # Agent layer
│   ├── evaluation/        # Evaluation layer
│   ├── monitoring/        # Monitoring layer
│   ├── api/               # API layer
│   └── pipeline/          # Orchestration
├── docs/                  # Documentation
├── config/                # Configuration
├── data/                  # Runtime data
└── logs/                  # Logs
```

---

## 🔗 Related Documents

- [Component Details](components.md) - Deep dive into each component
- [Data Flow](data_flow.md) - Detailed data flow diagrams
- [Design Decisions](design_decisions.md) - Why we made these choices

---

**Next**: Read [Component Details](components.md) for in-depth information
