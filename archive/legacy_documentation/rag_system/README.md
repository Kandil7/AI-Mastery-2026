# Arabic Islamic Literature RAG System

Production-grade Retrieval-Augmented Generation system for Arabic Islamic literature corpus (2026).

## 📋 Overview

This RAG system enables intelligent querying of 8,425+ Islamic books across multiple domains:

- **التفسير** (Tafsir - Quranic Exegesis)
- **الحديث** (Hadith - Prophetic Traditions)
- **الفقه** (Fiqh - Islamic Jurisprudence)
- **العقيدة** (Aqeedah - Theology)
- **اللغة العربية** (Arabic Language)
- **التاريخ** (Islamic History)
- **الرقائق** (Spirituality)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  RAG Pipeline 2026                          │
├─────────────────────────────────────────────────────────────┤
│  Data Sources → Ingestion → Processing → Vector Store      │
│       ↓                                                      │
│  Query → Transform → Retrieve → Rerank → Generate         │
│       ↓                                                      │
│  Evaluate → Monitor → Log                                  │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Components

| Component | File | Description |
|-----------|------|-------------|
| **Ingestion** | `src/data/multi_source_ingestion.py` | Multi-source connectors (files, APIs, DBs, webhooks) |
| **Chunking** | `src/processing/advanced_chunker.py` | 6 strategies (fixed, recursive, semantic, late, agentic, islamic) |
| **Embeddings** | `src/processing/embedding_pipeline.py` | Multi-provider (Sentence Transformers, OpenAI, Cohere) |
| **Vector Store** | `src/retrieval/vector_store.py` | Qdrant, ChromaDB, In-memory support |
| **Retrieval** | `src/retrieval/hybrid_retriever.py` | Hybrid (semantic + BM25) + Reranking |
| **Query Transform** | `src/retrieval/query_transformer.py` | Rewriting, Decomposition, HyDE, Step-back |
| **Generation** | `src/generation/generator.py` | Multi-provider LLM (OpenAI, Anthropic, Ollama) |
| **Specialists** | `src/specialists/islamic_scholars.py` | Domain experts (Tafsir, Hadith, Fiqh, etc.) |
| **Agents** | `src/agents/enhanced_agents.py` | 8 specialized agent roles |
| **Evaluation** | `src/evaluation/evaluator.py` | Islamic-specific metrics |
| **Monitoring** | `src/monitoring/monitoring.py` | Cost tracking, query logs |
| **API** | `src/api/service.py` | FastAPI REST endpoints |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core
pip install numpy pydantic python-multipart

# Embeddings
pip install sentence-transformers torch torchvision

# Vector DBs
pip install qdrant-client chromadb

# LLMs
pip install openai anthropic

# Document Parsing
pip install pdfplumber python-docx beautifulsoup4

# API
pip install fastapi uvicorn

# Other
pip install aiohttp rank-bm25 scikit-learn
```

### 2. Set Environment Variables

Create `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Basic Usage

```python
import asyncio
from rag_system import create_islamic_rag

async def main():
    # Create and initialize
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Query
    result = await rag.query("ما هو التوحيد في الإسلام؟")
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

asyncio.run(main())
```

### 4. Index Documents

```python
async def index_docs():
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Index with limit for testing
    await rag.index_documents(limit=100)
    
    # Or index all (takes hours)
    # await rag.index_documents()

asyncio.run(index_docs())
```

### 5. Start API Server

```bash
uvicorn rag_system.src.api.service:app --reload --host 0.0.0.0 --port 8000
```

### 6. Query API

```bash
# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هو التوحيد؟", "top_k": 5}'

# Get stats
curl http://localhost:8000/stats

# Start indexing
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"limit": 100}'
```

## 📊 Dataset

### Structure

```
datasets/
├── extracted_books/         # 8,425 Islamic books
│   ├── 1_الفواكه_العذاب.txt
│   ├── ...
│   └── 10999_*.txt
├── arabic_web/
│   └── arabicweb24_clean.txt
├── Sanadset 368K Data on Hadith Narrators/
├── metadata/
│   ├── books.json          # Book metadata
│   ├── authors.json        # Author info
│   ├── categories.json     # Categories
│   └── guid_index.json     # Index
└── system_book_datasets/
```

### Categories

| Category | Books | Description |
|----------|-------|-------------|
| التفسير | 500+ | Quranic exegesis |
| كتب السنة | 300+ | Hadith collections |
| الفقه الحنفي | 800+ | Hanafi jurisprudence |
| الفقه المالكي | 600+ | Maliki jurisprudence |
| الفقه الشافعي | 700+ | Shafi'i jurisprudence |
| الفقه الحنبلي | 500+ | Hanbali jurisprudence |
| العقيدة | 400+ | Islamic theology |
| اللغة العربية | 600+ | Arabic language |
| التاريخ | 500+ | Islamic history |
| التراجم | 400+ | Biographies |

## 🔧 Configuration

### config.yaml

```yaml
# RAG System Configuration

data:
  datasets_path: "K:/learning/technical/ai-ml/AI-Mastery-2026/datasets"
  output_path: "K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data"

embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  batch_size: 32
  max_length: 512

vector_db:
  type: "qdrant"
  collection_name: "arabic_islamic_literature"
  vector_size: 768

retrieval:
  top_k: 5
  hybrid_weights:
    semantic: 0.7
    bm25: 0.3

llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.3
```

## 🎯 Agent Roles

| Role | Arabic | Description |
|------|--------|-------------|
| **Muhaqqiq** | محقق | Deep researcher |
| **Mufti** | مفتي | Fiqh researcher (not real fatwa) |
| **Mufassir** | مفسر | Quranic exegesis specialist |
| **Muhaddith** | محدث | Hadith specialist |
| **Lughawi** | لغوي | Arabic linguist |
| **Muarrikh** | مؤرخ | Islamic historian |
| **Murabbi** | مربّي | Educator |
| **Muqarin** | مقارن | Comparative scholar |

### Use Agent

```python
from rag_system import create_islamic_rag

rag = create_islamic_rag()
await rag.initialize()

# Use specific agent
result = await rag.ask_as_researcher("ما حكم الصلاة؟")
result = await rag.ask_as_student("التوحيد")
result = await rag.ask_fatwa("ما حكم الزكاة؟")

# Compare madhhabs
result = await rag.compare_madhhabs("الصلاة")
```

## 📈 Evaluation

### Islamic-Specific Metrics

- **Source Authenticity**: Authority of cited sources
- **Evidence Presence**: Quran, Hadith, Scholar citations
- **Madhhab Coverage**: Balance across 4 madhhabs
- **Citation Quality**: Proper attribution
- **Bias Detection**: Sectarian language detection

### Run Evaluation

```python
from rag_system.src.evaluation.evaluator import RAGEvaluator, ArabicTestDataset

evaluator = RAGEvaluator(pipeline)
samples = ArabicTestDataset.get_samples(language="arabic")
results = await evaluator.evaluate_dataset(samples[:10])

print(f"Precision@K: {results['retrieval']['precision_at_k']:.3f}")
print(f"Recall@K: {results['retrieval']['recall_at_k']:.3f}")
print(f"MRR: {results['retrieval']['mrr']:.3f}")
```

## 🔍 Monitoring

### Cost Tracking

```python
from rag_system.src.monitoring.monitoring import get_monitor

monitor = get_monitor()

# Log query
monitor.log_query(
    query="ما هو التوحيد؟",
    latency_ms=150,
    tokens_input=100,
    tokens_output=200,
    retrieval_count=5,
)

# Get dashboard
metrics = monitor.get_dashboard_metrics()
print(f"Daily spend: ${metrics['cost']['daily']['spent']:.2f}")
print(f"Queries (24h): {metrics['queries_last_24h']['query_count']}")
```

## 📝 Best Practices

### 1. Chunking Strategy

| Document Type | Strategy | Chunk Size |
|---------------|----------|------------|
| Quran/Tafsir | islamic | 384 |
| Hadith | islamic | 256 |
| Fiqh | islamic | 512 |
| General | recursive | 512 |

### 2. Retrieval Optimization

```python
# High precision (fewer, better results)
weights = {"semantic": 0.7, "bm25": 0.3}
top_k = 3

# High recall (more results)
weights = {"semantic": 0.5, "bm25": 0.5}
top_k = 10
```

### 3. Cost Optimization

- Use caching for embeddings
- Use local models when possible
- Set budget limits
- Monitor query patterns

## 🐛 Troubleshooting

### Out of Memory

```python
# Reduce batch size
pipeline = MultiSourceIngestionPipeline(batch_size=50)

# Use smaller embedding model
config = EmbeddingConfig(model="bert-base-multilingual-cased")
```

### Slow Indexing

```python
# Use GPU for embeddings
config = EmbeddingConfig(device="cuda")

# Process in parallel
pipeline = MultiSourceIngestionPipeline(max_concurrent=10)
```

### Poor Retrieval

```python
# Enable reranking
reranker = Reranker(model_name="BAAI/bge-reranker-large")

# Adjust weights
weights = {"semantic": 0.5, "bm25": 0.5}
```

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Version**: 1.0.0  
**Last Updated**: March 27, 2026  
**Status**: Production Ready
