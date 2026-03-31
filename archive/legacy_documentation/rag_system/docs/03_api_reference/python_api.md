# Python API Reference

**Version**: 1.0.0  
**Last Updated**: March 27, 2026

---

## 🐍 Python API

Complete reference for using the RAG system from Python.

---

## Quick Start

```python
import asyncio
from src.integration import create_islamic_rag

async def main():
    # Initialize
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Query
    result = await rag.query("ما هو التوحيد في الإسلام؟")
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

asyncio.run(main())
```

---

## Core API

### `create_islamic_rag(config=None, **kwargs)`

Create and configure the RAG system.

**Parameters:**
- `config` (IslamicRAGConfig, optional): Configuration object
- `**kwargs`: Configuration overrides

**Returns:** `IslamicRAG` instance

**Example:**
```python
from src.integration import IslamicRAGConfig, create_islamic_rag

# With config object
config = IslamicRAGConfig(
    llm_provider="openai",
    chunk_size=512
)
rag = create_islamic_rag(config=config)

# With kwargs
rag = create_islamic_rag(
    llm_provider="anthropic",
    retrieval_top_k=10
)
```

---

### `IslamicRAG` Class

Main RAG system interface.

#### Methods

##### `async initialize(load_existing=True)`

Initialize the RAG system.

**Parameters:**
- `load_existing` (bool): Load existing indexes

**Example:**
```python
rag = create_islamic_rag()
await rag.initialize()
```

##### `async query(question, top_k=5, filters=None)`

Query the RAG system.

**Parameters:**
- `question` (str): User question
- `top_k` (int): Number of results (default: 5)
- `filters` (dict, optional): Metadata filters

**Returns:** `dict` with answer and sources

**Example:**
```python
# Basic query
result = await rag.query("ما هو التوحيد؟")

# With filters
result = await rag.query(
    "ما حكم الصلاة؟",
    filters={"category": "الفقه العام"}
)

# With more results
result = await rag.query("التوحيد", top_k=10)
```

##### `async query_tafsir(question)`

Query Tafsir (Quranic exegesis) specialist.

**Parameters:**
- `question` (str): Tafsir question

**Returns:** `dict` with domain-specific answer

**Example:**
```python
result = await rag.query_tafsir("ما تفسير سورة الإخلاص؟")
print(f"Domain: {result['domain_name']}")
```

##### `async query_hadith(question)`

Query Hadith specialist.

**Parameters:**
- `question` (str): Hadith question

**Example:**
```python
result = await rag.query_hadith("ما حديث إنما الأعمال بالنيات؟")
```

##### `async query_fiqh(question)`

Query Fiqh (jurisprudence) specialist.

**Example:**
```python
result = await rag.query_fiqh("ما شروط الصلاة؟")
```

##### `async compare_madhhabs(question)`

Compare rulings across 4 madhhabs.

**Parameters:**
- `question` (str): Fiqh question

**Returns:** `dict` with madhhab comparisons

**Example:**
```python
result = await rag.compare_madhhabs("ما حكم القنوت؟")

print(f"Consensus: {result['consensus']}")
for madhhab, view in result['madhhab_results'].items():
    print(f"{madhhab}: {view['answer'][:100]}...")
```

##### `async ask_as_researcher(question)`

Ask researcher agent.

**Example:**
```python
result = await rag.ask_as_researcher("ما أدلة وجود الله؟")
```

##### `async ask_as_student(topic)`

Get educational content for students.

**Example:**
```python
result = await rag.ask_as_student("الصلاة")
print(f"Lesson: {result['content']}")
```

##### `async ask_fatwa(question)`

Get fiqh research (not real fatwa).

**Example:**
```python
result = await rag.ask_fatwa("ما حكم الربا؟")
print(f"Disclaimer: {result.get('disclaimer', '')}")
```

##### `async index_documents(limit=None, categories=None)`

Index documents.

**Parameters:**
- `limit` (int, optional): Limit number of documents
- `categories` (list, optional): Filter by categories

**Example:**
```python
# Index sample
await rag.index_documents(limit=100)

# Index specific categories
await rag.index_documents(
    categories=["التفسير", "الحديث"]
)

# Index all
await rag.index_documents()
```

##### `get_stats()`

Get system statistics.

**Returns:** `dict` with statistics

**Example:**
```python
stats = rag.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

---

## Component API

### Direct Component Access

You can also use components directly:

```python
# Chunking
from src.processing.advanced_chunker import create_chunker

chunker = create_chunker(strategy="recursive")
chunks = chunker.chunk({"id": "1", "content": text})

# Embeddings
from src.processing.embedding_pipeline import create_embedding_pipeline

embeddings = create_embedding_pipeline()
result = await embeddings.embed_texts(["text1", "text2"])

# Retrieval
from src.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(...)
results = await retriever.search(query, top_k=5)

# Generation
from src.generation.generator import LLMClient, LLMProvider

llm = LLMClient(provider=LLMProvider.OPENAI)
response = await llm.generate(prompt)
```

---

## Configuration

### `IslamicRAGConfig`

Configuration dataclass.

**Fields:**
```python
@dataclass
class IslamicRAGConfig:
    # Paths
    datasets_path: str = "..."
    output_path: str = "..."
    
    # Embedding
    embedding_model: str = "sentence-transformers/..."
    embedding_device: str = "cpu"
    
    # LLM
    llm_provider: str = "mock"
    llm_model: str = "gpt-4o"
    
    # Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval
    retrieval_top_k: int = 50
    rerank_top_k: int = 5
    
    # Features
    enable_authority_ranking: bool = True
```

---

## Error Handling

### Common Exceptions

```python
from src.pipeline.complete_pipeline import RAGError

try:
    result = await rag.query("السؤال")
except RAGError as e:
    print(f"RAG error: {e}")
except Exception as e:
    print(f"General error: {e}")
```

### Error Types

- `RAGError` - Base RAG error
- `IndexError` - No index available
- `LLMError` - LLM generation error
- `RetrievalError` - Retrieval error

---

## Best Practices

### 1. Reuse Instances

```python
# ✅ Good - Reuse
rag = create_islamic_rag()
await rag.initialize()
for question in questions:
    result = await rag.query(question)

# ❌ Bad - Create new each time
for question in questions:
    rag = create_islamic_rag()
    await rag.initialize()
    result = await rag.query(question)
```

### 2. Handle Errors

```python
try:
    result = await rag.query(question)
except Exception as e:
    logger.error(f"Query failed: {e}")
    result = {"answer": "Sorry, I encountered an error"}
```

### 3. Use Filters

```python
# More efficient with filters
result = await rag.query(
    "ما حكم الصلاة؟",
    filters={"category": "الفقه العام"}
)
```

### 4. Batch Operations

```python
# Index in batches
await rag.index_documents(limit=100)
await rag.index_documents(limit=200)
```

---

## Examples

### Complete Example

```python
import asyncio
from src.integration import create_islamic_rag

async def main():
    # Initialize
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Check if indexed
    stats = rag.get_stats()
    if stats.get('total_chunks', 0) == 0:
        print("Indexing documents...")
        await rag.index_documents(limit=100)
    
    # Query
    questions = [
        "ما هو التوحيد؟",
        "ما شروط الصلاة؟",
        "ما أركان الإيمان؟"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = await rag.query(question, top_k=3)
        print(f"A: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])}")

asyncio.run(main())
```

---

## Related Documents

- [REST API Reference](rest_api.md) - HTTP API
- [Usage Examples](../04_user_guides/basic_queries.md) - Query examples
- [Configuration Guide](../01_getting_started/configuration.md) - Config options

---

**For more examples**, see [USAGE_EXAMPLES.md](../USAGE_EXAMPLES.md)
