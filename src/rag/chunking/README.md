# Chunking Module Documentation

Production-ready text chunking strategies for Retrieval-Augmented Generation (RAG) systems.

## Overview

This module provides unified, well-tested chunking implementations that consolidate 7 scattered implementations (~1,760 lines) into a single, maintainable codebase (~900 lines).

## Installation

The chunking module is part of the `src/rag` package. Ensure you have the required dependencies:

```bash
pip install tiktoken  # For token-aware chunking
pip install sentence-transformers  # For semantic chunking (optional)
```

## Quick Start

```python
from src.rag.chunking import create_chunker, Chunk

# Simple usage with default settings
chunker = create_chunker("recursive", chunk_size=512, chunk_overlap=50)

# Chunk a document
document = {
    "id": "doc_001",
    "content": "Your document text here...",
    "metadata": {"source": "example.pdf"}
}

chunks: list[Chunk] = chunker.chunk(document)

for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}: {chunk.word_count} words")
```

## Available Strategies

### 1. Fixed-Size Chunking (`fixed`)

Fast, predictable chunking with uniform chunk sizes.

```python
from src.rag.chunking import create_fixed_chunker

chunker = create_fixed_chunker(
    chunk_size=512,      # Tokens per chunk
    chunk_overlap=50,    # Overlap tokens
)
```

**Best for:** Speed-critical applications, uniform documents

**Limitations:** May break sentences and semantic units

---

### 2. Recursive Chunking (`recursive`) ⭐ RECOMMENDED

Structure-preserving chunking using hierarchical separators.

```python
from src.rag.chunking import create_recursive_chunker

chunker = create_recursive_chunker(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],  # Custom separators
)
```

**Best for:** Most general-purpose use cases (default recommendation)

**Features:**
- Preserves paragraphs, sentences, words
- Auto-detects Arabic text
- Code-aware with language detection

---

### 3. Semantic Chunking (`semantic`)

Embedding-based chunking that preserves semantic coherence.

```python
from src.rag.chunking import create_semantic_chunker

chunker = create_semantic_chunker(
    chunk_size=512,
    similarity_threshold=0.5,  # Lower = more chunks
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
```

**Best for:** High-value documents, context preservation

**Requirements:** `sentence-transformers` package

---

### 4. Hierarchical Chunking (`hierarchical`)

Parent-child relationships for multi-stage retrieval.

```python
from src.rag.chunking import create_hierarchical_chunker

chunker = create_hierarchical_chunker(
    parent_chunk_size=2000,  # Large chunks for context
    child_chunk_size=500,    # Small chunks for retrieval
    chunk_overlap=100,
)

result = chunker.chunk(document)
print(f"Created {len(result.children)} children, {len(result.parents)} parents")

# Retrieve children, expand to parents
parents = result.get_parents_for_children(["child_id_1", "child_id_2"])
```

**Best for:** Multi-stage retrieval, complex documents

---

### 5. Code-Aware Chunking (`code`)

Language-aware code splitting preserving structure.

```python
from src.rag.chunking import create_code_chunker

chunker = create_code_chunker(
    chunk_size=1000,
    language="python",  # or "auto" for detection
)

chunks = chunker.chunk({
    "id": "main.py",
    "content": "def main():\n    pass"
})
```

**Supported languages:** Python, JavaScript, TypeScript, Java, C++, Go, Rust

**Best for:** Code documentation, technical repositories

---

### 6. Token-Aware Chunking (`token_aware`)

Precise token counting using tiktoken.

```python
from src.rag.chunking import create_token_aware_chunker

chunker = create_token_aware_chunker(
    chunk_size=512,
    tokenizer_name="cl100k_base",  # GPT-4 tokenizer
)

# Utility functions
from src.rag.chunking import count_tokens, truncate_to_tokens

token_count = count_tokens("Hello, world!")
truncated = truncate_to_tokens(long_text, max_tokens=512)
```

**Best for:** LLM context window management, cost estimation

---

## Configuration

### ChunkingConfig

All chunkers accept a `ChunkingConfig` object:

```python
from src.rag.chunking import ChunkingConfig, ChunkingStrategy

config = ChunkingConfig(
    strategy=ChunkingStrategy.RECURSIVE,
    chunk_size=512,
    chunk_overlap=50,
    min_chunk_size=100,
    max_chunk_size=2000,
    keep_separator=False,
    tokenizer_name="cl100k_base",
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    similarity_threshold=0.5,
    parent_chunk_size=2000,
    child_chunk_size=500,
    language="auto",
)
```

### Factory Functions

Use factory functions for quick setup:

```python
from src.rag.chunking import create_chunker, get_recommended_config

# Quick creation
chunker = create_chunker(
    "recursive",
    chunk_size=512,
    chunk_overlap=50,
)

# Get recommended config for content type
config = get_recommended_config("code", chunk_size=1000)
chunker = create_chunker(config.strategy, config=config)
```

---

## Chunk Data Structure

Each chunk is a `Chunk` dataclass:

```python
from src.rag.chunking import Chunk

chunk = Chunk(
    content="The chunk text...",
    document_id="doc_001",
    start_index=0,
    end_index=100,
    metadata={"source": "file.pdf", "page": 1},
    parent_id=None,  # For hierarchical chunking
)

# Properties
print(chunk.word_count)    # Number of words
print(chunk.char_count)    # Number of characters
print(chunk.chunk_id)      # Auto-generated ID
print(chunk.is_empty)      # Check if empty

# Convert to/from dict
data = chunk.to_dict()
chunk = Chunk.from_dict(data)
```

---

## Advanced Usage

### Multiple Documents

```python
from src.rag.chunking import create_chunker

chunker = create_chunker("recursive")

texts = ["Document 1...", "Document 2...", "Document 3..."]
doc_ids = ["doc_001", "doc_002", "doc_003"]

all_chunks = chunker.chunk_texts(texts, doc_ids=doc_ids)
# all_chunks is a list of lists
```

### Custom Embedding Function

```python
from src.rag.chunking import create_semantic_chunker

def custom_embed(text: str) -> list[float]:
    # Your embedding logic
    return [0.1, 0.2, 0.3, ...]

chunker = create_semantic_chunker(
    embedding_function=custom_embed,
)
```

### ChunkerFactory

```python
from src.rag.chunking import ChunkerFactory, ChunkingStrategy

# Using enum
chunker = ChunkerFactory.create(ChunkingStrategy.SEMANTIC)

# Using string
chunker = ChunkerFactory.create("semantic")

# Get available strategies
strategies = ChunkerFactory.get_available_strategies()

# Get recommended strategy
strategy = ChunkerFactory.get_recommended_strategy("legal_documents")
```

---

## Migration Guide

### From Old Implementations

If you were using the old scattered implementations:

**Before:**
```python
# Old import paths
from rag_system.src.processing.advanced_chunker import AdvancedChunker
from src.llm_engineering.module_3_2.splitting import RecursiveSplitter
```

**After:**
```python
# New unified imports
from src.rag.chunking import create_chunker, RecursiveChunker

chunker = create_chunker("recursive")
```

### Backward Compatibility

The old implementations remain in place but are deprecated. Update your imports gradually:

| Old Path | New Path |
|----------|----------|
| `rag_system.src.processing.advanced_chunker` | `src.rag.chunking` |
| `src.llm_engineering.module_3_2.splitting` | `src.rag.chunking` |
| `research.rag_engine.chunking` | `src.rag.chunking` |

---

## Performance Guidelines

### Strategy Selection

| Use Case | Recommended Strategy | Expected Speed |
|----------|---------------------|----------------|
| General documents | Recursive | Fast |
| Code files | Code | Fast |
| Legal/Research | Semantic | Medium |
| Long books | Hierarchical | Fast |
| API/Real-time | Fixed/Token-aware | Fastest |

### Chunk Size Recommendations

| Content Type | Chunk Size | Overlap |
|--------------|------------|---------|
| General text | 512 tokens | 50 tokens |
| Code | 1000 tokens | 100 tokens |
| Legal documents | 768 tokens | 75 tokens |
| Technical docs | 512 tokens | 50 tokens |
| Conversations | 256 tokens | 25 tokens |

---

## Testing

Run the test suite:

```bash
pytest tests/rag/chunking/ -v --cov=src.rag.chunking
```

Expected coverage: 95%+

---

## Troubleshooting

### tiktoken Not Available

```
Warning: tiktoken not installed. Using character-based estimation.
```

**Solution:** `pip install tiktoken`

### sentence-transformers Not Available

```
Warning: sentence-transformers not installed. Falling back to recursive chunking.
```

**Solution:** `pip install sentence-transformers` (optional)

### Chunks Too Small/Large

Adjust `chunk_size` and `min_chunk_size`:

```python
chunker = create_chunker(
    "recursive",
    chunk_size=1024,      # Larger chunks
    min_chunk_size=200,   # Allow smaller minimum
)
```

---

## Contributing

When adding new chunking strategies:

1. Inherit from `BaseChunker`
2. Implement the `chunk()` method
3. Add comprehensive type hints
4. Include Google-style docstrings
5. Add tests with 95%+ coverage
6. Register in `ChunkerFactory`

---

## License

Part of AI-Mastery-2026 project.
