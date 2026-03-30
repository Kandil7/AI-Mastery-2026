# Chunking Module Migration Guide

**Version:** 1.0.0  
**Date:** March 29, 2026  
**Status:** Ready for Migration

---

## Overview

This guide helps you migrate from the old scattered chunking implementations to the new unified `src.rag.chunking` module.

---

## Migration Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| 1 | Week 1 | Update new code, add deprecation warnings |
| 2 | Week 2 | Refactor existing code incrementally |
| 3 | Week 3 | Remove deprecated imports |
| 4 | Week 4 | Delete old implementations |

---

## Quick Reference

### Old → New Import Mapping

| Old Import | New Import | Notes |
|------------|------------|-------|
| `from rag_system.src.processing.advanced_chunker import AdvancedChunker` | `from src.rag.chunking import create_chunker` | Use factory function |
| `from rag_system.src.processing.advanced_chunker import FixedSizeChunker` | `from src.rag.chunking import FixedSizeChunker` | Direct import |
| `from src.llm_engineering.module_3_2.splitting import RecursiveSplitter` | `from src.rag.chunking import RecursiveChunker` | Renamed class |
| `from src.llm_engineering.module_3_2.splitting import TextSplitterFactory` | `from src.rag.chunking import ChunkerFactory` | Renamed class |
| `from research.rag_engine.chunking import chunk_text_token_aware` | `from src.rag.chunking import create_token_aware_chunker` | Use factory |
| `from research.week5-backend.rag.chunking import simple_chunk` | `from src.rag.chunking import create_chunker` | Use factory |

---

## Migration Examples

### Example 1: Basic Chunking

**Before:**
```python
from rag_system.src.processing.advanced_chunker import AdvancedChunker

chunker = AdvancedChunker(strategy="recursive")
chunks = chunker.chunk(document)
```

**After:**
```python
from src.rag.chunking import create_chunker

chunker = create_chunker("recursive")
chunks = chunker.chunk(document)
```

---

### Example 2: Fixed-Size Chunking

**Before:**
```python
from rag_system.src.processing.advanced_chunker import FixedSizeChunker, ChunkingConfig

config = ChunkingConfig(strategy="fixed", chunk_size=512)
chunker = FixedSizeChunker(config)
chunks = chunker.chunk(document)
```

**After:**
```python
from src.rag.chunking import create_fixed_chunker

chunker = create_fixed_chunker(chunk_size=512)
chunks = chunker.chunk(document)
```

---

### Example 3: Recursive Chunking

**Before:**
```python
from src.llm_engineering.module_3_2.splitting import RecursiveSplitter

splitter = RecursiveSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

**After:**
```python
from src.rag.chunking import create_recursive_chunker

chunker = create_recursive_chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(document)
```

---

### Example 4: Semantic Chunking

**Before:**
```python
from rag_system.src.processing.advanced_chunker import SemanticChunker

chunker = SemanticChunker(config)
chunks = chunker.chunk(document)
```

**After:**
```python
from src.rag.chunking import create_semantic_chunker

chunker = create_semantic_chunker(
    chunk_size=512,
    similarity_threshold=0.5,
)
chunks = chunker.chunk(document)
```

---

### Example 5: Hierarchical Chunking

**Before:**
```python
from src.llm_engineering.module_3_2.splitting import HierarchicalSplitter

splitter = HierarchicalSplitter(
    parent_chunk_size=2000,
    child_chunk_size=500,
)
children, parent_map = splitter.split(text)
```

**After:**
```python
from src.rag.chunking import create_hierarchical_chunker

chunker = create_hierarchical_chunker(
    parent_chunk_size=2000,
    child_chunk_size=500,
)
result = chunker.chunk(document)
children = result.children
parents = result.parents
```

---

### Example 6: Code Chunking

**Before:**
```python
from src.llm_engineering.module_3_2.splitting import CodeSplitter

splitter = CodeSplitter(language="python", chunk_size=1000)
chunks = splitter.split_text(code)
```

**After:**
```python
from src.rag.chunking import create_code_chunker

chunker = create_code_chunker(
    language="python",
    chunk_size=1000,
)
chunks = chunker.chunk({"id": "code.py", "content": code})
```

---

### Example 7: Token Counting

**Before:**
```python
from research.rag_engine.chunking import count_tokens, truncate_to_tokens

token_count = count_tokens(text)
truncated = truncate_to_tokens(text, max_tokens=512)
```

**After:**
```python
from src.rag.chunking import count_tokens, truncate_to_tokens

token_count = count_tokens(text)
truncated = truncate_to_tokens(text, max_tokens=512)
```

Function signatures remain the same!

---

### Example 8: Factory Pattern

**Before:**
```python
from src.llm_engineering.module_3_2.splitting import TextSplitterFactory

splitter = TextSplitterFactory.create("recursive", chunk_size=512)
```

**After:**
```python
from src.rag.chunking import ChunkerFactory

chunker = ChunkerFactory.create("recursive", chunk_size=512)
```

---

## API Changes Summary

### Class Name Changes

| Old Name | New Name | Module |
|----------|----------|--------|
| `RecursiveSplitter` | `RecursiveChunker` | `src.rag.chunking` |
| `TokenSplitter` | `TokenAwareChunker` | `src.rag.chunking` |
| `TextSplitterFactory` | `ChunkerFactory` | `src.rag.chunking` |
| `AdvancedChunker` | `create_chunker()` | `src.rag.chunking` |

### Method Name Changes

| Old Method | New Method | Notes |
|------------|------------|-------|
| `split_text(text)` | `chunk(document)` | Now takes dict |
| `split_documents(docs)` | `chunk_texts(texts)` | Renamed |
| `create_chunk(...)` | `_create_chunk(...)` | Internal method |

### Data Class Changes

**Chunk:**
- Old: `TextChunk` with `id`, `content`, `metadata`
- New: `Chunk` with `chunk_id`, `content`, `document_id`, `metadata`, `parent_id`

**Config:**
- Old: Various config classes
- New: Unified `ChunkingConfig`

---

## Step-by-Step Migration Process

### Step 1: Audit Existing Code

Find all chunking imports:

```bash
# Search for old imports
grep -r "from.*advanced_chunker" src/
grep -r "from.*splitting import" src/
grep -r "from.*chunking import" research/
```

### Step 2: Update Imports

Replace old imports with new ones using the mapping table above.

### Step 3: Update Method Calls

Update method calls to match new API:

```python
# Old
chunks = chunker.split_text(text)

# New
chunks = chunker.chunk({"id": "doc1", "content": text})
```

### Step 4: Update Data Access

Update how you access chunk properties:

```python
# Old
chunk_id = chunk.id
text = chunk.content

# New (same for content)
chunk_id = chunk.chunk_id
text = chunk.content
```

### Step 5: Test Thoroughly

Run your test suite to ensure everything works:

```bash
pytest tests/ -v --cov=src.rag.chunking
```

---

## Deprecation Strategy

### Phase 1: Add Deprecation Warnings (Week 1)

Add to old modules:

```python
import warnings

warnings.warn(
    "advanced_chunker is deprecated. Use src.rag.chunking instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

### Phase 2: Update Documentation (Week 2)

Update all documentation to reference new module.

### Phase 3: Remove Old Code (Week 4)

After confirming all code is migrated, delete old implementations.

---

## Troubleshooting

### Issue: Import Error After Migration

**Solution:** Ensure you're importing from the correct path:

```python
# Correct
from src.rag.chunking import create_chunker

# Incorrect
from src.rag import create_chunker  # Missing .chunking
```

### Issue: Method Signature Mismatch

**Solution:** The new `chunk()` method takes a dictionary:

```python
# Old
chunks = chunker.split_text("text content")

# New
chunks = chunker.chunk({
    "id": "doc1",
    "content": "text content",
    "metadata": {}
})
```

### Issue: Missing Features

If you need a feature that was in the old implementation:

1. Check the new module's README.md
2. The feature may have a different name
3. If truly missing, create an issue for the team

---

## Testing Your Migration

### Quick Test Script

```python
"""Test script to verify migration."""

from src.rag.chunking import (
    create_chunker,
    ChunkingStrategy,
    Chunk,
)

# Test 1: Create chunker
chunker = create_chunker("recursive", chunk_size=100)

# Test 2: Chunk document
doc = {
    "id": "test",
    "content": "This is test content. " * 10,
    "metadata": {"source": "test.txt"}
}

chunks = chunker.chunk(doc)

# Test 3: Verify chunks
assert len(chunks) > 0
assert all(isinstance(c, Chunk) for c in chunks)
assert all(c.content for c in chunks)

print("✓ Migration test passed!")
```

---

## Support

For questions or issues:

1. Check the module README: `src/rag/chunking/README.md`
2. Review test examples: `tests/rag/chunking/test_chunking.py`
3. Contact the AI-Mastery-2026 team

---

## Checklist

Use this checklist for each file you migrate:

- [ ] Find old imports
- [ ] Replace with new imports
- [ ] Update method calls
- [ ] Update data access
- [ ] Run tests
- [ ] Commit changes

---

**Last Updated:** March 29, 2026  
**Version:** 1.0.0
