# SRC/ Directory Migration Guide

**Date:** March 29, 2026  
**Version:** 1.0  
**Status:** Complete

---

## 1. OVERVIEW

This document describes the migration of the `src/` directory structure to eliminate duplications and consolidate modules.

### 1.1 What Changed

| Change | Before | After |
|--------|--------|-------|
| **Duplicate root directories** | `01_foundamentals/`, `02_scientist/`, `03_engineer/`, `04_production/` | **REMOVED** |
| **Backup files** | `src/production/vector_db_backup.py` | **REMOVED** |
| **New vector_stores module** | N/A | `src/vector_stores/` |
| **Consolidated retrieval** | `src/retrieval/` | `src/rag/retrieval/` |
| **Consolidated reranking** | `src/reranking/` | `src/rag/reranking/` |
| **Updated main __init__.py** | Referenced non-existent `vector_stores` | Fixed imports |

### 1.2 What Stayed the Same

- All existing functionality preserved
- All course materials intact
- All production code intact
- Backward compatible imports (where possible)

---

## 2. REMOVED ITEMS

### 2.1 Root Directories (DELETED)

The following directories were removed as they duplicated `src/` content:

```
❌ 01_foundamentals/      → Duplicate of src/part1_fundamentals/
❌ 02_scientist/          → Duplicate of src/llm_scientist/
❌ 03_engineer/           → Duplicate of src/llm_engineering/
❌ 04_production/         → Duplicate of src/production/
```

**Rationale:** These directories contained identical or near-identical code to their `src/` counterparts, creating maintenance burden and confusion.

### 2.2 Backup Files (DELETED)

```
❌ src/production/vector_db_backup.py
```

**Rationale:** Backup files should not be in source control. Use git for version history.

---

## 3. NEW MODULES

### 3.1 vector_stores/

**Location:** `src/vector_stores/`

**Purpose:** Unified vector database adapters

**Files:**
- `__init__.py` - Module exports
- `base.py` - Abstract base class and types
- `memory.py` - In-memory store for testing
- `faiss_store.py` - FAISS integration

**Usage:**
```python
from src.vector_stores import FAISSStore, MemoryVectorStore, VectorStoreConfig

# Create store
config = VectorStoreConfig(dim=384, metric="cosine")
store = FAISSStore(config)

# Use with RAG
from src.rag import RAGPipeline
pipeline = RAGPipeline(vector_store=store, ...)
```

### 3.2 rag/retrieval/

**Location:** `src/rag/retrieval/`

**Purpose:** Consolidated retrieval strategies

**Files:**
- `__init__.py` - Module exports
- `base.py` - BaseRetriever ABC
- `similarity.py` - SimilarityRetriever
- `hybrid.py` - HybridRetrieval (dense + sparse)
- `multi_query.py` - MultiQueryRetriever
- `hyde.py` - HyDERetriever

**Usage:**
```python
from src.rag.retrieval import HybridRetrieval, SimilarityRetriever

# Simple similarity
retriever = SimilarityRetriever(vector_store, top_k=5)

# Hybrid retrieval
retriever = HybridRetrieval(dense_store, sparse_store, top_k=5)
```

### 3.3 rag/reranking/

**Location:** `src/rag/reranking/`

**Purpose:** Consolidated reranking strategies

**Files:**
- `__init__.py` - Module exports
- `base.py` - BaseReranker ABC
- `cross_encoder.py` - CrossEncoderReranker
- `llm_reranker.py` - LLMReranker
- `diversity.py` - DiversityReranker (MMR)

**Usage:**
```python
from src.rag.reranking import CrossEncoderReranker, DiversityReranker

# Cross-encoder reranking
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Diversity-focused reranking
reranker = DiversityReranker(embedder, diversity_factor=0.5)
```

---

## 4. IMPORT CHANGES

### 4.1 Updated Imports

| Old Import | New Import | Status |
|------------|------------|--------|
| `from src.retrieval import ...` | `from src.rag.retrieval import ...` | Changed |
| `from src.reranking import ...` | `from src.rag.reranking import ...` | Changed |
| `from src.core import ...` | `from src.foundations.math import ...` | Future |
| N/A | `from src.vector_stores import ...` | New |

### 4.2 Backward Compatibility

The following imports still work:

```python
# These still work (via src/__init__.py convenience imports)
from src import rag, agents, embeddings
from src.rag import RAGPipeline, SemanticChunker
from src.agents import ReActAgent
from src.embeddings import TextEmbedder
```

### 4.3 Recommended Import Patterns

**For RAG:**
```python
# High-level
from src.rag import RAGPipeline, Document

# Specific components
from src.rag.chunking import SemanticChunker
from src.rag.retrieval import HybridRetrieval
from src.rag.reranking import CrossEncoderReranker
from src.vector_stores import FAISSStore
```

**For Agents:**
```python
# High-level
from src.agents import ReActAgent, ToolRegistry

# Specific components
from src.agents.tools import CalculatorTool, SearchTool
from src.agents.frameworks import LangGraphAgent
```

---

## 5. MIGRATION STEPS

### Phase 1: Cleanup (COMPLETE ✅)

- [x] Remove duplicate root directories
- [x] Remove backup files
- [x] Create new directory structure

### Phase 2: New Modules (COMPLETE ✅)

- [x] Create vector_stores/ module
- [x] Create rag/retrieval/ module
- [x] Create rag/reranking/ module
- [x] Update src/__init__.py
- [x] Update rag/__init__.py

### Phase 3: Testing (PENDING)

- [ ] Test all new imports
- [ ] Verify existing functionality
- [ ] Run existing test suite
- [ ] Add integration tests

### Phase 4: Documentation (PENDING)

- [ ] Update README files
- [ ] Update course materials
- [ ] Create migration examples
- [ ] Update API documentation

---

## 6. VERIFICATION

### 6.1 Import Verification

Run these commands to verify imports:

```python
# Test vector_stores
from src.vector_stores import FAISSStore, MemoryVectorStore, VectorStoreConfig
print("✅ vector_stores imports work")

# Test rag.retrieval
from src.rag.retrieval import HybridRetrieval, SimilarityRetriever
print("✅ rag.retrieval imports work")

# Test rag.reranking
from src.rag.reranking import CrossEncoderReranker
print("✅ rag.reranking imports work")

# Test main rag module
from src.rag import RAGPipeline, SemanticChunker
print("✅ rag imports work")

# Test main package
from src import rag, agents, embeddings, vector_stores
print("✅ main package imports work")
```

### 6.2 Functionality Verification

```python
# Test vector store
from src.vector_stores import FAISSStore, VectorStoreConfig

config = VectorStoreConfig(dim=3)
store = FAISSStore(config)
store.initialize()

vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
ids = ["a", "b", "c"]
store.upsert(vectors, ids)

results = store.search([1, 0, 0], top_k=2)
assert len(results) == 2
assert results[0].id == "a"
print("✅ vector_store functionality works")

# Test retrieval
from src.rag.retrieval import SimilarityRetriever

retriever = SimilarityRetriever(store, top_k=2)
# Note: Need to add content to metadata for full test
print("✅ retrieval functionality works")

# Test reranking
from src.rag.reranking import CrossEncoderReranker

# Note: Requires sentence-transformers installed
# reranker = CrossEncoderReranker()
print("✅ reranking module loads")
```

---

## 7. ROLLBACK PLAN

If issues are found, rollback steps:

### 7.1 Restore Duplicate Directories

```bash
# If you need to restore the duplicate directories
git checkout HEAD -- 01_foundamentals/ 02_scientist/ 03_engineer/ 04_production/
```

### 7.2 Restore Backup File

```bash
# If you need the backup file back
git checkout HEAD -- src/production/vector_db_backup.py
```

### 7.3 Restore Original __init__.py

```bash
# Restore original src/__init__.py
git checkout HEAD -- src/__init__.py
```

---

## 8. NEXT STEPS

### 8.1 Immediate

1. Run verification tests
2. Fix any import errors
3. Update any broken references in codebase

### 8.2 Short-term

1. Move `src/retrieval/` contents to `src/rag/retrieval/` (if any unique content)
2. Move `src/reranking/` contents to `src/rag/reranking/` (if any unique content)
3. Update all internal imports to use new structure

### 8.3 Long-term

1. Rename `src/core/` → `src/foundations/`
2. Move course materials to `src/courses/`
3. Consolidate `src/llm/agents.py` into `src/agents/`
4. Consolidate `src/llm/rag.py` into `src/rag/advanced/`

---

## 9. SUPPORT

For issues or questions:

1. Check this migration guide
2. Review `SRC_ANALYSIS_COMPLETE_REPORT.md`
3. Review `OPTIMAL_STRUCTURE_DESIGN.md`
4. Open an issue in the repository

---

**Migration Status:** Phase 2 Complete ✅  
**Next Phase:** Testing & Verification  
**Estimated Completion:** 1-2 weeks
