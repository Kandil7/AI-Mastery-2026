# Week 2: Chunking Consolidation Plan

## Current State Analysis

### Identified Chunking Implementations (7 total)

| File | Lines | Strategies | Key Features |
|------|-------|------------|--------------|
| `rag_system/src/processing/advanced_chunker.py` | 1,357 | Fixed, Recursive, Semantic, Late, Agentic, Islamic | Arabic optimization, Islamic text handling |
| `src/llm_engineering/module_3_2_building_vector_storage/splitting.py` | ~700 | Recursive, Token, Semantic, Code, Hierarchical | Parent-child relationships, code-aware |
| `research/rag_engine/rag-engine-mini/src/application/services/chunking.py` | ~200 | Token-aware, Hierarchical | Token-based with tiktoken |
| `research/week5-backend/week5_backend/rag/chunking.py` | ~100 | Simple, Structured | Word-based, heading-aware |
| `research/rag_engine/rag-engine-mini/src/domain/entities.py` | - | ChunkSpec, Chunk entities | Domain models |
| `rag_system/src/processing/advanced_chunker.py` (Chunk/ChunkingConfig) | - | Data classes | Shared data models |
| Various imports/utilities | - | - | Supporting code |

**Total Duplicate Code: ~1,760 lines**

### Common Interfaces Identified

```python
# Base interface all chunkers should implement
class BaseChunker(ABC):
    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """Split document into chunks."""
        pass
    
    def split_text(self, text: str) -> List[str]:
        """Split text into string chunks."""
        pass
```

### Common Data Models

```python
@dataclass
class Chunk:
    content: str
    chunk_id: str
    document_id: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None

@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    strategy: str = "recursive"
```

### Unique Features to Preserve

1. **Arabic/Islamic Text Support** (from advanced_chunker.py)
   - Quranic verse preservation
   - Hadith isnad+matn units
   - Arabic sentence boundary detection

2. **Code-Aware Chunking** (from splitting.py)
   - Language-specific separators
   - Syntax validation
   - Function/class boundary detection

3. **Hierarchical Parent-Child** (from splitting.py, chunking.py)
   - Small chunks for retrieval
   - Large parent chunks for context
   - Parent-child mapping

4. **Token-Aware Splitting** (from chunking.py)
   - tiktoken integration
   - Accurate token counting
   - Model context window alignment

5. **Semantic Chunking** (from advanced_chunker.py, splitting.py)
   - Embedding-based boundaries
   - Sentence grouping
   - Similarity threshold detection

---

## Target Architecture

```
src/rag/chunking/
├── __init__.py           # Public API exports
├── base.py               # BaseChunker ABC, Chunk, ChunkingConfig
├── fixed_size.py         # Fixed-size chunking
├── recursive.py          # Recursive character chunking
├── semantic.py           # Semantic/embedding-based chunking
├── hierarchical.py       # Parent-child hierarchical chunking
├── code.py               # Code-aware chunking
├── token_aware.py        # Token-based chunking with tiktoken
├── factory.py            # ChunkerFactory for creation
└── README.md             # Usage documentation

tests/rag/chunking/
├── __init__.py
├── test_base.py          # Base class tests
├── test_fixed_size.py    # Fixed-size tests
├── test_recursive.py     # Recursive tests
├── test_semantic.py      # Semantic tests
├── test_hierarchical.py  # Hierarchical tests
├── test_code.py          # Code chunking tests
├── test_token_aware.py   # Token-aware tests
├── test_factory.py       # Factory tests
└── test_integration.py   # Integration tests
```

---

## Consolidation Strategy

### Phase 1: Create Base Module (base.py)
- Abstract `BaseChunker` class
- `Chunk` dataclass with all fields
- `ChunkingConfig` dataclass
- Common utilities (token counting, text cleaning)

### Phase 2: Implement Strategies
1. **fixed_size.py** - Simple token/character-based
2. **recursive.py** - Hierarchical separator-based
3. **semantic.py** - Embedding similarity-based
4. **hierarchical.py** - Parent-child relationships
5. **code.py** - Language-aware code splitting
6. **token_aware.py** - tiktoken-based splitting

### Phase 3: Factory & Integration
- `ChunkerFactory` for easy creation
- Strategy registry
- Backward compatibility layer

### Phase 4: Testing
- Unit tests for each strategy
- Integration tests
- Edge case coverage
- 95%+ coverage target

### Phase 5: Migration
- Update all imports
- Deprecation warnings for old paths
- Migration guide

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Total Lines | ~1,760 | ~900 |
| Files | 7 scattered | 8 organized |
| Duplication | High | Eliminated |
| Test Coverage | ~40% | 95%+ |
| Import Paths | Inconsistent | Unified |
| Documentation | Minimal | Comprehensive |

---

## Implementation Timeline

- **Day 1**: Base module + Fixed/Recursive strategies
- **Day 2**: Semantic + Hierarchical strategies
- **Day 3**: Code + Token-aware strategies + Factory
- **Day 4**: Comprehensive tests
- **Day 5**: Migration + Documentation

---

## Success Criteria

- [ ] All 7 implementations consolidated
- [ ] 95%+ test coverage
- [ ] All existing imports updated
- [ ] Backward compatibility maintained
- [ ] Documentation complete
- [ ] Performance benchmarks met
