# OPTIMAL SRC/ STRUCTURE DESIGN

**Date:** March 29, 2026  
**Status:** Ready for Implementation  
**Version:** 1.0

---

## 1. PROPOSED ARCHITECTURE

### 1.1 Design Principles

1. **Single Source of Truth**: Each component exists in exactly one location
2. **Clear Module Boundaries**: Each module has a well-defined responsibility
3. **Unified Import System**: Consistent, predictable import patterns
4. **Progressive Disclosure**: Simple imports for common use, detailed imports for advanced use
5. **Backward Compatibility**: Maintain existing working imports during migration

### 1.2 High-Level Structure

```
src/
│
├── __init__.py              # Package root with unified imports
├── README.md                # Module documentation
│
├── foundations/             # Mathematical & ML foundations
├── ml/                      # Machine learning algorithms
├── llm/                     # LLM architecture & inference
├── rag/                     # Retrieval-Augmented Generation (CONSOLIDATED)
├── agents/                  # AI Agents (CONSOLIDATED)
├── embeddings/              # Embedding models
├── vector_stores/           # Vector database adapters (NEW)
├── evaluation/              # Evaluation frameworks
├── production/              # Production components
├── safety/                  # AI safety & guardrails
├── orchestration/           # Workflow orchestration
├── utils/                   # Shared utilities
└── courses/                 # Course materials (reorganized)
```

---

## 2. DETAILED MODULE SPECIFICATIONS

### 2.1 foundations/ (renamed from core/)

**Purpose:** Mathematical foundations and ML basics implemented from scratch

```
foundations/
├── __init__.py
├── README.md
├── math/
│   ├── __init__.py
│   ├── linear_algebra.py      # Vector, Matrix, operations
│   ├── calculus.py            # Derivatives, gradients, optimization
│   ├── probability.py         # Distributions, Bayes, hypothesis testing
│   └── statistics.py          # Statistical methods
└── ml_basics/
    ├── __init__.py
    ├── classical.py           # Basic ML algorithms
    └── neural_networks.py     # Basic NN components
```

**Imports:**
```python
from ai_mastery.foundations.math import Vector, Matrix, Optimizer
from ai_mastery.foundations.ml_basics import LinearRegression, MLP
```

---

### 2.2 ml/

**Purpose:** Classical and deep learning algorithms

```
ml/
├── __init__.py
├── README.md
├── classical/
│   ├── __init__.py
│   ├── linear_models.py       # Linear/Logistic Regression
│   ├── tree_models.py         # Decision Trees, Random Forest
│   ├── svm.py
│   └── clustering.py          # K-Means, DBSCAN
└── deep_learning/
    ├── __init__.py
    ├── layers.py              # Dense, Conv, Pool, Norm layers
    ├── networks.py            # CNN, RNN, LSTM, GRU
    ├── transformers_dl.py     # Deep learning transformers
    └── training.py            # Training loops, callbacks
```

**Imports:**
```python
from ai_mastery.ml.classical import RandomForest, KMeans
from ai_mastery.ml.deep_learning import CNN, LSTM, Transformer
```

---

### 2.3 llm/

**Purpose:** LLM architecture, fine-tuning, and inference

```
llm/
├── __init__.py
├── README.md
├── architecture/
│   ├── __init__.py
│   ├── transformer.py         # Complete transformer
│   ├── attention.py           # Multi-head, flash attention
│   ├── positional.py          # Positional encodings (RoPE, etc.)
│   └── sampling.py            # Generation strategies
├── fine_tuning/
│   ├── __init__.py
│   ├── full_finetune.py
│   ├── lora.py
│   ├── qlora.py
│   └── dpo.py
└── inference/
    ├── __init__.py
    ├── kv_cache.py
    ├── batching.py
    ├── speculative.py
    └── quantization.py
```

**Imports:**
```python
from ai_mastery.llm.architecture import Transformer, MultiHeadAttention
from ai_mastery.llm.fine_tuning import LoRAConfig, DPOTrainer
from ai_mastery.llm.inference import PagedKVCache, SpeculativeDecoder
```

---

### 2.4 rag/ (CONSOLIDATED)

**Purpose:** Unified RAG pipeline with all strategies

```
rag/
├── __init__.py
├── README.md
├── core.py                    # Main RAGPipeline class
├── types.py                   # Document, Chunk, Query types
├── chunking/
│   ├── __init__.py
│   ├── base.py
│   ├── fixed_size.py
│   ├── recursive.py
│   ├── semantic.py
│   ├── hierarchical.py
│   ├── token_aware.py
│   ├── code.py
│   └── factory.py
├── retrieval/
│   ├── __init__.py
│   ├── base.py
│   ├── similarity.py
│   ├── hybrid.py
│   ├── multi_query.py
│   ├── hyde.py
│   └── ensemble.py
├── reranking/
│   ├── __init__.py
│   ├── base.py
│   ├── cross_encoder.py
│   ├── llm_reranker.py
│   └── diversity.py
├── advanced/
│   ├── __init__.py
│   ├── query_construction.py  # SQL, Cypher generation
│   ├── tools.py               # Tool integration
│   ├── post_processing.py     # RAG-Fusion, synthesis
│   └── program_llm.py         # DSPy integration
└── specialized/
    ├── __init__.py
    ├── multimodal.py
    ├── temporal.py
    ├── graph_enhanced.py
    ├── privacy_preserving.py
    └── continual_learning.py
```

**Imports:**
```python
from ai_mastery.rag import RAGPipeline, Document, DocumentChunk
from ai_mastery.rag.chunking import SemanticChunker, RecursiveChunker
from ai_mastery.rag.retrieval import HybridRetrieval, MultiQueryRetriever
from ai_mastery.rag.reranking import CrossEncoderReranker
from ai_mastery.rag.specialized import GraphEnhancedRAG
```

---

### 2.5 agents/ (CONSOLIDATED)

**Purpose:** AI agent systems and tool orchestration

```
agents/
├── __init__.py
├── README.md
├── core.py                    # BaseAgent, ReActAgent
├── types.py                   # Agent state, message types
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── calculator.py
│   ├── search.py
│   ├── code_interpreter.py
│   └── api_tool.py
├── frameworks/
│   ├── __init__.py
│   ├── langgraph_wrapper.py
│   ├── crewai_wrapper.py
│   └── autogen_wrapper.py
├── protocols/
│   ├── __init__.py
│   ├── mcp.py                 # Model Context Protocol
│   └── a2a.py                 # Agent-to-Agent protocol
└── multi_agent/
    ├── __init__.py
    ├── orchestrator.py
    ├── collaboration.py
    └── debate.py
```

**Imports:**
```python
from ai_mastery.agents import ReActAgent, ToolRegistry
from ai_mastery.agents.tools import CalculatorTool, SearchTool
from ai_mastery.agents.frameworks import LangGraphAgent
from ai_mastery.agents.multi_agent import MultiAgentSystem
```

---

### 2.6 embeddings/

**Purpose:** Unified embedding model interfaces

```
embeddings/
├── __init__.py
├── README.md
├── base.py                    # EmbeddingModel ABC
├── sentence_transformers.py   # ST integration
├── openai_embeddings.py       # OpenAI API
├── local.py                   # Local models
├── caching.py                 # Embedding cache
└── multimodal.py              # Text + image embeddings
```

**Imports:**
```python
from ai_mastery.embeddings import TextEmbedder, ImageEmbedder
from ai_mastery.embeddings.sentence_transformers import STEmbedder
from ai_mastery.embeddings.openai_embeddings import OpenAIEmbedder
```

---

### 2.7 vector_stores/ (NEW)

**Purpose:** Vector database adapters

```
vector_stores/
├── __init__.py
├── README.md
├── base.py                    # VectorStore ABC
├── memory.py                  # In-memory store
├── faiss_store.py             # FAISS integration
├── qdrant_store.py            # Qdrant integration
├── chroma_store.py            # ChromaDB integration
├── weaviate_store.py          # Weaviate integration
├── pgvector_store.py          # PostgreSQL + pgvector
└── hybrid.py                  # Hybrid search wrapper
```

**Imports:**
```python
from ai_mastery.vector_stores import VectorStore, FAISSStore, QdrantStore
from ai_mastery.vector_stores.hybrid import HybridVectorStore
```

---

### 2.8 evaluation/

**Purpose:** Evaluation frameworks for LLM and RAG

```
evaluation/
├── __init__.py
├── README.md
├── ragas_integration.py
├── llm_judge.py
├── metrics.py
├── benchmarks/
│   ├── __init__.py
│   ├── mmlu.py
│   ├── truthful_qa.py
│   ├── gsm8k.py
│   └── humaneval.py
└── human_eval/
    ├── __init__.py
    └── interface.py
```

**Imports:**
```python
from ai_mastery.evaluation import RAGEvaluator, LLMJudge
from ai_mastery.evaluation.benchmarks import MMLUBenchmark
```

---

### 2.9 production/

**Purpose:** Production-ready components

```
production/
├── __init__.py
├── README.md
├── api/
│   ├── __init__.py
│   ├── fastapi_app.py
│   ├── routes.py
│   └── middleware.py
├── caching/
│   ├── __init__.py
│   ├── semantic_cache.py
│   └── cost_optimizer.py
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py
│   ├── observability.py
│   └── alerts.py
├── deployment/
│   ├── __init__.py
│   ├── docker.py
│   ├── kubernetes.py
│   └── edge.py
└── security/
    ├── __init__.py
    ├── auth.py
    ├── rate_limiting.py
    └── trust_layer.py
```

**Imports:**
```python
from ai_mastery.production.api import FastAPIApp
from ai_mastery.production.caching import SemanticCache
from ai_mastery.production.monitoring import RAGObservability
```

---

### 2.10 safety/

**Purpose:** AI safety and content moderation

```
safety/
├── __init__.py
├── README.md
├── guardrails.py
├── content_moderation.py
├── safety_classifier.py
├── prompt_security.py
└── red_teaming.py
```

**Imports:**
```python
from ai_mastery.safety import ContentModerator, SafetyGuardrails
from ai_mastery.safety.prompt_security import PromptSecurityAnalyzer
```

---

### 2.11 orchestration/

**Purpose:** Workflow orchestration

```
orchestration/
├── __init__.py
├── README.md
├── workflows.py
├── pipelines.py
└── scheduling.py
```

**Imports:**
```python
from ai_mastery.orchestration import Workflow, Pipeline
```

---

### 2.12 utils/

**Purpose:** Shared utilities

```
utils/
├── __init__.py
├── README.md
├── logging.py                 # Unified logging system
├── config.py                  # Configuration management
├── errors.py                  # Custom exceptions
├── types.py                   # Shared type definitions
└── decorators.py              # Common decorators
```

**Imports:**
```python
from ai_mastery.utils import get_logger, log_performance
from ai_mastery.utils.errors import AIError, ConfigurationError
```

---

### 2.13 courses/

**Purpose:** Course materials (reorganized from current structure)

```
courses/
├── __init__.py
├── README.md
├── fundamentals/              # Was part1_fundamentals/
│   ├── mathematics/
│   ├── python_ml/
│   ├── neural_networks/
│   └── nlp/
├── scientist/                 # Was llm_scientist/
│   ├── llm_architecture/
│   ├── pretraining/
│   ├── fine_tuning/
│   ├── preference_alignment/
│   ├── evaluation/
│   ├── quantization/
│   └── new_trends/
└── engineering/               # Was llm_engineering/
    ├── running_llms/
    ├── vector_storage/
    ├── rag/
    ├── advanced_rag/
    ├── agents/
    ├── inference_optimization/
    ├── deployment/
    └── security/
```

**Imports:**
```python
from ai_mastery.courses.fundamentals.mathematics import VectorOperations
from ai_mastery.courses.scientist.fine_tuning import LoRATrainer
from ai_mastery.courses.engineering.rag import RAGOrchestrator
```

---

## 3. UNIFIED IMPORT SYSTEM

### 3.1 Package Root (__init__.py)

```python
"""
AI-Mastery-2026: Unified AI Engineering Toolkit
================================================

A comprehensive AI engineering platform built from first principles.

Quick Start:
    >>> from ai_mastery import rag, embeddings, vector_stores
    >>> from ai_mastery.rag import RAGPipeline, SemanticChunker
    >>> from ai_mastery.agents import ReActAgent, ToolRegistry

Modules:
    - foundations: Mathematics and ML basics
    - ml: Classical and deep learning
    - llm: LLM architecture and inference
    - rag: Retrieval-Augmented Generation
    - agents: AI agent systems
    - embeddings: Embedding models
    - vector_stores: Vector database adapters
    - evaluation: Evaluation frameworks
    - production: Production components
    - safety: AI safety
    - orchestration: Workflow orchestration
    - utils: Shared utilities
    - courses: Course materials
"""

__version__ = "2.0.0"
__author__ = "AI-Mastery-2026 Team"
__license__ = "MIT"

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    
    # Core modules
    "foundations",
    "ml",
    "llm",
    "rag",
    "agents",
    "embeddings",
    "vector_stores",
    "evaluation",
    "production",
    "safety",
    "orchestration",
    "utils",
    
    # Course materials
    "courses",
]

# Import submodules
from src import foundations
from src import ml
from src import llm
from src import rag
from src import agents
from src import embeddings
from src import vector_stores
from src import evaluation
from src import production
from src import safety
from src import orchestration
from src import utils
from src import courses


# Convenience imports for common use cases
from src.rag.core import RAGPipeline, Document, DocumentChunk
from src.rag.chunking import SemanticChunker, RecursiveChunker
from src.rag.retrieval import HybridRetrieval
from src.agents.core import ReActAgent
from src.agents.tools import ToolRegistry
from src.embeddings import TextEmbedder
from src.vector_stores import FAISSStore, QdrantStore
from src.utils.logging import get_logger, log_performance
```

### 3.2 Import Patterns

**Pattern 1: High-level imports (recommended for most users)**
```python
from ai_mastery import rag, agents, embeddings
from ai_mastery.rag import RAGPipeline
from ai_mastery.agents import ReActAgent
```

**Pattern 2: Specific component imports**
```python
from ai_mastery.rag.chunking import SemanticChunker
from ai_mastery.rag.retrieval import HybridRetriever
from ai_mastery.agents.tools import CalculatorTool
```

**Pattern 3: Course material imports**
```python
from ai_mastery.courses.fundamentals.mathematics import VectorOperations
from ai_mastery.courses.scientist.fine_tuning import LoRATrainer
```

---

## 4. MIGRATION PLAN

### Phase 1: Preparation (Days 1-2)

1. **Backup current structure**
   ```bash
   cp -r src src_backup_$(date +%Y%m%d)
   ```

2. **Create new directory structure**
   ```bash
   mkdir -p src/{foundations,ml,llm,rag,agents,embeddings,vector_stores,evaluation,production,safety,orchestration,utils,courses}
   ```

3. **Remove duplicate root directories**
   ```bash
   rm -rf 01_foundamentals 02_scientist 03_engineer 04_production
   ```

4. **Remove backup files**
   ```bash
   rm src/production/vector_db_backup.py
   ```

### Phase 2: Core Module Migration (Days 3-7)

1. **Rename core/ → foundations/**
   - Move files
   - Update internal imports
   - Update src/__init__.py

2. **Create vector_stores/**
   - Extract from production/vector_db.py
   - Create base class and adapters

3. **Consolidate rag/**
   - Move src/llm/rag.py → src/rag/advanced/
   - Move src/llm/advanced_rag.py → src/rag/advanced/
   - Move src/rag_specialized/* → src/rag/specialized/
   - Move src/reranking/* → src/rag/reranking/
   - Move src/retrieval/* → src/rag/retrieval/

4. **Consolidate agents/**
   - Move src/llm/agents.py → src/agents/core.py
   - Move src/llm_engineering/module_3_5_agents/* → src/agents/

### Phase 3: Import Updates (Days 8-10)

1. **Update all __init__.py files**
2. **Update internal imports across all modules**
3. **Update src/__init__.py with clean imports**
4. **Create migration compatibility layer** (optional)

### Phase 4: Testing & Verification (Days 11-14)

1. **Run existing tests**
   ```bash
   pytest src/part1_fundamentals/ -v
   ```

2. **Create new integration tests**
3. **Verify all imports work**
4. **Update documentation**

### Phase 5: Course Reorganization (Days 15-21)

1. **Move part1_fundamentals/ → courses/fundamentals/**
2. **Move llm_scientist/ → courses/scientist/**
3. **Move llm_engineering/ → courses/engineering/**
4. **Update course material imports**

---

## 5. NAMING CONVENTIONS

### 5.1 Directory Naming

- **Lowercase with underscores**: `vector_stores/`, `fine_tuning/`
- **Descriptive names**: `chunking/` not `chunks/`
- **Consistent prefixes**: No mixed `01_`, `module_1_` prefixes in production code

### 5.2 File Naming

- **Lowercase with underscores**: `semantic_chunker.py`
- **Descriptive class names**: `SemanticChunker` not `Chunker`
- **No backup files**: Use git for version control

### 5.3 Class/Function Naming

- **Classes**: PascalCase (`RAGPipeline`, `SemanticChunker`)
- **Functions**: snake_case (`chunk_documents`, `embed_query`)
- **Constants**: UPPER_CASE (`DEFAULT_CHUNK_SIZE`, `MAX_RETRIES`)

### 5.4 Type Hints

- **Always annotate function signatures**
- **Use Optional for nullable returns**
- **Use Union for multiple types**
- **Use TypedDict for complex dictionaries**
- **Use Protocol for duck typing**

---

## 6. QUALITY STANDARDS

### 6.1 Type Hints

**Required for all new code:**
```python
from typing import List, Optional, Dict, Any, Callable, Protocol

class ChunkingStrategy(Protocol):
    def chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]: ...

def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 512,
    overlap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Chunk documents with specified strategy."""
```

### 6.2 Docstrings

**Required for all public APIs:**
```python
def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 512,
) -> List[Dict[str, Any]]:
    """
    Split documents into chunks.

    Args:
        documents: List of documents with 'id' and 'content' fields
        chunk_size: Maximum tokens per chunk

    Returns:
        List of document chunks

    Example:
        >>> docs = [{"id": "1", "content": "Long text..."}]
        >>> chunks = chunk_documents(docs, chunk_size=512)
        >>> len(chunks)
        3
    """
```

### 6.3 Error Handling

**Required patterns:**
```python
from src.utils.errors import ConfigurationError, ProcessingError

def load_model(model_name: str) -> Any:
    """Load embedding model."""
    if not model_name:
        raise ConfigurationError("Model name cannot be empty")
    
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError as e:
        raise ProcessingError(
            f"sentence-transformers not installed: {e}"
        ) from e
    except Exception as e:
        raise ProcessingError(f"Failed to load model: {e}") from e
```

### 6.4 Logging

**Required for production code:**
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)

def process_query(query: str) -> str:
    """Process user query."""
    logger.info("Processing query", extra={"query_length": len(query)})
    
    try:
        result = _process(query)
        logger.info("Query processed successfully")
        return result
    except Exception as e:
        logger.error("Query processing failed", exc_info=True)
        raise
```

### 6.5 Testing

**Required coverage:**
- Core modules: >90%
- Production modules: >85%
- Course modules: >70%

```python
import pytest
from src.rag.chunking import SemanticChunker

class TestSemanticChunker:
    def test_chunk_single_document(self):
        chunker = SemanticChunker()
        doc = {"id": "1", "content": "Test content."}
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        assert chunks[0]["document_id"] == "1"
```

---

## 7. SUCCESS CRITERIA

### 7.1 Structural

- [ ] No duplicate directories
- [ ] All modules have clear boundaries
- [ ] vector_stores/ module exists
- [ ] RAG consolidated in rag/
- [ ] Agents consolidated in agents/

### 7.2 Import System

- [ ] All imports resolve correctly
- [ ] No try/except for core imports
- [ ] Convenience imports work
- [ ] Course imports work

### 7.3 Code Quality

- [ ] Type hints on all public APIs
- [ ] Docstrings on all public APIs
- [ ] Error handling in production code
- [ ] Logging in production code

### 7.4 Testing

- [ ] All existing tests pass
- [ ] New integration tests added
- [ ] Coverage thresholds met
- [ ] CI/CD pipeline configured

---

## 8. APPENDIX

### A. Module Dependency Graph

```
foundations/ ─┬─> ml/ ──> llm/ ──┬─> rag/ ──> agents/
              │                  │
              │                  └─> embeddings/ ──> vector_stores/
              │
              └─> utils/ <─────────────────────────────── (all modules)
```

### B. Import Compatibility Matrix

| Old Import | New Import | Status |
|------------|------------|--------|
| `from src.core import Vector` | `from src.foundations.math import Vector` | Changed |
| `from src.rag import RAGPipeline` | `from src.rag import RAGPipeline` | Same |
| `from src.llm import Transformer` | `from src.llm.architecture import Transformer` | Changed |
| `from src.agents import Agent` | `from src.agents.core import Agent` | Changed |

### C. File Movement Summary

| From | To | Action |
|------|-----|--------|
| `src/core/` | `src/foundations/math/` | Rename + reorganize |
| `src/llm/rag.py` | `src/rag/advanced/` | Move |
| `src/llm/agents.py` | `src/agents/core.py` | Move |
| `src/rag_specialized/` | `src/rag/specialized/` | Move |
| `src/reranking/` | `src/rag/reranking/` | Move |
| `src/retrieval/` | `src/rag/retrieval/` | Move |
| `src/production/vector_db.py` | `src/vector_stores/` | Extract |
| `src/part1_fundamentals/` | `src/courses/fundamentals/` | Move |
| `src/llm_scientist/` | `src/courses/scientist/` | Move |
| `src/llm_engineering/` | `src/courses/engineering/` | Move |

---

**Document Version:** 1.0  
**Last Updated:** March 29, 2026  
**Next Review:** After Phase 1 completion
