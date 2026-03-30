"""
AI-Mastery-2026: Unified AI Engineering Toolkit
===============================================

A comprehensive AI engineering platform built from first principles,
featuring production-ready components for RAG systems, LLM engineering,
and machine learning operations.

Import Structure:
-----------------
    from src import rag, embeddings, vector_stores
    from src.rag import RAGPipeline, SemanticChunker
    from src.agents import ReActAgent, ToolRegistry
    from src.vector_stores import FAISSStore, QdrantStore

Quick Start:
------------
    >>> from src import rag, embeddings, vector_stores
    >>> from src.rag.chunking import SemanticChunker
    >>> from src.vector_stores import FAISSStore, VectorStoreConfig
    >>>
    >>> # Initialize components
    >>> embed_model = TextEmbedder("all-MiniLM-L6-v2")
    >>> vector_store = FAISSStore(VectorStoreConfig(dim=384))
    >>> chunker = SemanticChunker(chunk_size=512)
    >>>
    >>> # Create RAG pipeline
    >>> pipeline = rag.RAGPipeline(embed_model, vector_store, chunker)
    >>>
    >>> # Add documents
    >>> docs = [{"id": "1", "content": "AI is transforming industries."}]
    >>> pipeline.add_documents(docs)
    >>>
    >>> # Query
    >>> results = pipeline.query("How is AI impacting business?")
    >>> print(results[0].content)

Modules:
--------
- **core**: Mathematics from scratch (linear algebra, optimization, probability)
- **ml**: Classical and deep learning implementations
- **llm**: Transformer architectures and attention mechanisms
- **rag**: Unified RAG pipeline with chunking, retrieval, reranking
- **rag_specialized**: Advanced RAG architectures (multimodal, temporal, graph)
- **agents**: Multi-agent systems and tool orchestration
- **embeddings**: Unified embedding model interfaces
- **vector_stores**: Vector database adapters (FAISS, Qdrant, Chroma)
- **evaluation**: LLM and RAG evaluation frameworks
- **production**: Production components (API, caching, monitoring, auth)
- **orchestration**: Workflow orchestration and pipelines
- **safety**: AI safety, guardrails, content moderation
- **utils**: Shared utilities (logging, errors, config, types)

Version: 2.1.0
Author: AI-Mastery-2026 Team
License: MIT
"""

__version__ = "2.1.0"
__author__ = "AI-Mastery-2026 Team"
__email__ = "medokandeal7@gmail.com"
__license__ = "MIT"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Core modules
    "core",
    "ml",
    "llm",
    "rag",
    "rag_specialized",
    "agents",
    "embeddings",
    "vector_stores",
    "evaluation",
    "production",
    "orchestration",
    "safety",
    "utils",
]

# Import all submodules (lazy to avoid circular imports)
from src import core
from src import ml
from src import llm
from src import rag
from src import rag_specialized
from src import agents
from src import embeddings
from src import vector_stores
from src import evaluation
from src import production
from src import orchestration
from src import safety
from src import utils

# Convenience imports from rag
try:
    from src.rag.chunking import SemanticChunker
    from src.rag.chunking import RecursiveChunker
except ImportError:
    pass

# Convenience imports from embeddings
try:
    from src.embeddings import TextEmbedder, ImageEmbedder
except ImportError:
    pass

# Convenience imports from vector_stores
try:
    from src.vector_stores import FAISSStore, MemoryVectorStore, VectorStoreConfig
except ImportError:
    pass

# Convenience imports from utils
try:
    from src.utils.logging import get_logger, log_performance
except ImportError:
    pass


# Module metadata
__module_info__ = {
    "core": {
        "description": "Mathematics from scratch - linear algebra, calculus, optimization, probability",
        "submodules": [
            "linear_algebra",
            "calculus",
            "optimization",
            "probability",
            "statistics",
        ],
    },
    "ml": {
        "description": "Classical and deep learning implementations",
        "submodules": [
            "classical",
            "deep_learning",
            "vision",
            "gnn_recommender",
        ],
    },
    "llm": {
        "description": "Transformer architectures and LLM implementations",
        "submodules": [
            "transformer",
            "attention",
            "fine_tuning",
            "rag",
            "advanced_rag",
            "agents",
        ],
    },
    "rag": {
        "description": "Unified RAG pipeline with chunking, retrieval, and reranking",
        "submodules": [
            "core",
            "chunking",
            "retrieval",
            "reranking",
            "advanced",
            "specialized",
        ],
    },
    "rag_specialized": {
        "description": "Advanced RAG architectures",
        "submodules": [
            "adaptive_multimodal",
            "temporal_aware",
            "graph_enhanced",
            "privacy_preserving",
            "continual_learning",
        ],
    },
    "agents": {
        "description": "Multi-agent systems and tool orchestration",
        "submodules": [
            "core",
            "tools",
            "multi_agent_systems",
            "integrations",
        ],
    },
    "embeddings": {
        "description": "Unified embedding model interfaces and implementations",
        "submodules": [
            "embeddings",
        ],
    },
    "vector_stores": {
        "description": "Vector database adapters",
        "submodules": [
            "base",
            "memory",
            "faiss_store",
        ],
    },
    "evaluation": {
        "description": "LLM and RAG evaluation frameworks",
        "submodules": [
            "evaluation",
        ],
    },
    "production": {
        "description": "Production-ready components",
        "submodules": [
            "api",
            "caching",
            "monitoring",
            "observability",
            "auth",
            "data_pipeline",
            "query_enhancement",
        ],
    },
    "orchestration": {
        "description": "Workflow orchestration and pipelines",
        "submodules": [
            "orchestration",
        ],
    },
    "safety": {
        "description": "AI safety and content moderation",
        "submodules": [
            "guardrails",
            "content_moderation",
        ],
    },
    "utils": {
        "description": "Shared utilities",
        "submodules": [
            "logging",
            "errors",
            "config",
            "types",
        ],
    },
}


def get_module_info(module_name: str) -> dict:
    """
    Get information about a specific module.

    Args:
        module_name: Name of the module

    Returns:
        Dictionary with module description and submodules

    Example:
        >>> info = get_module_info("rag")
        >>> print(info["description"])
        Unified RAG pipeline with chunking, retrieval, and reranking
    """
    return __module_info__.get(module_name, {})


def list_modules() -> list[str]:
    """
    List all available modules.

    Returns:
        List of module names

    Example:
        >>> modules = list_modules()
        >>> print(modules)
        ['core', 'ml', 'llm', 'rag', ...]
    """
    return list(__module_info__.keys())


def print_module_tree() -> None:
    """
    Print a tree view of all modules and submodules.

    Example:
        >>> print_module_tree()
        AI-Mastery-2026 Modules
        =======================
        core
          ├── linear_algebra
          ├── calculus
          └── ...
        ml
          ├── classical
          └── ...
        ...
    """
    print("AI-Mastery-2026 Modules")
    print("=" * 40)

    for module_name, info in __module_info__.items():
        print(f"\n{module_name}")
        print(f"  {info['description']}")
        print("  Submodules:")
        for submodule in info["submodules"][:5]:  # Show first 5
            print(f"    - {submodule}")
        if len(info["submodules"]) > 5:
            print(f"    ... and {len(info['submodules']) - 5} more")
