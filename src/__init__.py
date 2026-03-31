"""
AI-Mastery-2026: Unified AI Engineering Toolkit
===============================================

A comprehensive AI engineering platform built from first principles,
featuring production-ready components for RAG systems, LLM engineering,
and machine learning operations.

Import Structure:
-----------------
    from src import rag, core, llm, ml, agents, production
    from src.rag import RAGPipeline
    from src.rag.chunking import SemanticChunker
    from src.agents import ReActAgent
    from src.rag.vector_stores import FAISSStore

Quick Start:
------------
    >>> from src import rag
    >>> from src.rag.chunking import SemanticChunker
    >>> from src.rag.vector_stores import FAISSStore, VectorStoreConfig
    >>> from src.rag.embeddings import TextEmbedder
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
- **core**: Mathematics from scratch (linear algebra, optimization, probability, utils)
- **ml**: Classical and deep learning implementations
- **llm**: Transformer architectures, attention, safety, and evaluation
- **rag**: Unified RAG pipeline with chunking, retrieval, reranking, and specialized architectures
- **agents**: Multi-agent systems, tool orchestration, and workflow orchestration
- **production**: Production components (API, caching, monitoring, auth, MLOps)
- **api**: REST and GraphQL API interfaces
- **app**: Main application entry points

Version: 2.2.0
Author: AI-Mastery-2026 Team
License: MIT
"""

__version__ = "2.2.0"
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
    "agents",
    "production",
    "api",
    "app",
]

# Import submodules
from src import core
from src import ml
from src import llm
from src import rag
from src import agents
from src import production
from src import api
from src import app

# Convenience imports from rag
try:
    from src.rag.chunking import SemanticChunker, RecursiveChunker
    from src.rag.embeddings import TextEmbedder, ImageEmbedder
    from src.rag.vector_stores import FAISSStore, MemoryVectorStore, VectorStoreConfig
except ImportError:
    pass

# Convenience imports from core
try:
    from src.core.utils.logging import get_logger, log_performance
except ImportError:
    pass


# Module metadata
__module_info__ = {
    "core": {
        "description": "Mathematics from scratch and shared utilities",
        "submodules": [
            "linear_algebra",
            "optimization",
            "probability",
            "utils",
            "data",
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
        "description": "Transformer architectures, LLM engineering, safety and evaluation",
        "submodules": [
            "transformer",
            "attention",
            "fine_tuning",
            "safety",
            "evaluation",
            "benchmarks",
        ],
    },
    "rag": {
        "description": "Unified RAG pipeline and advanced architectures",
        "submodules": [
            "chunking",
            "retrieval",
            "reranking",
            "embeddings",
            "vector_stores",
            "specialized",
        ],
    },
    "agents": {
        "description": "Multi-agent systems and orchestration",
        "submodules": [
            "tools",
            "multi_agent_systems",
            "orchestration",
        ],
    },
    "production": {
        "description": "Production-ready components and MLOps",
        "submodules": [
            "caching",
            "monitoring",
            "observability",
            "auth",
            "llm_ops",
        ],
    },
    "api": {
        "description": "API interfaces (REST, GraphQL)",
        "submodules": [],
    },
    "app": {
        "description": "Application entry points",
        "submodules": [],
    },
}


def get_module_info(module_name: str) -> dict:
    """Get information about a specific module."""
    return __module_info__.get(module_name, {})


def list_modules() -> list[str]:
    """List all available modules."""
    return list(__module_info__.keys())


def print_module_tree() -> None:
    """Print a tree view of all modules and submodules."""
    print("AI-Mastery-2026 Modules")
    print("=" * 40)

    for module_name, info in __module_info__.items():
        print(f"\n{module_name}")
        print(f"  {info['description']}")
        if info['submodules']:
            print("  Submodules:")
            for submodule in info["submodules"]:
                print(f"    - {submodule}")
