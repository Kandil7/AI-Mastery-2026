"""
Arabic Islamic Literature RAG System
====================================

A production-ready RAG (Retrieval-Augmented Generation) system
for querying Arabic Islamic literature corpus.

Features:
- Complete RAG Pipeline with hybrid retrieval
- Domain Specialists (Tafsir, Hadith, Fiqh, Aqeedah, etc.)
- Multi-Madhab Comparative Analysis
- Multi-hop Reasoning
- Islamic Authority Ranking
- Specialized Agent System
- Comprehensive Evaluation

Usage:
    # Quick start
    from rag_system import IslamicRAG
    rag = IslamicRAG()
    await rag.initialize()
    result = await rag.query("ما هو التوحيد؟")

    # Domain-specific queries
    result = await rag.query_tafsir("ما تفسير آية الكرسي؟")
    result = await rag.query_fiqh("ما شروط الصلاة؟")

    # Comparative analysis
    result = await rag.compare_madhhabs("ما حكم الزكاة؟")

    # Advanced reasoning
    result = await rag.reason_with_chain("ما الدليل على...")

    # Agent interactions
    result = await rag.ask_fatwa("ما حكم...?")
    result = await rag.ask_as_student("اريد تعلم...")
"""

__version__ = "1.0.0"
__author__ = "AI-Mastery-2026"

# Core Pipeline
from .pipeline.complete_pipeline import (
    CompleteRAGPipeline,
    RAGConfig,
    create_rag_pipeline,
    QueryResult,
)

# Data Ingestion
from .data.multi_source_ingestion import (
    MultiSourceIngestionPipeline,
    DataSource,
    DataSourceType,
)

# Generation
from .generation.generator import (
    LLMClient,
    LLMProvider,
    RAGGenerator,
    ArabicPrompts,
)

# Integration (recommended)
from .integration import (
    IslamicRAG,
    IslamicRAGConfig,
    create_islamic_rag,
    quick_query,
)

# Specialized Components
from .specialists.islamic_scholars import (
    IslamicScholar,
    ComparativeFiqhScholar,
    ChainOfScholarship,
    IslamicDomain,
    create_islamic_scholar,
    create_comparative_fiqh_scholar,
)

from .specialists.advanced_features import (
    AuthorityRanker,
    CrossReferenceSystem,
    MultiHopReasoning,
    TimelineReconstructor,
    ProgressiveRetrieval,
    ConceptExtractor,
)

# Agents
from .agents.agent_system import (
    IslamicRAGAgent,
    AgentTeam,
    AgentRole,
    create_agent,
)

# Evaluation
from .evaluation.islamic_metrics import (
    IslamicRAGEvaluator,
    create_islamic_evaluator,
)

__all__ = [
    # Version
    "__version__",
    # Core Pipeline
    "CompleteRAGPipeline",
    "RAGConfig",
    "create_rag_pipeline",
    "QueryResult",
    # Data
    "DataIngestionPipeline",
    "MetadataIngestionPipeline",
    "DataSource",
    "DataSourceType",
    # Generation
    "LLMClient",
    "LLMProvider",
    "RAGGenerator",
    "ArabicPrompts",
    # Integration (Recommended)
    "IslamicRAG",
    "IslamicRAGConfig",
    "create_islamic_rag",
    "quick_query",
    # Specialists
    "IslamicScholar",
    "ComparativeFiqhScholar",
    "ChainOfScholarship",
    "IslamicDomain",
    "create_islamic_scholar",
    "create_comparative_fiqh_scholar",
    # Advanced Features
    "AuthorityRanker",
    "CrossReferenceSystem",
    "MultiHopReasoning",
    "TimelineReconstructor",
    "ProgressiveRetrieval",
    "ConceptExtractor",
    # Agents
    "IslamicRAGAgent",
    "AgentTeam",
    "AgentRole",
    "create_agent",
    # Evaluation
    "IslamicRAGEvaluator",
    "create_islamic_evaluator",
]
