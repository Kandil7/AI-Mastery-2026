"""
Arabic Islamic Literature RAG System

Production-grade Retrieval-Augmented Generation system for Arabic Islamic literature.

Usage:
    from rag_system.src.integration import create_islamic_rag
    
    rag = create_islamic_rag()
    await rag.initialize()
    
    result = await rag.query("ما هو التوحيد في الإسلام؟")
"""

__version__ = "1.0.0"
__author__ = "Islamic RAG Team"

# Note: All imports are from src/ subdirectory
# The main entry point is src/integration.py

# ============================================================================
# MAIN INTEGRATION (Recommended Entry Point)
# ============================================================================
from .src.integration import (
    IslamicRAG,
    IslamicRAGConfig,
    create_islamic_rag,
    quick_query,
)

# ============================================================================
# CORE PIPELINE
# ============================================================================
from .src.pipeline.complete_pipeline import (
    CompleteRAGPipeline,
    RAGConfig as PipelineRAGConfig,  # Renamed to avoid conflict
    create_rag_pipeline,
    QueryResult,
)

# ============================================================================
# DATA INGESTION
# ============================================================================
from .src.data.multi_source_ingestion import (
    MultiSourceIngestionPipeline,
    DataSource,
    DataSourceType,
    ConnectorType,
    Document,
    create_file_source,
    create_api_source,
    create_database_source,
)

# ============================================================================
# PROCESSING
# ============================================================================
from .src.processing.advanced_chunker import (
    AdvancedChunker,
    create_chunker,
    ChunkingStrategy,
    get_recommended_chunking,
)

from .src.processing.embedding_pipeline import (
    EmbeddingPipeline,
    EmbeddingConfig,
    EmbeddingProvider,
    create_embedding_pipeline,
    get_recommended_model,
)

# ============================================================================
# RETRIEVAL
# ============================================================================
from .src.retrieval.vector_store import (
    VectorStore,
    VectorStoreConfig,
    VectorStoreType,
    create_vector_store,
    SearchResult,
)

from .src.retrieval.hybrid_retriever import (
    HybridRetriever,
    BM25Index,
    Reranker,
    RetrievalResult,
)

from .src.retrieval.query_transformer import (
    QueryTransformer,
    create_query_transformer,
    TransformedQuery,
    QueryType,
)

# ============================================================================
# GENERATION
# ============================================================================
from .src.generation.generator import (
    LLMClient,
    LLMProvider,
    RAGGenerator,
    GenerationResult,
    ArabicPrompts,
)

# ============================================================================
# SPECIALISTS
# ============================================================================
from .src.specialists.islamic_scholars import (
    IslamicScholar,
    ComparativeFiqhScholar,
    IslamicDomain,
    create_islamic_scholar,
    create_comparative_fiqh_scholar,
)

from .src.specialists.advanced_features import (
    AuthorityRanker,
    CrossReferenceSystem,
    MultiHopReasoning,
    TimelineReconstructor,
    create_authority_ranker,
    create_cross_reference_system,
    create_multi_hop_reasoning,
)

# ============================================================================
# AGENTS
# ============================================================================
from .src.agents.agent_system import (
    IslamicRAGAgent,
    AgentTeam,
    AgentRole,
    create_agent,
)

from .src.agents.enhanced_agents import (
    EnhancedIslamicRAGAgent,
    EnhancedAgentTeam,
    EnhancedAgentRole,
    create_enhanced_agent,
    create_enhanced_agent_team,
)

# ============================================================================
# EVALUATION
# ============================================================================
from .src.evaluation.evaluator import (
    RAGEvaluator,
    ArabicTestDataset,
    EvaluationSample,
)

from .src.evaluation.islamic_metrics import (
    IslamicEvaluationMetrics,
    create_islamic_evaluator,
)

# ============================================================================
# MONITORING
# ============================================================================
from .src.monitoring.monitoring import (
    RAGMonitor,
    CostTracker,
    QueryLogger,
    get_monitor,
)

# ============================================================================
# PUBLIC API - All Exports
# ============================================================================
__all__ = [
    # Main (Recommended)
    "IslamicRAG",
    "IslamicRAGConfig",
    "create_islamic_rag",
    "quick_query",
    # Pipeline
    "CompleteRAGPipeline",
    "PipelineRAGConfig",
    "create_rag_pipeline",
    "QueryResult",
    # Data
    "MultiSourceIngestionPipeline",
    "DataSource",
    "DataSourceType",
    "ConnectorType",
    "Document",
    "create_file_source",
    "create_api_source",
    "create_database_source",
    # Processing
    "AdvancedChunker",
    "create_chunker",
    "ChunkingStrategy",
    "get_recommended_chunking",
    "EmbeddingPipeline",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "create_embedding_pipeline",
    "get_recommended_model",
    # Retrieval
    "VectorStore",
    "VectorStoreConfig",
    "VectorStoreType",
    "create_vector_store",
    "SearchResult",
    "HybridRetriever",
    "BM25Index",
    "Reranker",
    "RetrievalResult",
    "QueryTransformer",
    "create_query_transformer",
    "TransformedQuery",
    "QueryType",
    # Generation
    "LLMClient",
    "LLMProvider",
    "RAGGenerator",
    "GenerationResult",
    "ArabicPrompts",
    # Specialists
    "IslamicScholar",
    "ComparativeFiqhScholar",
    "IslamicDomain",
    "create_islamic_scholar",
    "create_comparative_fiqh_scholar",
    "AuthorityRanker",
    "CrossReferenceSystem",
    "MultiHopReasoning",
    "TimelineReconstructor",
    "create_authority_ranker",
    "create_cross_reference_system",
    "create_multi_hop_reasoning",
    # Agents
    "IslamicRAGAgent",
    "AgentTeam",
    "AgentRole",
    "create_agent",
    "EnhancedIslamicRAGAgent",
    "EnhancedAgentTeam",
    "EnhancedAgentRole",
    "create_enhanced_agent",
    "create_enhanced_agent_team",
    # Evaluation
    "RAGEvaluator",
    "ArabicTestDataset",
    "EvaluationSample",
    "IslamicEvaluationMetrics",
    "create_islamic_evaluator",
    # Monitoring
    "RAGMonitor",
    "CostTracker",
    "QueryLogger",
    "get_monitor",
]
