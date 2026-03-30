"""
Module 3.3: RAG (Retrieval-Augmented Generation)

Production-ready RAG implementations:
- Orchestrator: LangChain, LlamaIndex, custom orchestration
- Retrievers: Similarity search, multi-query, HyDE, contextual compression
- Memory: Conversation buffer, summary, vector store, entity memory
- Evaluation: RAGAS integration, context precision, faithfulness, answer relevancy
"""

from .orchestrator import (
    RAGOrchestrator,
    LangChainOrchestrator,
    LlamaIndexOrchestrator,
    RAGConfig,
    RAGResponse,
)
from .retrievers import (
    BaseRetriever,
    SimilarityRetriever,
    MultiQueryRetriever,
    HyDERetriever,
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from .memory import (
    ConversationMemory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
    VectorStoreMemory,
    EntityMemory,
    MemoryManager,
)
from .evaluation import (
    RAGEvaluator,
    RAGASWrapper,
    EvaluationResult,
    ContextPrecisionMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

__all__ = [
    # Orchestrator
    "RAGOrchestrator",
    "LangChainOrchestrator",
    "LlamaIndexOrchestrator",
    "RAGConfig",
    "RAGResponse",
    # Retrievers
    "BaseRetriever",
    "SimilarityRetriever",
    "MultiQueryRetriever",
    "HyDERetriever",
    "ContextualCompressionRetriever",
    "EnsembleRetriever",
    # Memory
    "ConversationMemory",
    "ConversationBufferMemory",
    "ConversationSummaryMemory",
    "VectorStoreMemory",
    "EntityMemory",
    "MemoryManager",
    # Evaluation
    "RAGEvaluator",
    "RAGASWrapper",
    "EvaluationResult",
    "ContextPrecisionMetric",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
]

__version__ = "1.0.0"
