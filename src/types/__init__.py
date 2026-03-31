"""
Shared Type Definitions for AI-Mastery-2026
============================================

Centralized type definitions used across the codebase.

This module re-exports types from src.core.utils.types for cleaner imports.

Usage:
    >>> from src.types import DocumentProtocol, EmbeddingVector, SimilarityScore
    >>> def process(doc: DocumentProtocol) -> EmbeddingVector:
    ...     ...
"""

# Re-export from core.utils.types
from src.core.utils.types import (
    # Basic types
    TextLike,
    PathLike,
    IDType,
    IDTypeVar,
    ScoreType,
    SimilarityScore,
    DistanceScore,
    BatchSize,
    BatchIndex,
    ConfigDict,
    ConfigMapping,
    MetadataDict,
    MetadataMapping,
    # Generics
    T,
    K,
    V,
    T_co,
    T_contra,
    DocumentT,
    ChunkT,
    # Protocols
    DocumentProtocol,
    ChunkProtocol,
    EmbeddingModelProtocol,
    VectorStoreProtocol,
    SearchResultProtocol,
    ChunkerProtocol,
    RetrieverProtocol,
    RerankerProtocol,
    LLMProtocol,
    CacheProtocol,
    SerializableProtocol,
    ComparableProtocol,
    # Result types
    Result,
    # Pagination
    PageInfo,
    PaginatedResult,
    # Performance
    PerformanceMetrics,
    # Strategy pattern
    Strategy,
    # Builder pattern
    Builder,
    # Observer pattern
    Observer,
    Subject,
    # Repository pattern
    Repository,
)

# Additional types specific to AI-Mastery-2026
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np


# ============================================================
# NUMPY ARRAY TYPES
# ============================================================

EmbeddingVector = np.ndarray  # 1D array of floats
EmbeddingMatrix = np.ndarray  # 2D array of floats
AttentionMatrix = np.ndarray  # 2D array of floats (seq_len x seq_len)
Logits = np.ndarray  # 1D or 2D array of floats


# ============================================================
# TENSOR TYPES (PyTorch)
# ============================================================

try:
    import torch
    Tensor = torch.Tensor
    OptionalTensor = Optional[torch.Tensor]
except ImportError:
    Tensor = np.ndarray  # type: ignore
    OptionalTensor = Optional[np.ndarray]  # type: ignore


# ============================================================
# MODEL OUTPUT TYPES
# ============================================================

@dataclass
class ModelOutput:
    """Base class for model outputs."""
    loss: Optional[float] = None
    logits: Optional[np.ndarray] = None
    hidden_states: Optional[Tuple[np.ndarray, ...]] = None
    attentions: Optional[Tuple[np.ndarray, ...]] = None


@dataclass
class TransformerOutput(ModelOutput):
    """Transformer model output."""
    last_hidden_state: Optional[np.ndarray] = None
    pooler_output: Optional[np.ndarray] = None


@dataclass
class RAGOutput:
    """RAG system output."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    latency_ms: float


# ============================================================
# EVALUATION TYPES
# ============================================================

@dataclass
class MetricResult:
    """Result of a metric computation."""
    name: str
    value: float
    metadata: Dict[str, Any] = None


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    metrics: Dict[str, float]
    predictions: List[Any]
    ground_truth: List[Any]


# ============================================================
# PROTOCOLS FOR AI COMPONENTS
# ============================================================

@runtime_checkable
class Embeddable(Protocol):
    """Protocol for objects that can be embedded."""
    def to_text(self) -> str:
        """Convert to text for embedding."""
        ...


@runtime_checkable
class Trainable(Protocol):
    """Protocol for trainable models."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Trainable":
        """Train the model."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...


@runtime_checkable
class Saveable(Protocol):
    """Protocol for saveable objects."""
    def save(self, path: Path) -> None:
        """Save to disk."""
        ...

    @classmethod
    def load(cls, path: Path) -> "Saveable":
        """Load from disk."""
        ...


# ============================================================
# TYPE ALIASES FOR COMMON PATTERNS
# ============================================================

# Document types
DocumentId = str
ChunkId = str
DocumentContent = str

# Query types
QueryText = str
QueryEmbedding = np.ndarray

# Retrieval types
RetrievalScore = float
RerankingScore = float

# Training types
LearningRate = float
LossValue = float
Gradient = np.ndarray

# ============================================================
# __ALL__ EXPORTS
# ============================================================

__all__ = [
    # Re-exported types
    "TextLike",
    "PathLike",
    "IDType",
    "IDTypeVar",
    "ScoreType",
    "SimilarityScore",
    "DistanceScore",
    "BatchSize",
    "BatchIndex",
    "ConfigDict",
    "ConfigMapping",
    "MetadataDict",
    "MetadataMapping",
    "T",
    "K",
    "V",
    "T_co",
    "T_contra",
    "DocumentT",
    "ChunkT",
    "DocumentProtocol",
    "ChunkProtocol",
    "EmbeddingModelProtocol",
    "VectorStoreProtocol",
    "SearchResultProtocol",
    "ChunkerProtocol",
    "RetrieverProtocol",
    "RerankerProtocol",
    "LLMProtocol",
    "CacheProtocol",
    "SerializableProtocol",
    "ComparableProtocol",
    "Result",
    "PageInfo",
    "PaginatedResult",
    "PerformanceMetrics",
    "Strategy",
    "Builder",
    "Observer",
    "Subject",
    "Repository",
    # Numpy types
    "EmbeddingVector",
    "EmbeddingMatrix",
    "AttentionMatrix",
    "Logits",
    # Tensor types
    "Tensor",
    "OptionalTensor",
    # Model outputs
    "ModelOutput",
    "TransformerOutput",
    "RAGOutput",
    # Evaluation
    "MetricResult",
    "EvaluationResult",
    # Protocols
    "Embeddable",
    "Trainable",
    "Saveable",
    # Type aliases
    "DocumentId",
    "ChunkId",
    "DocumentContent",
    "QueryText",
    "QueryEmbedding",
    "RetrievalScore",
    "RerankingScore",
    "LearningRate",
    "LossValue",
    "Gradient",
]
