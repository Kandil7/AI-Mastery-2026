"""
Shared Type Definitions for AI-Mastery-2026
============================================

Common type hints, protocols, and type aliases used across the codebase.

Usage:
------
    from src.core.utils.types import (
        DocumentLike,
        EmbeddingVector,
        AsyncCallable,
        SupportsSimilarity,
    )

Type Categories:
----------------
- Basic types: Common type aliases
- Protocols: Structural typing for duck-typing
- Callables: Function type definitions
- Generics: Generic type utilities
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (Any, AsyncIterable, Awaitable, Callable, Dict, Generic,
                    Iterable, List, Literal, Mapping, Optional, Protocol,
                    Sequence, Tuple, TypeVar, Union, runtime_checkable)

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ============================================================
# Basic Type Aliases
# ============================================================

# Numeric types
Number = Union[int, float]
NumericArray = Union[List[Number], NDArray[np.number]]

# Embedding types
EmbeddingVector = NDArray[np.float32]
EmbeddingMatrix = NDArray[np.float32]

# Text types
TextLike = Union[str, bytes]
PathLike = Union[str, Path]

# ID types (concrete alias and TypeVar for Protocols)
IDType = Union[str, int]
IDTypeVar = TypeVar("IDTypeVar", bound=IDType)

# Score types
ScoreType = float
SimilarityScore = float  # 0.0 to 1.0
DistanceScore = float  # Lower is better

# Batch types
BatchSize = int
BatchIndex = int

# Configuration types
ConfigDict = Dict[str, Any]
ConfigMapping = Mapping[str, Any]

# Metadata types
MetadataDict = Dict[str, Any]
MetadataMapping = Mapping[str, Any]


# ============================================================
# Generic Type Variables
# ============================================================

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

# Document type variable
DocumentT = TypeVar("DocumentT", bound="DocumentProtocol")

# Chunk type variable
ChunkT = TypeVar("ChunkT", bound="ChunkProtocol")

# Model type variable
ModelT = TypeVar("ModelT")


# ============================================================
# Protocols (Structural Types)
# ============================================================


@runtime_checkable
class DocumentProtocol(Protocol):
    """Protocol for document-like objects."""

    id: IDType
    content: str
    metadata: MetadataDict

    def __len__(self) -> int:
        """Return document length (e.g., character count)."""
        ...


@runtime_checkable
class ChunkProtocol(Protocol):
    """Protocol for chunk-like objects."""

    id: IDType
    content: str
    doc_id: IDType
    chunk_index: int
    metadata: MetadataDict

    start_char: Optional[int]
    end_char: Optional[int]


@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    """Protocol for embedding models."""

    dim: int

    def encode(self, text: str) -> EmbeddingVector:
        """Encode text to embedding."""
        ...

    def encode_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> EmbeddingMatrix:
        """Encode multiple texts to embeddings."""
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector stores."""

    dim: int
    size: int

    def add(
        self,
        embedding: EmbeddingVector,
        document_id: IDType,
        metadata: Optional[MetadataDict] = None,
    ) -> None:
        """Add a vector to the store."""
        ...

    def add_batch(
        self,
        embeddings: EmbeddingMatrix,
        document_ids: Sequence[IDType],
        metadatas: Optional[Sequence[MetadataDict]] = None,
    ) -> None:
        """Add multiple vectors to the store."""
        ...

    def search(
        self,
        embedding: EmbeddingVector,
        top_k: int = 5,
        filters: Optional[MetadataDict] = None,
    ) -> List["SearchResultProtocol"]:
        """Search for similar vectors."""
        ...

    def delete(self, document_id: IDType) -> bool:
        """Delete a vector from the store."""
        ...

    def clear(self) -> None:
        """Clear all vectors from the store."""
        ...


@runtime_checkable
class SearchResultProtocol(Protocol):
    """Protocol for search results."""

    id: IDType
    score: ScoreType
    metadata: Optional[MetadataDict]


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Protocol for chunking strategies."""

    def chunk(self, document: DocumentProtocol) -> List[ChunkProtocol]:
        """Split document into chunks."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrieval strategies."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[MetadataDict] = None,
    ) -> List[SearchResultProtocol]:
        """Retrieve relevant documents."""
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for reranking strategies."""

    def rerank(
        self,
        query: str,
        results: List[SearchResultProtocol],
        top_k: Optional[int] = None,
    ) -> List[SearchResultProtocol]:
        """Rerank search results."""
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM models."""

    model_name: str
    max_tokens: int

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        ...

    async def generate_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text asynchronously."""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in cache."""
        ...

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        ...

    def clear(self) -> None:
        """Clear cache."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


@runtime_checkable
class SerializableProtocol(Protocol):
    """Protocol for serializable objects."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializableProtocol":
        """Deserialize from dictionary."""
        ...


@runtime_checkable
class ComparableProtocol(Protocol):
    """Protocol for comparable objects."""

    def __lt__(self, other: Any) -> bool:
        """Less than comparison."""
        ...

    def __le__(self, other: Any) -> bool:
        """Less than or equal comparison."""
        ...

    def __gt__(self, other: Any) -> bool:
        """Greater than comparison."""
        ...

    def __ge__(self, other: Any) -> bool:
        """Greater than or equal comparison."""
        ...


# ============================================================
# Callable Types
# ============================================================

# Synchronous callables
SyncCallable = Callable[..., T]
SyncCallableNoArgs = Callable[[], T]
SyncCallable1Arg = Callable[[T], T]
SyncCallable2Args = Callable[[T, T], T]

# Asynchronous callables
AsyncCallable = Callable[..., Awaitable[T]]
AsyncCallableNoArgs = Callable[[], Awaitable[T]]
AsyncIterableType = AsyncIterable[T]

# Processor types
Processor = Callable[[T], T]
AsyncProcessor = Callable[[T], Awaitable[T]]
Transformer = Callable[[T], V]
AsyncTransformer = Callable[[T], Awaitable[V]]

# Validator types
Validator = Callable[[T], bool]
ValidatorWithMessage = Callable[[T], Tuple[bool, str]]

# Factory types
Factory = Callable[[], T]
FactoryWithArgs = Callable[..., T]

# Callback types
Callback = Callable[[T], None]
AsyncCallback = Callable[[T], Awaitable[None]]
ErrorCallback = Callable[[Exception], None]
SuccessCallback = Callable[[T], None]


# ============================================================
# Result Types
# ============================================================


@dataclass
class Result(Generic[T]):
    """
    Generic result type for operations that can fail.

    Example:
        >>> def safe_divide(a: float, b: float) -> Result[float]:
        ...     if b == 0:
        ...         return Result.failure("Division by zero")
        ...     return Result.success(a / b)
        >>>
        >>> result = safe_divide(10, 2)
        >>> if result.is_success:
        ...     print(result.value)
    """

    value: Optional[T] = None
    error: Optional[str] = None
    is_success: bool = True

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Create a successful result."""
        return cls(value=value, is_success=True)

    @classmethod
    def failure(cls, error: str) -> "Result[T]":
        """Create a failed result."""
        return cls(error=error, is_success=False)

    def map(self, func: Callable[[T], V]) -> "Result[V]":
        """Map the value if successful."""
        if self.is_success and self.value is not None:
            return Result.success(func(self.value))
        return Result.failure(self.error or "Unknown error")

    def and_then(self, func: Callable[[T], "Result[V]"]) -> "Result[V]":
        """Chain operations if successful."""
        if self.is_success and self.value is not None:
            return func(self.value)
        return Result.failure(self.error or "Unknown error")

    def unwrap(self) -> T:
        """Get the value or raise an exception."""
        if not self.is_success:
            raise ValueError(self.error or "Unknown error")
        if self.value is None:
            raise ValueError("Value is None")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get the value or return default."""
        return self.value if self.is_success and self.value is not None else default


# ============================================================
# Time and Duration Types
# ============================================================

DurationSeconds = float
DurationMilliseconds = float
Timestamp = datetime
TimestampSeconds = float
TimestampMilliseconds = float


# ============================================================
# Pagination Types
# ============================================================


@dataclass
class PageInfo:
    """Pagination information."""

    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

    @classmethod
    def create(
        cls,
        page: int,
        page_size: int,
        total_items: int,
    ) -> "PageInfo":
        """Create PageInfo from parameters."""
        total_pages = (total_items + page_size - 1) // page_size
        return cls(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )


@dataclass
class PaginatedResult(Generic[T]):
    """Paginated result set."""

    items: List[T]
    page_info: PageInfo


# ============================================================
# Performance Types
# ============================================================


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""

    duration_ms: DurationMilliseconds
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    items_processed: int = 0

    @property
    def throughput(self) -> float:
        """Items processed per second."""
        if self.duration_ms == 0:
            return 0.0
        return self.items_processed / (self.duration_ms / 1000)


# ============================================================
# Strategy Pattern Types
# ============================================================


class Strategy(Protocol[T]):
    """Protocol for strategy pattern."""

    def execute(self, context: Any) -> T:
        """Execute the strategy."""
        ...


# ============================================================
# Builder Pattern Types
# ============================================================


class Builder(Protocol[T]):
    """Protocol for builder pattern."""

    def build(self) -> T:
        """Build the final object."""
        ...


# ============================================================
# Observer Pattern Types
# ============================================================


class Observer(Protocol[T]):
    """Protocol for observer pattern."""

    def update(self, subject: T) -> None:
        """Update observer with subject state."""
        ...


class Subject(Protocol[T]):
    """Protocol for subject pattern."""

    def attach(self, observer: Observer[T]) -> None:
        """Attach an observer."""
        ...

    def detach(self, observer: Observer[T]) -> None:
        """Detach an observer."""
        ...

    def notify(self) -> None:
        """Notify all observers."""
        ...


# ============================================================
# Repository Pattern Types
# ============================================================


class Repository(Protocol[T, IDTypeVar]):
    """Protocol for repository pattern."""

    def get(self, id: IDTypeVar) -> Optional[T]:
        """Get entity by ID."""
        ...

    def get_all(self) -> List[T]:
        """Get all entities."""
        ...

    def add(self, entity: T) -> IDTypeVar:
        """Add entity."""
        ...

    def update(self, entity: T) -> None:
        """Update entity."""
        ...

    def delete(self, id: IDType) -> bool:
        """Delete entity by ID."""
        ...


# ============================================================
# Utility Functions
# ============================================================


def is_numeric(value: Any) -> bool:
    """Check if value is numeric."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_sequence(value: Any) -> bool:
    """Check if value is a sequence (but not string)."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def is_mapping(value: Any) -> bool:
    """Check if value is a mapping."""
    return isinstance(value, Mapping)


def ensure_list(value: Union[T, List[T]]) -> List[T]:
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    return [value]


def ensure_sequence(value: Union[T, Sequence[T]]) -> Sequence[T]:
    """Ensure value is a sequence."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return [value]

