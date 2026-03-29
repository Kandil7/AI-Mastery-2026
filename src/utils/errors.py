"""
Unified Error Handling for AI-Mastery-2026
==========================================

All exceptions inherit from AIMasteryError for consistent handling.

Usage:
------
    from src.utils.errors import RAGError, ChunkingError, EmbeddingError
    
    try:
        chunks = chunker.chunk(document)
    except ChunkingError as e:
        logger.error(f"Chunking failed: {e.message}", extra=e.to_dict())
        raise

Error Hierarchy:
----------------
    AIMasteryError (base)
    ├── RAGError
    │   ├── ChunkingError
    │   ├── EmbeddingError
    │   ├── RetrievalError
    │   ├── VectorStoreError
    │   └── RerankingError
    ├── ModelError
    │   ├── TrainingError
    │   ├── InferenceError
    │   └── FineTuningError
    ├── ConfigurationError
    ├── DataError
    │   ├── ValidationError
    │   ├── LoadingError
    │   └── ProcessingError
    └── InfrastructureError
        ├── DatabaseError
        ├── CacheError
        └── APIError
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ErrorContext:
    """Structured error context for debugging."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    module: str = ""
    function: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


class AIMasteryError(Exception):
    """
    Base exception for all AI-Mastery-2026 errors.
    
    All custom exceptions should inherit from this class to ensure
    consistent error handling, logging, and API responses.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code (e.g., "CHUNKING_FAILED")
        context: Additional context for debugging
        cause: Original exception that triggered this error
        is_retryable: Whether the operation can be retried
    
    Example:
        >>> raise AIMasteryError(
        ...     message="Operation failed",
        ...     error_code="OPERATION_FAILED",
        ...     context={"operation_id": "123"},
        ...     cause=original_exception,
        ... )
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        is_retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.__cause__ = cause
        self.is_retryable = is_retryable
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for logging/API responses.
        
        Returns:
            Dictionary with all error information
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "is_retryable": self.is_retryable,
            "has_cause": self.__cause__ is not None,
        }
    
    def to_log_extra(self) -> Dict[str, Any]:
        """
        Convert error to extra dict for logging.
        
        Returns:
            Dictionary suitable for logger.extra parameter
        """
        return {
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "error_message": self.message,
            "error_context": self.context,
        }
    
    def __str__(self) -> str:
        if self.__cause__:
            return f"{self.error_code}: {self.message} (caused by: {type(self.__cause__).__name__})"
        return f"{self.error_code}: {self.message}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, error_code={self.error_code!r})"


# ============================================================
# RAG Errors
# ============================================================

class RAGError(AIMasteryError):
    """Base error for RAG operations."""
    pass


class ChunkingError(RAGError):
    """
    Error during document chunking.
    
    Common causes:
    - Invalid document format
    - Chunking strategy misconfiguration
    - Text encoding issues
    """
    pass


class EmbeddingError(RAGError):
    """
    Error during embedding generation.
    
    Common causes:
    - Model loading failure
    - Invalid input text
    - Dimension mismatch
    - API rate limiting (for cloud embeddings)
    """
    pass


class RetrievalError(RAGError):
    """
    Error during retrieval.
    
    Common causes:
    - Vector store connection failure
    - Query processing error
    - Index corruption
    """
    pass


class VectorStoreError(RAGError):
    """
    Error in vector store operations.
    
    Common causes:
    - Connection failure
    - Authentication error
    - Index not found
    - Dimension mismatch
    """
    pass


class RerankingError(RAGError):
    """
    Error during reranking.
    
    Common causes:
    - Model loading failure
    - Invalid input format
    - Score computation error
    """
    pass


class QueryEnhancementError(RAGError):
    """
    Error during query enhancement.
    
    Common causes:
    - LLM API failure
    - Invalid query format
    - Enhancement strategy error
    """
    pass


# ============================================================
# Model Errors
# ============================================================

class ModelError(AIMasteryError):
    """Base error for model operations."""
    pass


class TrainingError(ModelError):
    """
    Error during model training.
    
    Common causes:
    - Invalid training data
    - Hyperparameter misconfiguration
    - Resource exhaustion (OOM)
    - Convergence failure
    """
    pass


class InferenceError(ModelError):
    """
    Error during model inference.
    
    Common causes:
    - Model not loaded
    - Input format mismatch
    - Resource exhaustion
    """
    pass


class FineTuningError(ModelError):
    """
    Error during model fine-tuning.
    
    Common causes:
    - Invalid dataset format
    - Adapter configuration error
    - Resource exhaustion
    """
    pass


class ModelLoadingError(ModelError):
    """
    Error loading a model.
    
    Common causes:
    - Model file not found
    - Corrupted weights
    - Architecture mismatch
    - Missing dependencies
    """
    pass


# ============================================================
# Configuration Errors
# ============================================================

class ConfigurationError(AIMasteryError):
    """
    Error in configuration.
    
    Common causes:
    - Missing required config value
    - Invalid config format
    - Type mismatch in config
    """
    pass


class ValidationError(ConfigurationError):
    """
    Validation error for configuration or input data.
    """
    pass


# ============================================================
# Data Errors
# ============================================================

class DataError(AIMasteryError):
    """Base error for data operations."""
    pass


class DataLoadingError(DataError):
    """
    Error loading data.
    
    Common causes:
    - File not found
    - Invalid file format
    - Encoding issues
    """
    pass


class DataProcessingError(DataError):
    """
    Error processing data.
    
    Common causes:
    - Invalid data format
    - Processing pipeline error
    - Resource exhaustion
    """
    pass


class DataValidationError(DataError):
    """
    Error validating data.
    
    Common causes:
    - Schema mismatch
    - Missing required fields
    - Type validation failure
    """
    pass


# ============================================================
# Infrastructure Errors
# ============================================================

class InfrastructureError(AIMasteryError):
    """Base error for infrastructure operations."""
    pass


class DatabaseError(InfrastructureError):
    """
    Error in database operations.
    
    Common causes:
    - Connection failure
    - Query execution error
    - Constraint violation
    """
    pass


class CacheError(InfrastructureError):
    """
    Error in cache operations.
    
    Common causes:
    - Connection failure
    - Serialization error
    - Cache eviction issues
    """
    pass


class APIError(InfrastructureError):
    """
    Error in external API calls.
    
    Common causes:
    - Network failure
    - Authentication error
    - Rate limiting
    - Invalid response format
    """
    pass


class AuthenticationError(APIError):
    """
    Authentication error.
    
    Common causes:
    - Invalid credentials
    - Expired token
    - Missing authentication
    """
    pass


class RateLimitError(APIError):
    """
    Rate limit exceeded error.
    """
    pass


# ============================================================
# Agent Errors
# ============================================================

class AgentError(AIMasteryError):
    """Base error for agent operations."""
    pass


class ToolExecutionError(AgentError):
    """
    Error executing an agent tool.
    
    Common causes:
    - Tool not found
    - Invalid tool arguments
    - Tool execution failure
    """
    pass


class AgentOrchestrationError(AgentError):
    """
    Error in agent orchestration.
    
    Common causes:
    - Invalid workflow definition
    - Circular dependency
    - Execution timeout
    """
    pass


# ============================================================
# Safety Errors
# ============================================================

class SafetyError(AIMasteryError):
    """Base error for safety operations."""
    pass


class ContentModerationError(SafetyError):
    """
    Error in content moderation.
    
    Common causes:
    - Model loading failure
    - Classification error
    """
    pass


class GuardrailViolation(SafetyError):
    """
    Raised when a guardrail is violated.
    
    Common causes:
    - PII detected in input/output
    - Harmful content detected
    - Policy violation
    """
    pass


# ============================================================
# Error Handling Utilities
# ============================================================

def raise_with_context(
    exception_class: type[Exception],
    message: str,
    context: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
    Raise an exception with standardized context.
    
    Args:
        exception_class: Exception class to raise
        message: Error message
        context: Additional context
        cause: Original exception
    
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     raise_with_context(
        ...         OperationError,
        ...         "Operation failed",
        ...         context={"operation_id": "123"},
        ...         cause=e,
        ...     )
    """
    raise exception_class(
        message=message,
        error_code=exception_class.__name__.upper().replace("ERROR", "_FAILED"),
        context=context,
        cause=cause,
    )


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.
    
    Args:
        error: Exception to check
    
    Returns:
        True if the error is retryable
    
    Example:
        >>> try:
        ...     api_call()
        ... except Exception as e:
        ...     if is_retryable_error(e):
        ...         retry()
    """
    if isinstance(error, AIMasteryError):
        return error.is_retryable
    
    # Common retryable errors
    retryable_types = (
        RateLimitError,
        CacheError,
        APIError,
    )
    
    return isinstance(error, retryable_types)


def get_error_chain(error: Exception) -> List[Exception]:
    """
    Get the full chain of exceptions.
    
    Args:
        error: Exception to trace
    
    Returns:
        List of exceptions from current to root cause
    
    Example:
        >>> chain = get_error_chain(error)
        >>> for exc in chain:
        ...     print(f"{type(exc).__name__}: {exc}")
    """
    chain = []
    current = error
    
    while current is not None:
        chain.append(current)
        current = current.__cause__
    
    return chain


def format_error_for_api(error: Exception) -> Dict[str, Any]:
    """
    Format an error for API response.
    
    Args:
        error: Exception to format
    
    Returns:
        Dictionary suitable for JSON API response
    
    Example:
        >>> error_response = format_error_for_api(exception)
        >>> return JSONResponse(status_code=500, content=error_response)
    """
    if isinstance(error, AIMasteryError):
        return {
            "error": {
                "code": error.error_code,
                "message": error.message,
                "context": error.context,
                "type": type(error).__name__,
            }
        }
    
    return {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": str(error),
            "type": type(error).__name__,
        }
    }
