"""
RAG Engine SDK - Exceptions Module

Custom exceptions for error handling.
"""

from typing import Optional


class RAGEngineError(Exception):
    """Base exception for RAG Engine SDK."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(RAGEngineError):
    """Raised when authentication fails (401)."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class RateLimitError(RAGEngineError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(RAGEngineError):
    """Raised when request validation fails (422)."""

    def __init__(self, message: str = "Validation error"):
        super().__init__(message, status_code=422)


class ServerError(RAGEngineError):
    """Raised when server error occurs (5xx)."""

    def __init__(self, message: str = "Server error"):
        super().__init__(message, status_code=500)


class NotFoundError(RAGEngineError):
    """Raised when resource is not found (404)."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class ConflictError(RAGEngineError):
    """Raised when there's a conflict (409)."""

    def __init__(self, message: str = "Conflict"):
        super().__init__(message, status_code=409)
