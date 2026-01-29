"""
Domain Errors
==============
Custom exceptions for domain-level error handling.
These exceptions are raised by domain logic and caught at API boundaries.

أخطاء المجال - استثناءات مخصصة للتعامل مع الأخطاء على مستوى المجال
"""


class DomainError(Exception):
    """
    Base class for all domain errors.
    
    الفئة الأساسية لجميع أخطاء المجال
    """
    
    def __init__(self, message: str, code: str = "DOMAIN_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


# =============================================================================
# Document Errors / أخطاء المستندات
# =============================================================================

class DocumentNotFoundError(DomainError):
    """Raised when a document is not found."""
    
    def __init__(self, document_id: str, tenant_id: str | None = None) -> None:
        message = f"Document not found: {document_id}"
        if tenant_id:
            message += f" for tenant: {tenant_id}"
        super().__init__(message, code="DOCUMENT_NOT_FOUND")
        self.document_id = document_id
        self.tenant_id = tenant_id


class DocumentAlreadyExistsError(DomainError):
    """Raised when trying to create a duplicate document (idempotency)."""
    
    def __init__(self, document_id: str, file_hash: str) -> None:
        super().__init__(
            f"Document with same content already exists: {document_id}",
            code="DOCUMENT_EXISTS"
        )
        self.document_id = document_id
        self.file_hash = file_hash


class DocumentProcessingError(DomainError):
    """Raised when document processing fails."""
    
    def __init__(self, document_id: str, reason: str) -> None:
        super().__init__(
            f"Failed to process document {document_id}: {reason}",
            code="DOCUMENT_PROCESSING_FAILED"
        )
        self.document_id = document_id
        self.reason = reason


# =============================================================================
# File Errors / أخطاء الملفات
# =============================================================================

class InvalidFileError(DomainError):
    """Raised for invalid file upload attempts."""
    
    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid file: {reason}", code="INVALID_FILE")
        self.reason = reason


class FileTooLargeError(DomainError):
    """Raised when uploaded file exceeds size limit."""
    
    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        size_mb = size_bytes / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(
            f"File too large: {size_mb:.1f}MB exceeds limit of {max_mb:.1f}MB",
            code="FILE_TOO_LARGE"
        )
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class UnsupportedFileTypeError(DomainError):
    """Raised when file type is not supported."""
    
    def __init__(self, extension: str, allowed: list[str]) -> None:
        super().__init__(
            f"Unsupported file type: {extension}. Allowed: {', '.join(allowed)}",
            code="UNSUPPORTED_FILE_TYPE"
        )
        self.extension = extension
        self.allowed = allowed


# =============================================================================
# Extraction Errors / أخطاء الاستخراج
# =============================================================================

class TextExtractionError(DomainError):
    """Raised when text extraction fails."""
    
    def __init__(self, reason: str) -> None:
        super().__init__(f"Text extraction failed: {reason}", code="EXTRACTION_FAILED")
        self.reason = reason


class NoTextExtractedError(DomainError):
    """Raised when no text could be extracted from document."""
    
    def __init__(self, document_id: str) -> None:
        super().__init__(
            f"No text extracted from document: {document_id}",
            code="NO_TEXT_EXTRACTED"
        )
        self.document_id = document_id


# =============================================================================
# Search Errors / أخطاء البحث
# =============================================================================

class SearchError(DomainError):
    """Raised when search operation fails."""
    
    def __init__(self, reason: str) -> None:
        super().__init__(f"Search failed: {reason}", code="SEARCH_FAILED")
        self.reason = reason


class EmbeddingError(DomainError):
    """Raised when embedding generation fails."""
    
    def __init__(self, reason: str) -> None:
        super().__init__(f"Embedding failed: {reason}", code="EMBEDDING_FAILED")
        self.reason = reason


# =============================================================================
# LLM Errors / أخطاء نموذج اللغة
# =============================================================================

class LLMError(DomainError):
    """Raised when LLM call fails."""
    
    def __init__(self, reason: str) -> None:
        super().__init__(f"LLM generation failed: {reason}", code="LLM_FAILED")
        self.reason = reason


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is hit."""
    
    def __init__(self, retry_after: int | None = None) -> None:
        super().__init__("Rate limit exceeded")
        self.code = "LLM_RATE_LIMITED"
        self.retry_after = retry_after


# =============================================================================
# Auth Errors / أخطاء المصادقة
# =============================================================================

class AuthenticationError(DomainError):
    """Raised for authentication failures."""
    
    def __init__(self, reason: str = "Invalid credentials") -> None:
        super().__init__(reason, code="AUTH_FAILED")


class AuthorizationError(DomainError):
    """Raised when user lacks permission."""
    
    def __init__(self, reason: str = "Access denied") -> None:
        super().__init__(reason, code="ACCESS_DENIED")


class TenantAccessError(DomainError):
    """Raised when accessing resources of another tenant."""
    
    def __init__(self, resource_id: str) -> None:
        super().__init__(
            f"Access denied to resource: {resource_id}",
            code="TENANT_ACCESS_DENIED"
        )
        self.resource_id = resource_id
