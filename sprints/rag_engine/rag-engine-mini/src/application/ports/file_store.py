"""
File Store Port
================
Interface for file storage operations.

منفذ تخزين الملفات
"""

from typing import Protocol, AsyncIterator, Tuple

from src.domain.entities import StoredFile


class FileStorePort(Protocol):
    """
    Port for file storage operations.
    
    Implementations: Local filesystem, S3, GCS, etc.
    
    Design Decision: Async interface for non-blocking I/O.
    File operations can be slow, especially with remote storage.
    
    قرار التصميم: واجهة غير متزامنة للإدخال/الإخراج غير المحجوب
    """
    
    async def save_upload(
        self,
        *,
        tenant_id: str,
        upload_filename: str,
        content_type: str,
        data: bytes,
    ) -> StoredFile:
        """
        Save an uploaded file.
        
        Args:
            tenant_id: Owner tenant (for path/isolation)
            upload_filename: Original filename
            content_type: MIME type
            data: File content
            
        Returns:
            Stored file information
            
        Raises:
            FileTooLargeError: If file exceeds size limit
            InvalidFileError: If file validation fails
        """
        ...

    async def save_upload_stream(
        self,
        *,
        tenant_id: str,
        upload_filename: str,
        content_type: str,
        data_stream: AsyncIterator[bytes],
    ) -> Tuple[StoredFile, str]:
        """
        Save an uploaded file from a stream.

        Args:
            tenant_id: Owner tenant (for path/isolation)
            upload_filename: Original filename
            content_type: MIME type
            data_stream: Async stream of bytes (chunked upload)

        Returns:
            Tuple of (StoredFile, sha256 hex digest)

        Raises:
            FileTooLargeError: If file exceeds size limit
            InvalidFileError: If file validation fails
        """
        ...
    
    async def delete(self, path: str) -> bool:
        """
        Delete a stored file.
        
        Args:
            path: File path (from StoredFile.path)
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    async def exists(self, path: str) -> bool:
        """
        Check if file exists.
        
        Args:
            path: File path
            
        Returns:
            True if file exists
        """
        ...
