"""
Document Management Services
=========================
Services for updating, merging, and managing documents.

خدمات إدارة المستندات
"""

from dataclasses import dataclass
from typing import Optional, BinaryIO
from datetime import datetime
import logging
import hashlib

log = logging.getLogger(__name__)


@dataclass
class DocumentUpdateRequest:
    """Request to update a document.

    طلب تحديث مستند
    """

    document_id: str
    filename: Optional[str] = None
    content: Optional[bytes] = None
    status: Optional[str] = None


@dataclass
class DocumentMergeRequest:
    """Request to merge multiple documents.

    طلب دمج عدة مستندات
    """

    source_document_ids: list[str]
    target_document_id: Optional[str] = None  # If None, create new
    merged_filename: str = "merged_document.pdf"


@dataclass
class DocumentMergeResult:
    """Result of document merge operation.

    نتيجة عملية الدمج
    """

    merged_document_id: str
    source_document_ids: list[str]
    status: str
    error: Optional[str] = None


class DocumentUpdateService:
    """
    Service for updating document metadata.

    خدمة تحديث بيانات المستند
    """

    def __init__(self, document_repo, chunk_repo):
        """
        Initialize document update service.

        Args:
            document_repo: Document repository
            chunk_repo: Chunk repository
        """
        self._doc_repo = document_repo
        self._chunk_repo = chunk_repo

    def update_document(
        self,
        request: DocumentUpdateRequest,
        tenant_id: str,
    ) -> dict[str, Any]:
        """
        Update a document.

        Args:
            request: Update request with document_id and fields to update
            tenant_id: Tenant ID for authorization

        Returns:
            Updated document metadata

        تحديث مستند
        """
        # Check document exists and belongs to tenant
        doc = self._doc_repo.find_by_id(request.document_id)

        if not doc:
            raise ValueError(f"Document {request.document_id} not found")

        if doc["user_id"] != tenant_id:
            raise PermissionError("Access denied: Document belongs to another tenant")

        # Update fields
        updates = {}
        if request.filename is not None:
            updates["filename"] = request.filename
        if request.status is not None:
            updates["status"] = request.status

        if updates:
            updated_doc = self._doc_repo.update(
                document_id=request.document_id,
                updates=updates,
            )

            log.info(
                "Document updated", document_id=request.document_id, updates=list(updates.keys())
            )

            return updated_doc

        return doc

    def update_document_content(
        self,
        document_id: str,
        new_content: bytes,
        content_type: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """
        Update document content (file replacement).

        Args:
            document_id: Document ID to update
            new_content: New file content
            content_type: MIME type of new content
            tenant_id: Tenant ID for authorization

        Returns:
            Updated document with new file metadata

        تحديث محتوى المستند (استبدال الملف)
        """
        # Check document exists
        doc = self._doc_repo.find_by_id(document_id)

        if not doc:
            raise ValueError(f"Document {document_id} not found")

        if doc["user_id"] != tenant_id:
            raise PermissionError("Access denied: Document belongs to another tenant")

        # Calculate new file hash and size
        file_hash = hashlib.sha256(new_content).hexdigest()
        size_bytes = len(new_content)

        # Store new file (TODO: Implement file storage)
        file_path = self._store_file(document_id, new_content)

        # Update document metadata
        updates = {
            "file_path": file_path,
            "size_bytes": size_bytes,
            "file_sha256": file_hash,
            "content_type": content_type,
            "updated_at": datetime.utcnow().isoformat(),
        }

        updated_doc = self._doc_repo.update(
            document_id=document_id,
            updates=updates,
        )

        # Re-index document (queue background job)
        self._queue_reindexing(document_id)

        log.info(
            "Document content updated",
            document_id=document_id,
            size_bytes=size_bytes,
        )

        return updated_doc

    def _store_file(self, document_id: str, content: bytes) -> str:
        """Store file content (placeholder for storage integration)."""
        # TODO: Integrate with file storage (S3, GCS, Azure Blob)
        # For now, use local storage
        import os

        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{document_id}"
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path

    def _queue_reindexing(self, document_id: str):
        """Queue document for re-indexing (placeholder)."""
        # TODO: Integrate with Celery/Redis queue
        log.info("Document queued for re-indexing", document_id=document_id)


class DocumentMergeService:
    """
    Service for merging multiple documents.

    خدمة دمج المستندات
    """

    def __init__(self, document_repo, chunk_repo, file_storage):
        """
        Initialize document merge service.

        Args:
            document_repo: Document repository
            chunk_repo: Chunk repository
            file_storage: File storage service
        """
        self._doc_repo = document_repo
        self._chunk_repo = chunk_repo
        self._storage = file_storage

    def merge_documents(
        self,
        request: DocumentMergeRequest,
        tenant_id: str,
    ) -> DocumentMergeResult:
        """
        Merge multiple documents into one.

        Args:
            request: Merge request with source document IDs
            tenant_id: Tenant ID for authorization

        Returns:
            Merge result with new document ID

        دمج عدة مستندات في مستند واحد
        """
        # Validate source documents exist and belong to tenant
        source_docs = []
        for doc_id in request.source_document_ids:
            doc = self._doc_repo.find_by_id(doc_id)
            if not doc:
                raise ValueError(f"Source document {doc_id} not found")
            if doc["user_id"] != tenant_id:
                raise PermissionError(f"Access denied: Document {doc_id} belongs to another tenant")
            source_docs.append(doc)

        # Retrieve source document contents
        source_contents = []
        for doc in source_docs:
            content = self._storage.read_file(doc["file_path"])
            source_contents.append(content)

        # Merge contents (simple concatenation for PDFs)
        merged_content = self._merge_file_contents(
            source_contents,
            request.merged_filename,
        )

        # Store merged file
        merged_file_path = self._storage.store_file(
            tenant_id=tenant_id,
            filename=request.merged_filename,
            content=merged_content,
        )

        # Create new document or update target
        file_hash = hashlib.sha256(merged_content).hexdigest()
        size_bytes = len(merged_content)

        content_type = self._infer_content_type(request.merged_filename)

        if request.target_document_id:
            # Update existing document
            target_doc = self._doc_repo.find_by_id(request.target_document_id)

            if not target_doc:
                raise ValueError(f"Target document {request.target_document_id} not found")

            updated_doc = self._doc_repo.update(
                document_id=request.target_document_id,
                updates={
                    "file_path": merged_file_path,
                    "size_bytes": size_bytes,
                    "file_sha256": file_hash,
                    "content_type": content_type,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            )

            merged_document_id = request.target_document_id
        else:
            # Create new merged document
            merged_document_id = self._doc_repo.create(
                tenant_id=tenant_id,
                filename=request.merged_filename,
                file_path=merged_file_path,
                content_type=content_type,
                size_bytes=size_bytes,
                file_sha256=file_hash,
                status="created",
            )

        # Queue for indexing
        self._queue_reindexing(merged_document_id)

        log.info(
            "Documents merged",
            source_count=len(request.source_document_ids),
            merged_document_id=merged_document_id,
        )

        return DocumentMergeResult(
            merged_document_id=merged_document_id,
            source_document_ids=request.source_document_ids,
            status="completed",
            error=None,
        )

    def _merge_file_contents(
        self,
        contents: list[bytes],
        output_filename: str,
    ) -> bytes:
        """Merge file contents (placeholder implementation)."""
        # TODO: Implement actual merging based on file types
        # For PDFs, merge using PyPDF2
        # For text files, simple concatenation

        if output_filename.endswith(".txt") or output_filename.endswith(".md"):
            # Simple concatenation for text files
            separator = b"\n\n---\n\n"
            merged = separator.join(contents)
            return merged
        elif output_filename.endswith(".pdf"):
            # For PDFs, use first content as placeholder
            # TODO: Implement PDF merging with PyPDF2
            return contents[0]
        else:
            # For other types, concatenate
            return b"".join(contents)

    def _infer_content_type(self, filename: str) -> str:
        """Infer content type from filename."""
        extension = filename.split(".")[-1].lower()
        content_types = {
            "pdf": "application/pdf",
            "txt": "text/plain",
            "md": "text/markdown",
            "csv": "text/csv",
            "json": "application/json",
        }
        return content_types.get(extension, "application/octet-stream")

    def _queue_reindexing(self, document_id: str):
        """Queue document for re-indexing (placeholder)."""
        # TODO: Integrate with Celery/Redis queue
        log.info("Document queued for re-indexing", document_id=document_id)


# -----------------------------------------------------------------------------
# File Storage Interface (placeholder)
# -----------------------------------------------------------------------------


class FileStorageService:
    """
    Interface for file storage operations.

    واجهة عمليات تخزين الملفات
    """

    def read_file(self, file_path: str) -> bytes:
        """Read file from storage."""
        # TODO: Implement actual storage (S3, GCS, Azure Blob)
        with open(file_path, "rb") as f:
            return f.read()

    def store_file(
        self,
        tenant_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Store file to storage."""
        # TODO: Implement actual storage
        import os

        os.makedirs("uploads", exist_ok=True)
        import uuid

        file_id = str(uuid.uuid4())
        file_path = f"uploads/{file_id}_{filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path


if __name__ == "__main__":
    from unittest.mock import Mock

    # Test document management
    doc_repo = Mock()
    chunk_repo = Mock()
    storage = FileStorageService()

    # Test update
    update_service = DocumentUpdateService(doc_repo, chunk_repo)

    request = DocumentUpdateRequest(
        document_id="doc-123",
        filename="updated.pdf",
        status="indexed",
    )

    doc_repo.find_by_id.return_value = {
        "id": "doc-123",
        "user_id": "tenant-456",
        "filename": "original.pdf",
    }
    doc_repo.update.return_value = {"id": "doc-123", "filename": "updated.pdf"}

    result = update_service.update_document(request, "tenant-456")
    print(f"Updated: {result}")

    # Test merge
    merge_service = DocumentMergeService(doc_repo, chunk_repo, storage)

    doc_repo.find_by_id.side_effect = [
        {"id": "doc-1", "user_id": "tenant-456", "file_path": "uploads/doc-1.pdf"},
        {"id": "doc-2", "user_id": "tenant-456", "file_path": "uploads/doc-2.pdf"},
    ]

    merge_request = DocumentMergeRequest(
        source_document_ids=["doc-1", "doc-2"],
        merged_filename="merged.pdf",
    )

    merge_result = merge_service.merge_documents(merge_request, "tenant-456")
    print(f"Merge result: {merge_result}")
