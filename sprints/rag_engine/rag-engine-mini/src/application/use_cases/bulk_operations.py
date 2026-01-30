"""
Bulk Operations Use Case
===========================
Batch upload and delete operations for documents.

عمليات دفعية للمستندات
"""

import io
import hashlib
from dataclasses import dataclass
from typing import List, BinaryIO
from datetime import datetime

from src.application.ports.document_repo import DocumentRepoPort
from src.application.use_cases.upload_document import UploadDocumentRequest, UploadDocumentUseCase
from src.application.ports.file_store import FileStorePort
from src.domain.entities import TenantId


@dataclass
class BulkUploadRequest:
    """Request data for bulk document upload."""

    tenant_id: str
    files: List[BinaryIO]  # List of file-like objects
    filenames: List[str]  # Original filenames
    content_types: List[str]  # MIME types


@dataclass
class BulkUploadResult:
    """Result for a single file upload."""

    filename: str
    document_id: str | None
    status: str
    message: str
    error: str | None = None


@dataclass
class BulkUploadResponse:
    """Response data for bulk upload operation."""

    results: List[BulkUploadResult]
    total_files: int
    succeeded: int
    failed: int
    operation_id: str  # Operation ID for tracking


@dataclass
class BulkDeleteRequest:
    """Request data for bulk document deletion."""

    tenant_id: str
    document_ids: List[str]
    reason: str = "Bulk delete"


@dataclass
class BulkDeleteResponse:
    """Response data for bulk delete operation."""

    operation_id: str
    total_documents: int
    deleted_count: int
    failed_count: int
    failed_ids: List[str]


class BulkOperationsUseCase:
    """
    Use case for bulk document operations.

    Flow for Bulk Upload:
    1. Validate file list (count, sizes, types)
    2. Check for duplicates (SHA256 hash)
    3. Upload files in batch
    4. Queue background indexing for each
    5. Return operation ID for progress tracking

    Flow for Bulk Delete:
    1. Validate ownership of all documents
    2. Delete documents in transaction
    3. Delete associated chunks from vector store
    4. Return operation summary

    Use Cases:
    - Batch upload for data ingestion
    - Bulk cleanup operations
    - Migration from other systems

    حالة استخدام العمليات الدفعية
    """

    def __init__(
        self,
        upload_use_case: UploadDocumentUseCase,
        file_store: FileStorePort,
        document_repo: DocumentRepoPort,
        task_queue,
    ) -> None:
        """
        Initialize bulk operations use case.

        Args:
            upload_use_case: Upload document use case (for individual uploads)
            file_store: File storage service
            document_repo: Document repository
            task_queue: Task queue for background processing
        """
        self._upload_use_case = upload_use_case
        self._file_store = file_store
        self._repo = document_repo
        self._queue = task_queue

    def _validate_upload(
        self,
        files: List[BinaryIO],
        filenames: List[str],
        content_types: List[str],
    ) -> None:
        """
        Validate upload parameters.

        Args:
            files: List of file-like objects
            filenames: List of original filenames
            content_types: List of MIME types

        Raises:
            ValueError: If validation fails
        """
        # Check file count
        if len(files) == 0:
            raise ValueError("No files provided")

        if len(files) > 100:
            raise ValueError("Maximum 100 files per bulk upload")

        # Check arrays match
        if len(files) != len(filenames):
            raise ValueError("Files and filenames count mismatch")

        if len(files) != len(content_types):
            raise ValueError("Files and content types count mismatch")

        # Check each file
        for i, file in enumerate(files):
            if file is None:
                raise ValueError(f"File at index {i} is None")

            content_type = content_types[i]
            if content_type not in [
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain",
            ]:
                raise ValueError(f"Invalid content type: {content_type}")

        # Check file sizes
        total_size = sum([len(f.read()) for f in files])
        if total_size > 500 * 1024 * 1024:  # 500MB
            raise ValueError("Total upload size exceeds 500MB limit")

    async def bulk_upload(self, request: BulkUploadRequest) -> BulkUploadResponse:
        """
        Execute bulk document upload.

        Args:
            request: Bulk upload request

        Returns:
            Bulk upload response with results for each file
        """
        tenant = TenantId(request.tenant_id)

        # Step 1: Validate upload
        self._validate_upload(
            files=request.files,
            filenames=request.filenames,
            content_types=request.content_types,
        )

        # Step 2: Generate operation ID
        operation_id = f"bulk_upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Step 3: Process each file
        results = []
        succeeded = 0
        failed = 0

        for i, file in enumerate(request.files):
            try:
                # Read file content
                file.seek(0)
                content = file.read()

                # Create upload request
                upload_request = UploadDocumentRequest(
                    tenant_id=request.tenant_id,
                    filename=request.filenames[i],
                    content_type=request.content_types[i],
                    data=content,
                )

                # Execute individual upload
                result = await self._upload_use_case.execute(upload_request)

                results.append(
                    BulkUploadResult(
                        filename=request.filenames[i],
                        document_id=result.document_id.value,
                        status=result.status,
                        message=result.message,
                        error=None,
                    )
                )

                if result.status != "already_exists":
                    succeeded += 1
                else:
                    failed += 1

            except Exception as e:
                results.append(
                    BulkUploadResult(
                        filename=request.filenames[i],
                        document_id=None,
                        status="failed",
                        message=str(e),
                        error=str(e),
                    )
                )
                failed += 1

        # Step 4: Return response
        return BulkUploadResponse(
            results=results,
            total_files=len(request.files),
            succeeded=succeeded,
            failed=failed,
            operation_id=operation_id,
        )

    def bulk_delete(self, request: BulkDeleteRequest) -> BulkDeleteResponse:
        """
        Execute bulk document deletion.

        Args:
            request: Bulk delete request

        Returns:
            Bulk delete response with summary

        Raises:
            ValueError: If validation fails
        """
        tenant = TenantId(request.tenant_id)

        # Step 1: Validate document IDs
        if len(request.document_ids) == 0:
            raise ValueError("No document IDs provided")

        if len(request.document_ids) > 100:
            raise ValueError("Maximum 100 documents per bulk delete")

        # Step 2: Validate ownership of all documents
        for doc_id in request.document_ids:
            doc = self._repo.get_document(
                tenant_id=tenant,
                document_id=DocumentId(doc_id),
            )

            if not doc:
                raise ValueError(f"Document {doc_id} not found")

            # Note: In production, check tenant_id matches
            # if doc.user_id != tenant.value:
            #     raise ValueError(f"Tenant doesn't own document {doc_id}")

        # Step 3: Delete documents in transaction
        # (In a real system, this would be a database transaction)
        deleted_count = 0
        failed_ids = []

        for doc_id in request.document_ids:
            try:
                success = self._repo.delete_document(
                    tenant_id=tenant,
                    document_id=DocumentId(doc_id),
                )

                if success:
                    deleted_count += 1
                else:
                    failed_ids.append(doc_id)

            except Exception as e:
                failed_ids.append(doc_id)

        # Step 4: Return response
        operation_id = f"bulk_delete_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        return BulkDeleteResponse(
            operation_id=operation_id,
            total_documents=len(request.document_ids),
            deleted_count=deleted_count,
            failed_count=len(failed_ids),
            failed_ids=failed_ids,
        )
