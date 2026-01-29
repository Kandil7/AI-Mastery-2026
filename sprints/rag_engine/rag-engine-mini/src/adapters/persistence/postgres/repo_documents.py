"""
Document Repository Implementation
===================================
PostgreSQL implementation of DocumentRepoPort.

تنفيذ مستودع المستندات بـ PostgreSQL
"""

import uuid
from typing import Sequence

from sqlalchemy import select, update, delete

from src.adapters.persistence.postgres.db import SessionLocal
from src.adapters.persistence.postgres.models import Document
from src.domain.entities import DocumentId, DocumentStatus, StoredFile, TenantId


class PostgresDocumentRepo:
    """
    PostgreSQL implementation of DocumentRepoPort.
    
    تنفيذ PostgreSQL لمنفذ مستودع المستندات
    """
    
    def create_document(
        self,
        *,
        tenant_id: TenantId,
        stored_file: StoredFile,
        file_sha256: str | None = None,
    ) -> DocumentId:
        """Create a new document record."""
        doc_id = DocumentId(str(uuid.uuid4()))
        
        with SessionLocal() as db:
            db.add(
                Document(
                    id=doc_id.value,
                    user_id=tenant_id.value,
                    filename=stored_file.filename,
                    content_type=stored_file.content_type,
                    file_path=stored_file.path,
                    size_bytes=stored_file.size_bytes,
                    file_sha256=file_sha256,
                    status="created",
                )
            )
            db.commit()
        
        return doc_id
    
    def set_status(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update document processing status."""
        with SessionLocal() as db:
            stmt = (
                update(Document)
                .where(
                    Document.id == document_id.value,
                    Document.user_id == tenant_id.value,
                )
                .values(status=status, error=error)
            )
            db.execute(stmt)
            db.commit()
    
    def get_status(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> DocumentStatus | None:
        """Get document status."""
        with SessionLocal() as db:
            stmt = select(Document).where(
                Document.id == document_id.value,
                Document.user_id == tenant_id.value,
            )
            doc = db.execute(stmt).scalar_one_or_none()
            
            if not doc:
                return None
            
            return DocumentStatus(
                document_id=DocumentId(doc.id),
                tenant_id=TenantId(doc.user_id),
                filename=doc.filename,
                status=doc.status,
                error=doc.error,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
    
    def list_documents(
        self,
        *,
        tenant_id: TenantId,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[DocumentStatus]:
        """List documents for a tenant."""
        with SessionLocal() as db:
            stmt = (
                select(Document)
                .where(Document.user_id == tenant_id.value)
                .order_by(Document.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            docs = db.execute(stmt).scalars().all()
            
            return [
                DocumentStatus(
                    document_id=DocumentId(doc.id),
                    tenant_id=TenantId(doc.user_id),
                    filename=doc.filename,
                    status=doc.status,
                    error=doc.error,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
                for doc in docs
            ]
    
    def delete_document(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> bool:
        """Delete a document."""
        with SessionLocal() as db:
            stmt = delete(Document).where(
                Document.id == document_id.value,
                Document.user_id == tenant_id.value,
            )
            result = db.execute(stmt)
            db.commit()
            return result.rowcount > 0
    
    def get_stored_file(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> StoredFile | None:
        """Get stored file info for document."""
        with SessionLocal() as db:
            stmt = select(Document).where(
                Document.id == document_id.value,
                Document.user_id == tenant_id.value,
            )
            doc = db.execute(stmt).scalar_one_or_none()
            
            if not doc:
                return None
            
            return StoredFile(
                path=doc.file_path,
                filename=doc.filename,
                content_type=doc.content_type,
                size_bytes=doc.size_bytes,
            )
