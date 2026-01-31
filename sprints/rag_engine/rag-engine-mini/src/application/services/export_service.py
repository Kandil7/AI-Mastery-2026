"""
Export Service

This module implements the export service that handles converting
data to various formats (PDF, Markdown, CSV, JSON) for user downloads.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import csv
import io
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from ...domain.entities import TenantId
from ...application.ports.repository_ports import DocumentRepositoryPort


class ExportFormat(str, Enum):
    """Supported export formats."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"


@dataclass
class ExportJob:
    """Represents an export job."""
    job_id: str
    tenant_id: TenantId
    format: ExportFormat
    content_type: str
    status: str  # pending, processing, completed, failed
    filters: Optional[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class ExportResult:
    """Result of an export operation."""
    filename: str
    format: ExportFormat
    content: bytes
    status: str
    file_size: int


@dataclass
class ExportRequest:
    """Request for creating an export job."""
    format: ExportFormat
    content_type: str  # documents, conversations, etc.
    filters: Optional[Dict[str, Any]] = None


@dataclass
class ExportResponse:
    """Response for export job creation."""
    job_id: str
    status: str
    format: ExportFormat
    content_type: str
    created_at: datetime


class ExportServicePort(ABC):
    """Abstract port for export services."""

    @abstractmethod
    async def create_export_job(
        self,
        tenant_id: TenantId,
        format: ExportFormat,
        content_type: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> ExportJob:
        """Create a new export job."""
        pass

    @abstractmethod
    async def get_export_status(
        self,
        job_id: str,
        tenant_id: TenantId
    ) -> Optional[ExportJob]:
        """Get the status of an export job."""
        pass

    @abstractmethod
    async def get_export_result(
        self,
        job_id: str,
        tenant_id: TenantId
    ) -> Optional[ExportResult]:
        """Get the result of an export job."""
        pass

    @abstractmethod
    async def preview_export(
        self,
        tenant_id: TenantId,
        format: ExportFormat,
        content_type: str,
        filters: Optional[Dict[str, Any]],
        limit: int = 5
    ) -> Optional[str]:
        """Preview export content without creating a job."""
        pass

    @abstractmethod
    async def export_to_pdf(
        self,
        tenant_id: TenantId,
        content: str,
        filename: str
    ) -> bytes:
        """Export content to PDF format."""
        pass

    @abstractmethod
    async def export_to_markdown(
        self,
        tenant_id: TenantId,
        content: str,
        filename: str
    ) -> str:
        """Export content to Markdown format."""
        pass

    @abstractmethod
    async def export_to_csv(
        self,
        tenant_id: TenantId,
        rows: List[Dict[str, Any]],
        filename: str
    ) -> str:
        """Export data to CSV format."""
        pass

    @abstractmethod
    async def export_to_json(
        self,
        tenant_id: TenantId,
        data: Any,
        filename: str
    ) -> str:
        """Export data to JSON format."""
        pass


class ExportService(ExportServicePort):
    """Concrete implementation of the export service."""

    def __init__(
        self,
        document_repo: DocumentRepositoryPort,
        storage_path: Optional[Path] = None
    ):
        self._document_repo = document_repo
        self._storage_path = storage_path or Path("./exports")
        self._storage_path.mkdir(exist_ok=True)
        self._active_jobs: Dict[str, ExportJob] = {}

    async def create_export_job(
        self,
        tenant_id: TenantId,
        format: ExportFormat,
        content_type: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> ExportJob:
        """Create a new export job."""
        import uuid
        
        job_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        job = ExportJob(
            job_id=job_id,
            tenant_id=tenant_id,
            format=format,
            content_type=content_type,
            status="pending",
            filters=filters,
            created_at=created_at
        )
        
        # Store the job temporarily (in production, this would go to a DB)
        self._active_jobs[job_id] = job
        
        # For now, process immediately for demo purposes
        # In production, this would be queued for background processing
        await self._process_export_job(job)
        
        return job

    async def _process_export_job(self, job: ExportJob):
        """Process an export job (simulated background processing)."""
        import time
        
        # Update status to processing
        job.status = "processing"
        
        # Simulate processing time
        time.sleep(1)
        
        # Generate export based on content type and format
        try:
            if job.content_type == "documents":
                # Get documents for tenant
                docs = self._document_repo.list_documents(
                    tenant_id=job.tenant_id,
                    limit=100,  # Limit for export
                    offset=0
                )
                
                if job.format == ExportFormat.PDF:
                    content = self._generate_document_pdf_content(docs)
                    export_result = await self.export_to_pdf(
                        tenant_id=job.tenant_id,
                        content=content,
                        filename=f"documents_{job.job_id}.pdf"
                    )
                elif job.format == ExportFormat.MARKDOWN:
                    content = self._generate_document_markdown_content(docs)
                    export_result = await self.export_to_markdown(
                        tenant_id=job.tenant_id,
                        content=content,
                        filename=f"documents_{job.job_id}.md"
                    )
                elif job.format == ExportFormat.CSV:
                    rows = self._generate_document_csv_rows(docs)
                    export_result = await self.export_to_csv(
                        tenant_id=job.tenant_id,
                        rows=rows,
                        filename=f"documents_{job.job_id}.csv"
                    )
                elif job.format == ExportFormat.JSON:
                    data = self._generate_document_json_data(docs)
                    export_result = await self.export_to_json(
                        tenant_id=job.tenant_id,
                        data=data,
                        filename=f"documents_{job.job_id}.json"
                    )
                
                # Mark as completed
                job.status = "completed"
                job.completed_at = datetime.now()
                job.file_size = len(export_result) if isinstance(export_result, (bytes, str)) else 0
                job.download_url = f"/exports/{job.job_id}/download"
            else:
                # Handle other content types
                job.status = "completed"
                job.completed_at = datetime.now()
                
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)

    def _generate_document_pdf_content(self, documents) -> str:
        """Generate content for PDF export."""
        content = "RAG Engine Documents Export\n\n"
        content += f"Generated on: {datetime.now().isoformat()}\n\n"
        
        for doc in documents:
            content += f"Title: {doc.filename}\n"
            content += f"Status: {doc.status}\n"
            content += f"Chunks: {doc.chunks_count}\n"
            content += f"Created: {doc.created_at.isoformat() if doc.created_at else 'N/A'}\n"
            content += "-" * 50 + "\n\n"
        
        return content

    def _generate_document_markdown_content(self, documents) -> str:
        """Generate content for Markdown export."""
        content = "# RAG Engine Documents Export\n\n"
        content += f"*Generated on: {datetime.now().isoformat()}*\n\n"
        
        for doc in documents:
            content += f"## {doc.filename}\n\n"
            content += f"- **Status**: {doc.status}\n"
            content += f"- **Chunks**: {doc.chunks_count}\n"
            content += f"- **Created**: {doc.created_at.isoformat() if doc.created_at else 'N/A'}\n\n"
            content += "---\n\n"
        
        return content

    def _generate_document_csv_rows(self, documents) -> List[Dict[str, str]]:
        """Generate rows for CSV export."""
        rows = []
        for doc in documents:
            rows.append({
                "filename": doc.filename,
                "status": doc.status.value if hasattr(doc.status, 'value') else str(doc.status),
                "chunks_count": str(doc.chunks_count),
                "created_at": doc.created_at.isoformat() if doc.created_at else "",
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else ""
            })
        return rows

    def _generate_document_json_data(self, documents) -> List[Dict[str, Any]]:
        """Generate data for JSON export."""
        data = []
        for doc in documents:
            data.append({
                "filename": doc.filename,
                "status": doc.status.value if hasattr(doc.status, 'value') else str(doc.status),
                "chunks_count": doc.chunks_count,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
            })
        return data

    async def get_export_status(
        self,
        job_id: str,
        tenant_id: TenantId
    ) -> Optional[ExportJob]:
        """Get the status of an export job."""
        job = self._active_jobs.get(job_id)
        if job and job.tenant_id == tenant_id:
            return job
        return None

    async def get_export_result(
        self,
        job_id: str,
        tenant_id: TenantId
    ) -> Optional[ExportResult]:
        """Get the result of an export job."""
        job = await self.get_export_status(job_id, tenant_id)
        if not job or job.status != "completed":
            return None
            
        # In a real implementation, this would retrieve the actual file
        # For now, we'll return a placeholder
        filename = f"{job.content_type}_{job.job_id}.{job.format}"
        
        # Generate content based on format
        if job.format == ExportFormat.PDF:
            content = await self.export_to_pdf(
                tenant_id=tenant_id,
                content="Sample content",
                filename=filename
            )
        elif job.format == ExportFormat.MARKDOWN:
            content = await self.export_to_markdown(
                tenant_id=tenant_id,
                content="# Sample content\n\nThis is a sample.",
                filename=filename
            )
        elif job.format == ExportFormat.CSV:
            content = await self.export_to_csv(
                tenant_id=tenant_id,
                rows=[{"col1": "value1", "col2": "value2"}],
                filename=filename
            )
        elif job.format == ExportFormat.JSON:
            content = await self.export_to_json(
                tenant_id=tenant_id,
                data={"sample": "data"},
                filename=filename
            )
        else:
            content = b""
            
        return ExportResult(
            filename=filename,
            format=job.format,
            content=content.encode() if isinstance(content, str) else content,
            status=job.status,
            file_size=len(content) if content else 0
        )

    async def preview_export(
        self,
        tenant_id: TenantId,
        format: ExportFormat,
        content_type: str,
        filters: Optional[Dict[str, Any]],
        limit: int = 5
    ) -> Optional[str]:
        """Preview export content without creating a job."""
        if content_type == "documents":
            docs = self._document_repo.list_documents(
                tenant_id=tenant_id,
                limit=limit,
                offset=0
            )
            
            if format == ExportFormat.MARKDOWN:
                return self._generate_document_markdown_content(docs[:limit])
            elif format == ExportFormat.JSON:
                data = self._generate_document_json_data(docs[:limit])
                return json.dumps(data, indent=2)
            else:
                # For other formats, return a text representation
                return f"Preview for {content_type} in {format} format with {len(docs[:limit])} items"
        
        return f"Preview for {content_type} in {format} format"

    async def export_to_pdf(
        self,
        tenant_id: TenantId,
        content: str,
        filename: str
    ) -> bytes:
        """Export content to PDF format."""
        buffer = io.BytesIO()
        
        # Create PDF
        pdf = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Get sample styles
        styles = getSampleStyleSheet()
        style = styles["Normal"]
        
        # Add content
        for line in content.split('\n'):
            elements.append(Paragraph(line, style))
            elements.append(Spacer(1, 12))
        
        pdf.build(elements)
        
        # Get the value and close buffer
        pdf_value = buffer.getvalue()
        buffer.close()
        
        return pdf_value

    async def export_to_markdown(
        self,
        tenant_id: TenantId,
        content: str,
        filename: str
    ) -> str:
        """Export content to Markdown format."""
        # For now, just return the content as-is
        # In a real implementation, we might want to format it differently
        return content

    async def export_to_csv(
        self,
        tenant_id: TenantId,
        rows: List[Dict[str, Any]],
        filename: str
    ) -> str:
        """Export data to CSV format."""
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        csv_content = output.getvalue()
        output.close()
        
        return csv_content

    async def export_to_json(
        self,
        tenant_id: TenantId,
        data: Any,
        filename: str
    ) -> str:
        """Export data to JSON format."""
        return json.dumps(data, indent=2, default=str)