"""
Export Use Case

This module implements the use case for exporting data in various formats.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

from ...domain.entities import TenantId
from ..services.export_service import (
    ExportServicePort, 
    ExportFormat, 
    ExportJob, 
    ExportResult
)


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


class ExportUseCase:
    """Use case for handling export requests."""

    def __init__(self, export_service: ExportServicePort):
        """
        Initialize with the export service.
        """
        self._export_service = export_service

    async def create_export_job(
        self,
        tenant_id: TenantId,
        format: ExportFormat,
        content_type: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> ExportResponse:
        """
        Create a new export job.
        
        Args:
            tenant_id: The tenant requesting the export
            format: The format to export to (PDF, Markdown, CSV, JSON)
            content_type: The type of content to export (documents, conversations, etc.)
            filters: Optional filters to apply to the export
            
        Returns:
            ExportResponse containing job information
        """
        job = await self._export_service.create_export_job(
            tenant_id=tenant_id,
            format=format,
            content_type=content_type,
            filters=filters
        )
        
        return ExportResponse(
            job_id=job.job_id,
            status=job.status,
            format=job.format,
            content_type=job.content_type,
            created_at=job.created_at
        )

    async def get_export_status(
        self,
        job_id: str,
        tenant_id: TenantId
    ) -> Optional[ExportResponse]:
        """
        Get the status of an export job.
        
        Args:
            job_id: The ID of the export job
            tenant_id: The tenant that owns the job
            
        Returns:
            ExportResponse if job exists, None otherwise
        """
        job = await self._export_service.get_export_status(
            job_id=job_id,
            tenant_id=tenant_id
        )
        
        if not job:
            return None
            
        return ExportResponse(
            job_id=job.job_id,
            status=job.status,
            format=job.format,
            content_type=job.content_type,
            created_at=job.created_at
        )

    async def get_export_result(
        self,
        job_id: str,
        tenant_id: TenantId
    ) -> Optional[ExportResult]:
        """
        Get the result of an export job.
        
        Args:
            job_id: The ID of the export job
            tenant_id: The tenant that owns the job
            
        Returns:
            ExportResult if job exists and is completed, None otherwise
        """
        return await self._export_service.get_export_result(
            job_id=job_id,
            tenant_id=tenant_id
        )

    async def preview_export(
        self,
        tenant_id: TenantId,
        format: ExportFormat,
        content_type: str,
        filters: Optional[Dict[str, Any]],
        limit: int = 5
    ) -> Optional[str]:
        """
        Preview export content without creating a job.
        
        Args:
            tenant_id: The tenant requesting the preview
            format: The format to preview (PDF, Markdown, CSV, JSON)
            content_type: The type of content to preview
            filters: Filters to apply to the preview
            limit: Maximum number of items to include in preview
            
        Returns:
            String representation of preview content
        """
        return await self._export_service.preview_export(
            tenant_id=tenant_id,
            format=format,
            content_type=content_type,
            filters=filters,
            limit=limit
        )