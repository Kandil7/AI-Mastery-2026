"""
Export API Routes

This module implements the export functionality API endpoints that allow users
to download their data in various formats (PDF, Markdown, CSV, JSON).
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import json
import csv
import io
from datetime import datetime

from ...domain.entities import TenantId
from ...application.services.export_service import ExportService
from ..dependencies import get_container, get_tenant_id
from ...application.use_cases.export_use_case import (
    ExportFormat,
    ExportRequest,
    ExportResponse
)

router = APIRouter(prefix="/exports", tags=["exports"])


@router.post("", response_model=ExportResponse)
async def create_export(
    request: ExportRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Create a new export job.
    
    This endpoint creates an asynchronous export job for the specified format
    and content type. The actual export is performed in the background.
    """
    try:
        container = get_container()
        export_service = container.resolve(ExportService)
        
        result = await export_service.create_export_job(
            tenant_id=TenantId(tenant_id),
            format=request.format,
            content_type=request.content_type,
            filters=request.filters
        )
        
        return ExportResponse(
            job_id=result.job_id,
            status=result.status,
            format=result.format,
            content_type=result.content_type,
            created_at=result.created_at
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create export job: {str(e)}")


@router.get("/{job_id}")
async def get_export_status(
    job_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Get the status of an export job.
    
    Returns the current status of the export job (pending, processing, completed, failed).
    """
    try:
        container = get_container()
        export_service = container.resolve(ExportService)
        
        status = await export_service.get_export_status(
            job_id=job_id,
            tenant_id=TenantId(tenant_id)
        )
        
        if not status:
            raise HTTPException(status_code=404, detail="Export job not found")
        
        return {
            "job_id": status.job_id,
            "status": status.status,
            "format": status.format,
            "content_type": status.content_type,
            "download_url": status.download_url,
            "created_at": status.created_at,
            "completed_at": status.completed_at,
            "file_size": status.file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get export status: {str(e)}")


@router.get("/{job_id}/download")
async def download_export(
    job_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Download the exported file.
    
    This endpoint returns the exported file when the job is completed.
    """
    try:
        container = get_container()
        export_service = container.resolve(ExportService)
        
        export_result = await export_service.get_export_result(
            job_id=job_id,
            tenant_id=TenantId(tenant_id)
        )
        
        if not export_result or export_result.status != "completed":
            raise HTTPException(
                status_code=404, 
                detail="Export not found or not completed"
            )
        
        # Return the file content with appropriate headers
        content_type_map = {
            ExportFormat.PDF: "application/pdf",
            ExportFormat.MARKDOWN: "text/markdown",
            ExportFormat.CSV: "text/csv",
            ExportFormat.JSON: "application/json"
        }
        
        headers = {
            "Content-Disposition": f"attachment; filename={export_result.filename}",
            "Content-Type": content_type_map.get(export_result.format, "application/octet-stream")
        }
        
        return export_result.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download export: {str(e)}")


@router.get("/formats")
async def get_supported_formats():
    """
    Get supported export formats.
    
    Returns a list of all supported export formats.
    """
    formats = [fmt.value for fmt in ExportFormat]
    return {"formats": formats}


@router.post("/preview")
async def preview_export(
    request: ExportRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Preview export content without creating a job.
    
    This endpoint returns a preview of the export content without
    creating a full export job. Useful for validating exports.
    """
    try:
        container = get_container()
        export_service = container.resolve(ExportService)
        
        preview = await export_service.preview_export(
            tenant_id=TenantId(tenant_id),
            format=request.format,
            content_type=request.content_type,
            filters=request.filters,
            limit=5  # Only preview first 5 items
        )
        
        return {
            "format": request.format,
            "content_type": request.content_type,
            "preview": preview,
            "estimated_size": len(preview) if preview else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")


# Legacy endpoints for direct format exports (maintaining compatibility)
@router.post("/pdf")
async def export_to_pdf(
    request: Dict[str, Any],
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Direct PDF export endpoint (legacy compatibility).
    
    Exports data to PDF format directly without background processing.
    """
    try:
        container = get_container()
        export_service = container.resolve(ExportService)
        
        content = request.get("content", "")
        filename = request.get("filename", f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        
        pdf_content = await export_service.export_to_pdf(
            tenant_id=TenantId(tenant_id),
            content=content,
            filename=filename
        )
        
        return {
            "filename": filename,
            "size": len(pdf_content),
            "format": "pdf",
            "download_url": f"/exports/pdf/download?filename={filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export to PDF: {str(e)}")


@router.post("/markdown")
async def export_to_markdown(
    request: Dict[str, Any],
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Direct Markdown export endpoint (legacy compatibility).
    
    Exports data to Markdown format directly without background processing.
    """
    try:
        container = get_container()
        export_service = container.resolve(ExportService)
        
        content = request.get("content", "")
        filename = request.get("filename", f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        md_content = await export_service.export_to_markdown(
            tenant_id=TenantId(tenant_id),
            content=content,
            filename=filename
        )
        
        return {
            "filename": filename,
            "size": len(md_content),
            "format": "markdown",
            "download_url": f"/exports/markdown/download?filename={filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export to Markdown: {str(e)}")


@router.post("/csv")
async def export_to_csv(
    request: Dict[str, Any],
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Direct CSV export endpoint (legacy compatibility).
    
    Exports data to CSV format directly without background processing.
    """
    try:
        container = get_container()
        export_service = container.resolve(ExportService)
        
        rows = request.get("rows", [])
        filename = request.get("filename", f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        csv_content = await export_service.export_to_csv(
            tenant_id=TenantId(tenant_id),
            rows=rows,
            filename=filename
        )
        
        return {
            "filename": filename,
            "size": len(csv_content),
            "format": "csv",
            "download_url": f"/exports/csv/download?filename={filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export to CSV: {str(e)}")


@router.post("/json")
async def export_to_json(
    request: Dict[str, Any],
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Direct JSON export endpoint (legacy compatibility).
    
    Exports data to JSON format directly without background processing.
    """
    try:
        container = get_container()
        export_service = container.resolve(ExportService)
        
        data = request.get("data", {})
        filename = request.get("filename", f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        json_content = await export_service.export_to_json(
            tenant_id=TenantId(tenant_id),
            data=data,
            filename=filename
        )
        
        return {
            "filename": filename,
            "size": len(json_content),
            "format": "json",
            "download_url": f"/exports/json/download?filename={filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export to JSON: {str(e)}")


# Helper endpoints for direct downloads (for legacy compatibility)
@router.get("/pdf/download")
async def download_pdf(filename: str, tenant_id: str = Depends(get_tenant_id)):
    """Download a PDF export file."""
    # Implementation would serve the file from storage
    # This is a placeholder for the actual implementation
    return {"message": f"PDF download endpoint for {filename}"}


@router.get("/markdown/download")
async def download_markdown(filename: str, tenant_id: str = Depends(get_tenant_id)):
    """Download a Markdown export file."""
    # Implementation would serve the file from storage
    # This is a placeholder for the actual implementation
    return {"message": f"Markdown download endpoint for {filename}"}


@router.get("/csv/download")
async def download_csv(filename: str, tenant_id: str = Depends(get_tenant_id)):
    """Download a CSV export file."""
    # Implementation would serve the file from storage
    # This is a placeholder for the actual implementation
    return {"message": f"CSV download endpoint for {filename}"}


@router.get("/json/download")
async def download_json(filename: str, tenant_id: str = Depends(get_tenant_id)):
    """Download a JSON export file."""
    # Implementation would serve the file from storage
    # This is a placeholder for the actual implementation
    return {"message": f"JSON download endpoint for {filename}"}


__all__ = ["router"]