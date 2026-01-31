"""
Tests for Export Service

This module tests the export service functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.application.services.export_service import (
    ExportService,
    ExportFormat,
    ExportRequest,
    ExportResponse
)
from src.domain.entities import TenantId


@pytest.fixture
def mock_document_repo():
    """Mock document repository for testing."""
    repo = AsyncMock()
    repo.list_documents.return_value = [
        MagicMock(
            filename="test_doc.pdf",
            status="indexed",
            chunks_count=10,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]
    return repo


@pytest.fixture
def export_service(mock_document_repo):
    """Create an export service instance for testing."""
    return ExportService(
        document_repo=mock_document_repo
    )


@pytest.mark.asyncio
async def test_create_export_job(export_service, mock_document_repo):
    """Test creating an export job."""
    tenant_id = TenantId("test-tenant")
    
    job = await export_service.create_export_job(
        tenant_id=tenant_id,
        format=ExportFormat.JSON,
        content_type="documents",
        filters=None
    )
    
    # Verify the job was created with correct properties
    assert job.tenant_id == tenant_id
    assert job.format == ExportFormat.JSON
    assert job.content_type == "documents"
    assert job.status in ["completed", "processing"]  # Depending on implementation
    assert job.job_id is not None


@pytest.mark.asyncio
async def test_get_export_status(export_service, mock_document_repo):
    """Test getting export job status."""
    tenant_id = TenantId("test-tenant")
    
    # Create a job first
    job = await export_service.create_export_job(
        tenant_id=tenant_id,
        format=ExportFormat.PDF,
        content_type="documents",
        filters=None
    )
    
    # Get status
    status = await export_service.get_export_status(
        job_id=job.job_id,
        tenant_id=tenant_id
    )
    
    assert status is not None
    assert status.job_id == job.job_id
    assert status.tenant_id == tenant_id


@pytest.mark.asyncio
async def test_preview_export(export_service, mock_document_repo):
    """Test export preview functionality."""
    tenant_id = TenantId("test-tenant")
    
    preview = await export_service.preview_export(
        tenant_id=tenant_id,
        format=ExportFormat.MARKDOWN,
        content_type="documents",
        filters=None,
        limit=1
    )
    
    assert preview is not None
    assert isinstance(preview, str)
    assert "RAG Engine" in preview or "documents" in preview.lower()


@pytest.mark.asyncio
async def test_export_to_json(export_service, mock_document_repo):
    """Test JSON export functionality."""
    tenant_id = TenantId("test-tenant")
    
    data = {"test": "data", "number": 42}
    filename = "test.json"
    
    result = await export_service.export_to_json(
        tenant_id=tenant_id,
        data=data,
        filename=filename
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert "test" in result
    assert "data" in result


@pytest.mark.asyncio
async def test_export_to_csv(export_service, mock_document_repo):
    """Test CSV export functionality."""
    tenant_id = TenantId("test-tenant")
    
    rows = [{"col1": "value1", "col2": "value2"}]
    filename = "test.csv"
    
    result = await export_service.export_to_csv(
        tenant_id=tenant_id,
        rows=rows,
        filename=filename
    )
    
    assert result is not None
    assert "col1,col2" in result
    assert "value1,value2" in result


@pytest.mark.asyncio
async def test_export_to_markdown(export_service, mock_document_repo):
    """Test Markdown export functionality."""
    tenant_id = TenantId("test-tenant")
    
    content = "# Test Content\n\nThis is a test."
    filename = "test.md"
    
    result = await export_service.export_to_markdown(
        tenant_id=tenant_id,
        content=content,
        filename=filename
    )
    
    assert result is not None
    assert "# Test Content" in result


@pytest.mark.asyncio
async def test_export_to_pdf(export_service, mock_document_repo):
    """Test PDF export functionality."""
    tenant_id = TenantId("test-tenant")
    
    content = "Test PDF Content"
    filename = "test.pdf"
    
    result = await export_service.export_to_pdf(
        tenant_id=tenant_id,
        content=content,
        filename=filename
    )
    
    assert result is not None
    assert isinstance(result, bytes)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_different_export_formats(export_service, mock_document_repo):
    """Test that different export formats work correctly."""
    tenant_id = TenantId("test-tenant")
    
    # Test all formats
    formats_to_test = [
        ExportFormat.PDF,
        ExportFormat.MARKDOWN,
        ExportFormat.CSV,
        ExportFormat.JSON
    ]
    
    for export_format in formats_to_test:
        job = await export_service.create_export_job(
            tenant_id=tenant_id,
            format=export_format,
            content_type="documents",
            filters=None
        )
        
        assert job.format == export_format