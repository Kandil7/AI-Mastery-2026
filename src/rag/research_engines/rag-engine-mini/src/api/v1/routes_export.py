"""
Export Routes (Legacy Compatibility)
===============================

This module provides legacy compatibility endpoints for export functionality.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/")
async def export_root():
    """
    Export root endpoint for legacy compatibility.
    
    This endpoint exists for backward compatibility.
    New implementations should use /exports endpoints.
    """
    return {
        "message": "Export API - Use /exports for new implementations",
        "available_formats": ["pdf", "markdown", "csv", "json"]
    }


@router.get("/status/{job_id}")
async def get_export_status_legacy(job_id: str):
    """
    Legacy endpoint for checking export status.
    
    This endpoint exists for backward compatibility.
    New implementations should use /exports/{job_id} endpoint.
    """
    return {
        "job_id": job_id,
        "status": "completed",  # Placeholder
        "message": "Use /exports/{job_id} for new implementations"
    }


__all__ = ["router"]
"""
Export Routes
=============
API endpoints for exporting documents and chat sessions.

نقاط نهاية API لتصدير المستندات وجلسات المحادثة
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container
from src.application.services.export_formats import (
    PDFExportService,
    MarkdownExportService,
    CSVExportService,
    JSONExportService,
)

router = APIRouter(tags=["export"])


class ExportRequest(BaseModel):
    """Request model for export operations."""

    document_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    format: str = "pdf"  # pdf, markdown, csv, json
    title: Optional[str] = None


@router.post("/export/documents")
async def export_documents(
    request: ExportRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> Response:
    """
    Export documents in specified format.

    Args:
        request: Export request with document IDs and format
        tenant_id: Tenant/user ID from auth

    Returns:
        File download with appropriate MIME type

    تصدير المستندات بالتنسيق المحدد
    """
    # Get repositories
    container = get_container()
    doc_repo = container.get("document_repo")

    # Get documents
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="document_ids required")

    documents = []
    for doc_id in request.document_ids:
        doc = doc_repo.find_by_id(doc_id)
        if doc:
            documents.append(
                {
                    "id": doc_id,
                    "filename": doc.get("filename", "unknown"),
                    "content_type": doc.get("content_type", "unknown"),
                    "size_bytes": doc.get("size_bytes", 0),
                    "status": doc.get("status", "unknown"),
                    "created_at": doc.get("created_at", "").isoformat()
                    if doc.get("created_at")
                    else None,
                }
            )

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")

    # Export based on format
    format_type = request.format.lower()

    if format_type == "pdf":
        service = PDFExportService()
        content = service.export_documents(documents, title=request.title or "Document Export")
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="documents.pdf"'},
        )

    elif format_type == "markdown":
        service = MarkdownExportService()
        content = service.export_documents(documents, title=request.title or "Document Export")
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="documents.md"'},
        )

    elif format_type == "csv":
        service = CSVExportService()
        content = service.export_documents(documents)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="documents.csv"'},
        )

    elif format_type == "json":
        service = JSONExportService()
        content = service.export_documents(documents)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="documents.json"'},
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format_type}. Use: pdf, markdown, csv, json",
        )


@router.post("/export/chat-sessions")
async def export_chat_sessions(
    request: ExportRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> Response:
    """
    Export chat sessions in specified format.

    Args:
        request: Export request with session IDs and format
        tenant_id: Tenant/user ID from auth

    Returns:
        File download with appropriate MIME type

    تصدير جلسات المحادثة بالتنسيق المحدد
    """
    # Get repositories
    container = get_container()
    chat_repo = container.get("chat_repo")
    query_repo = container.get("query_history_repo")

    # Get sessions
    if not request.session_ids:
        raise HTTPException(status_code=400, detail="session_ids required")

    sessions = []
    for session_id in request.session_ids:
        session = chat_repo.get_session(
            tenant_id=tenant_id,
            session_id=session_id,
        )
        if session:
            # Get turns
            turns = chat_repo.get_session_turns(
                tenant_id=tenant_id,
                session_id=session_id,
                limit=100,
            )

            sessions.append(
                {
                    "id": session_id,
                    "title": session.get("title", "Untitled"),
                    "created_at": session.get("created_at", "").isoformat()
                    if session.get("created_at")
                    else None,
                    "turn_count": len(turns),
                    "turns": [
                        {
                            "question": turn.question,
                            "answer": turn.answer,
                            "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
                        }
                        for turn in turns
                    ],
                }
            )

    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found")

    # Export based on format
    format_type = request.format.lower()

    if format_type == "pdf":
        service = PDFExportService()
        content = service.export_documents(sessions, title=request.title or "Chat Sessions Export")
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="chat_sessions.pdf"'},
        )

    elif format_type == "markdown":
        service = MarkdownExportService()
        content = service.export_documents(sessions, title=request.title or "Chat Sessions Export")
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="chat_sessions.md"'},
        )

    elif format_type == "csv":
        service = CSVExportService()
        content = service.export_documents(sessions)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="chat_sessions.csv"'},
        )

    elif format_type == "json":
        service = JSONExportService()
        content = service.export_documents(sessions)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="chat_sessions.json"'},
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format_type}. Use: pdf, markdown, csv, json",
        )


@router.post("/export/query-history")
async def export_query_history(
    format: str = "csv",
    tenant_id: str = Depends(get_tenant_id),
) -> Response:
    """
    Export query history in specified format.

    Args:
        format: Export format (pdf, markdown, csv, json)
        tenant_id: Tenant/user ID from auth

    Returns:
        File download with appropriate MIME type

    تصدير تاريخ الاستعلامات بالتنسيق المحدد
    """
    # Get repositories
    container = get_container()
    query_repo = container.get("query_history_repo")

    # Get query history
    history = query_repo.list_queries(
        tenant_id=tenant_id,
        limit=1000,
        offset=0,
    )

    if not history:
        raise HTTPException(status_code=404, detail="No query history found")

    # Format for export
    queries = [
        {
            "question": item.question,
            "answer": item.answer,
            "sources": ", ".join(item.sources),
            "timestamp": item.timestamp.isoformat() if item.timestamp else None,
        }
        for item in history
    ]

    # Export based on format
    format_type = format.lower()

    if format_type == "pdf":
        service = PDFExportService()
        content = service.export_documents(queries, title="Query History Export")
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="query_history.pdf"'},
        )

    elif format_type == "markdown":
        service = MarkdownExportService()
        content = service.export_documents(queries, title="Query History Export")
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="query_history.md"'},
        )

    elif format_type == "csv":
        service = CSVExportService()
        content = service.export_documents(queries)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="query_history.csv"'},
        )

    elif format_type == "json":
        service = JSONExportService()
        content = service.export_documents(queries)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="query_history.json"'},
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format_type}. Use: pdf, markdown, csv, json",
        )
