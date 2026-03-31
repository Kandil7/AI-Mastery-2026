from __future__ import annotations

from fastapi import APIRouter

from app.schemas import IngestRequest, IngestResponse
from rag.ingestion import ingest_document

router = APIRouter(prefix="/ingest")


@router.post("")
def ingest(request: IngestRequest) -> IngestResponse:
    ingestion_id = ingest_document(
        tenant_id=request.tenant_id,
        source_type=request.source_type,
        uri=request.uri,
        metadata=request.metadata,
    )
    return IngestResponse(ingestion_id=ingestion_id, status="accepted")
