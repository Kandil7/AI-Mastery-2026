from __future__ import annotations

from fastapi import APIRouter

from app.schemas import QueryRequest, QueryResponse
from pipelines.online_query import run_query_pipeline

router = APIRouter(prefix="/query")


@router.post("")
def query(request: QueryRequest) -> QueryResponse:
    result = run_query_pipeline(
        tenant_id=request.tenant_id,
        question=request.question,
        filters=request.filters,
        top_k=request.top_k,
        mode=request.mode,
    )
    return QueryResponse(**result)
