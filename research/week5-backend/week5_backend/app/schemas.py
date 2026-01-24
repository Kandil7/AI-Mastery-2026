from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    tenant_id: str
    source_type: str
    uri: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    ingestion_id: str
    status: str


class QueryRequest(BaseModel):
    tenant_id: str
    question: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 8
    mode: str = "rag"


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    trace_id: str
    model: str


class EvalRequest(BaseModel):
    dataset_id: str
    mode: str = "rag"


class EvalResponse(BaseModel):
    run_id: str
    status: str


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
