"""
Ask Routes
===========
Endpoint for answering questions via RAG pipeline.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container
from src.application.use_cases.ask_question_hybrid import AskHybridRequest, AskQuestionHybridUseCase

router = APIRouter(prefix="/api/v1", tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    document_id: str | None = None
    k: int | None = Field(default=None, ge=1, le=200, description="Top-K override")
    k_vec: int | None = Field(default=None, ge=1, le=200)
    k_kw: int | None = Field(default=None, ge=1, le=200)
    fused_limit: int | None = Field(default=None, ge=1, le=200)
    rerank_top_n: int | None = Field(default=None, ge=1, le=50)
    expand_query: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> AskResponse:
    container = get_container()
    use_case: AskQuestionHybridUseCase | None = container.get("ask_hybrid_use_case")
    if not use_case:
        raise HTTPException(status_code=501, detail="Ask use case not configured")

    k_override = request.k
    ask_request = AskHybridRequest(
        tenant_id=tenant_id,
        question=request.question,
        document_id=request.document_id,
        k_vec=request.k_vec or k_override or 30,
        k_kw=request.k_kw or k_override or 30,
        fused_limit=request.fused_limit or 40,
        rerank_top_n=request.rerank_top_n or 8,
        expand_query=request.expand_query,
    )

    answer = use_case.execute(ask_request)
    return AskResponse(answer=answer.text, sources=list(answer.sources))
