"""
Query Routes
=============
Endpoints for RAG question answering.

نقاط نهاية الأسئلة والأجوبة RAG
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container
from src.application.use_cases.ask_question_hybrid import (
    AskHybridRequest,
    AskQuestionHybridUseCase,
)
from src.domain.errors import DomainError

router = APIRouter(prefix="/api/v1/queries", tags=["queries"])


class AskHybridBody(BaseModel):
    """Request body for hybrid RAG question."""
    question: str = Field(..., min_length=2, max_length=2000)
    document_id: str | None = Field(default=None, description="Optional: restrict to single document")
    k_vec: int = Field(default=30, ge=1, le=200, description="Top-K for vector search")
    k_kw: int = Field(default=30, ge=1, le=200, description="Top-K for keyword search")
    rerank_top_n: int = Field(default=8, ge=1, le=50, description="Top-N after reranking")
    expand_query: bool = Field(default=False, description="Use LLM to expand query for better recall")


class AskResponse(BaseModel):
    """Response model for RAG answer."""
    answer: str
    sources: list[str]


@router.post("/ask-hybrid", response_model=AskResponse)
def ask_hybrid(
    body: AskHybridBody,
    tenant_id: str = Depends(get_tenant_id),
) -> AskResponse:
    """
    Ask a question using hybrid retrieval (vector + keyword).
    
    Flow:
    1. Embed question
    2. Vector search (semantic)
    3. Keyword search (lexical)
    4. RRF fusion
    5. Cross-Encoder reranking
    6. LLM answer generation
    
    Optional: Set document_id to restrict search to a single document (ChatPDF mode).
    
    طرح سؤال باستخدام الاسترجاع الهجين
    """
    container = get_container()
    use_case: AskQuestionHybridUseCase = container["ask_hybrid_use_case"]
    
    try:
        answer = use_case.execute(
            AskHybridRequest(
                tenant_id=tenant_id,
                question=body.question,
                document_id=body.document_id,
                k_vec=body.k_vec,
                k_kw=body.k_kw,
                rerank_top_n=body.rerank_top_n,
                expand_query=body.expand_query,
            )
        )
        
        return AskResponse(
            answer=answer.text,
            sources=list(answer.sources),
        )
        
    except DomainError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/ask-hybrid-stream")
async def ask_hybrid_stream(
    body: AskHybridBody,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Ask a question with streaming hybrid search.
    
    طرح سؤال مع الاستجابة المتدفقة
    """
    from fastapi.responses import StreamingResponse
    
    container = get_container()
    use_case: AskQuestionHybridUseCase = container["ask_hybrid_use_case"]
    
    try:
        def stream_generator():
            generator = use_case.execute_stream(
                AskHybridRequest(
                    tenant_id=tenant_id,
                    question=body.question,
                    document_id=body.document_id,
                    k_vec=body.k_vec,
                    k_kw=body.k_kw,
                    rerank_top_n=body.rerank_top_n,
                    expand_query=body.expand_query,
                )
            )
            for chunk in generator:
                yield chunk
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
        )
        
    except DomainError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/ask")
def ask_simple(
    body: AskHybridBody,
    tenant_id: str = Depends(get_tenant_id),
) -> AskResponse:
    """
    Alias for /ask-hybrid for convenience.
    """
    return ask_hybrid(body=body, tenant_id=tenant_id)


@router.get("/graph-search")
def graph_search(
    entity: str,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Query the Knowledge Graph for relationships related to an entity.
    
    البحث في رسم بياني المعرفة عن علاقات كيان معين
    """
    container = get_container()
    graph_repo = container["graph_repo"]
    
    try:
        triplets = graph_repo.get_triplets_by_entity(
            tenant_id=TenantId(tenant_id),
            entity_name=entity,
        )
        return {"entity": entity, "triplets": triplets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph search error: {e}")
