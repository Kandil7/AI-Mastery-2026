"""
Ask Question Hybrid Use Case
=============================
RAG with hybrid search (vector + keyword) and reranking.

حالة استخدام السؤال مع البحث الهجين وإعادة الترتيب
"""

from dataclasses import dataclass
from typing import Sequence

from src.application.ports.chunk_text_reader import ChunkTextReaderPort
from src.application.ports.keyword_store import KeywordStorePort
from src.application.ports.llm import LLMPort
from src.application.ports.reranker import RerankerPort
from src.application.ports.vector_store import VectorStorePort
from src.application.services.embedding_cache import CachedEmbeddings
from src.application.services.fusion import rrf_fusion
from src.application.services.hydrate import hydrate_chunk_texts
from src.application.services.prompt_builder import build_rag_prompt
from src.application.services.scoring import ScoredChunk
from src.domain.entities import Answer, Chunk, DocumentId, TenantId


@dataclass
class AskHybridRequest:
    """Request data for hybrid RAG question."""
    tenant_id: str
    question: str
    
    # Optional: restrict to single document (ChatPDF mode)
    document_id: str | None = None
    
    # Retrieval parameters
    k_vec: int = 30       # Top-K for vector search
    k_kw: int = 30        # Top-K for keyword search
    fused_limit: int = 40 # Max after fusion
    rerank_top_n: int = 8 # Final top results after reranking
    
    # Advanced features
    expand_query: bool = False  # Use LLM to expand query


class AskQuestionHybridUseCase:
    """
    Use case for answering questions with hybrid retrieval.
    
    Flow:
    1. Expand query (optional)
    2. Embed the question (cached)
    3. Vector search (Qdrant) - semantic similarity
    4. Hydrate vector results (get text from Postgres)
    5. Keyword search (Postgres FTS) - lexical match
    6. RRF fusion - merge results
    7. Rerank (Cross-Encoder) - precision boost
    8. Build prompt with guardrails
    9. Generate answer (LLM)
    10. Return answer with sources
    """
    
    def __init__(
        self,
        *,
        cached_embeddings: CachedEmbeddings,
        vector_store: VectorStorePort,
        keyword_store: KeywordStorePort,
        chunk_text_reader: ChunkTextReaderPort,
        reranker: RerankerPort,
        llm: LLMPort,
        query_expansion_service: object | None = None,
        self_critique: object | None = None,
        router: object | None = None,
        privacy: object | None = None,
        search_tool: object | None = None,
    ) -> None:
        """
        Initialize with all required ports and Stage 5 services.
        """
        self._embeddings = cached_embeddings
        self._vector = vector_store
        self._keyword = keyword_store
        self._text_reader = chunk_text_reader
        self._reranker = reranker
        self._llm = llm
        self._expansion = query_expansion_service
        self._critique = self_critique
        self._router = router
        self._privacy = privacy
        self._search = search_tool
    
    def execute(self, request: AskHybridRequest) -> Answer:
        """
        Execute the hybrid RAG pipeline with Stage 5 Autonomy (Routing & Privacy).
        """
        # Step 0: Privacy Guard (Redaction)
        original_question = request.question
        if self._privacy:
            request.question = self._privacy.redact(request.question) # type: ignore
        
        # Step 1: Semantic Routing
        if self._router:
            from src.application.services.semantic_router import QueryIntent
            intent = self._router.route(request.question) # type: ignore
            
            if intent == QueryIntent.CHITCHAT:
                answer_text = self._llm.generate(f"Respond politely to: {request.question}")
                return Answer(text=answer_text, sources=[])
        
        # Step 2: Initial Retrieval
        reranked = list(self.execute_retrieval_only(
            tenant_id=TenantId(request.tenant_id),
            question=request.question,
            document_id=request.document_id,
            k_vec=request.k_vec,
            k_kw=request.k_kw,
            fused_limit=request.fused_limit,
            rerank_top_n=request.rerank_top_n,
            expand_query=request.expand_query,
        ))
        
        # Step 3: Grade Retrieval & Web Fallback (Stage 5)
        if self._critique:
            grade = self._critique.grade_retrieval(request.question, reranked) # type: ignore
            
            # Fallback to Web Search if irrelevant and we have a tool
            if grade == "irrelevant" and self._search:
                log.info("triggering_web_search", query=request.question)
                web_results = self._search.search(request.question) # type: ignore
                
                # Convert web results to Source Chunks
                web_chunks = [
                    Chunk(id=f"web_{i}", tenant_id=TenantId("web"), document_id=None, 
                          text=f"[{r['title']}]({r['url']})\n{r['content']}")
                    for i, r in enumerate(web_results)
                ]
                reranked = web_chunks if web_chunks else reranked

            elif grade == "irrelevant" and not request.expand_query:
                # Try expansion if no web search available
                reranked = list(self.execute_retrieval_only(
                    tenant_id=TenantId(request.tenant_id),
                    question=request.question,
                    document_id=request.document_id,
                    k_vec=request.k_vec,
                    k_kw=request.k_kw,
                    fused_limit=request.fused_limit,
                    rerank_top_n=request.rerank_top_n,
                    expand_query=True,
                ))
        
        # Step 4: Generation & Hallucination Check
        prompt = build_rag_prompt(request.question, reranked)
        answer_text = self._llm.generate(prompt, temperature=0.1)
        
        if self._critique:
            grade = self._critique.grade_answer(request.question, answer_text, reranked) # type: ignore
            
            if grade == "hallucination":
                strict_prompt = prompt + "\n\nSTRICT: Answer ONLY using provided facts."
                answer_text = self._llm.generate(strict_prompt, temperature=0.0)
        
        # Step 5: Restore Privacy (De-redaction)
        if self._privacy:
            answer_text = self._privacy.restore(answer_text) # type: ignore
            self._privacy.clear() # type: ignore

        return Answer(
            text=answer_text,
            sources=[c.id for c in reranked],
        )

    def execute_retrieval_only(
        self,
        *,
        tenant_id: TenantId,
        question: str,
        document_id: str | None = None,
        k_vec: int = 30,
        k_kw: int = 30,
        fused_limit: int = 40,
        rerank_top_n: int = 8,
        expand_query: bool = False,
    ) -> Sequence[Chunk]:
        """
        Execute only the retrieval and ranking part of the pipeline.
        
        If expand_query is True, it generates related queries and searches for all.
        """
        # 1. Query Expansion
        queries = [question]
        if expand_query and self._expansion:
            # We cast to avoid type errors in this prototype, 
            # in real code define a protocol for QueryExpansionService
            queries = self._expansion.expand(question)  # type: ignore
            
        all_vector_hits: list[ScoredChunk] = []
        all_keyword_hits: list[ScoredChunk] = []
        
        for q in queries:
            # Step 1: Embed query (cached)
            question_vector = self._embeddings.embed_one(q)
            
            # Step 2: Vector search
            vector_results = self._vector.search_scored(
                query_vector=question_vector,
                tenant_id=tenant_id.value,
                top_k=k_vec,
                document_id=document_id,
            )
            
            # Convert to Chunk objects (no text yet)
            vector_chunks = [
                Chunk(
                    id=r.chunk_id,
                    tenant_id=tenant_id,
                    document_id=DocumentId(r.document_id),
                    text="",  # Will be hydrated
                )
                for r in vector_results
            ]
            
            # Step 3: Hydrate vector results (get text from DB)
            hydrated_vec_chunks = hydrate_chunk_texts(
                tenant_id=tenant_id,
                chunks=vector_chunks,
                reader=self._text_reader,
            )
            
            # Build ScoredChunk list
            all_vector_hits.extend([
                ScoredChunk(chunk=c, score=r.score)
                for c, r in zip(hydrated_vec_chunks, vector_results)
            ])
            
            # Step 4: Keyword search
            keyword_chunks = self._keyword.search(
                query=q,
                tenant_id=tenant_id,
                top_k=k_kw,
                document_id=document_id,
            )
            
            all_keyword_hits.extend([
                ScoredChunk(chunk=c, score=1.0)
                for c in keyword_chunks
            ])
        
        # 5. RRF Fusion (handles duplicates automatically)
        fused = rrf_fusion(
            vector_hits=all_vector_hits,
            keyword_hits=all_keyword_hits,
            out_limit=fused_limit,
        )
        
        fused_chunks = [s.chunk for s in fused]
        
        # 6. Rerank
        reranked = self._reranker.rerank(
            query=question,  # Original question for reranking
            chunks=fused_chunks,
            top_n=rerank_top_n,
        )
        
        return reranked

    def execute_stream(self, request: AskHybridRequest) -> any:  # Returns a generator
        """
        Execute streaming hybrid RAG pipeline.
        
        تنفيذ أنبوب RAG الهجين مع التدفق
        """
        # 1. Retrieve and rank chunks
        reranked = self.execute_retrieval_only(
            tenant_id=TenantId(request.tenant_id),
            question=request.question,
            document_id=request.document_id,
            k_vec=request.k_vec,
            k_kw=request.k_kw,
            fused_limit=request.fused_limit,
            rerank_top_n=request.rerank_top_n,
            expand_query=request.expand_query,
        )
        
        # 2. Build prompt
        prompt = build_rag_prompt(
            question=request.question,
            chunks=reranked,
        )
        
        # 3. Generate answer chunks
        yield from self._llm.generate_stream(
            prompt,
            temperature=0.2,
            max_tokens=700,
        )
