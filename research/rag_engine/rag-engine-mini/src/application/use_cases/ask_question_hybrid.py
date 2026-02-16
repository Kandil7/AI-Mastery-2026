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
from src.core.observability import (
    API_REQUEST_LATENCY,
    TOKEN_USAGE,
    EMBEDDING_CACHE_HIT,
    RETRIEVAL_SCORE,
    RERANK_DURATION,
    QUERY_EXPANSION_COUNT,
)
from src.core.tracing import get_rag_tracer
from opentelemetry import trace


@dataclass
class AskHybridRequest:
    """Request data for hybrid RAG question."""

    tenant_id: str
    question: str

    # Optional: restrict to single document (ChatPDF mode)
    document_id: str | None = None

    # Retrieval parameters
    k_vec: int = 30  # Top-K for vector search
    k_kw: int = 30  # Top-K for keyword search
    fused_limit: int = 40  # Max after fusion
    rerank_top_n: int = 8  # Final top results after reranking

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
        self._rag_tracer = get_rag_tracer()

    def execute(self, request: AskHybridRequest) -> Answer:
        """
        Execute the hybrid RAG pipeline with Stage 5 Autonomy (Routing & Privacy).
        Metrics tracked: API latency, token usage, cache hits, retrieval scores.
        Tracing: Full distributed trace of pipeline execution.
        """
        import time
        import logging
        log = logging.getLogger(__name__)
        start_time = time.time()
        
        # Start main RAG pipeline span
        with self._rag_tracer.start_rag_pipeline(request.tenant_id, request.question):
            try:
                # Step 0: Privacy Guard (Redaction)
                original_question = request.question
                if self._privacy:
                    request.question = self._privacy.redact(request.question)
                
                # Step 1: Semantic Routing
                if self._router:
                    from src.application.services.semantic_router import QueryIntent
                    with trace.get_tracer("rag").start_as_current_span("semantic_routing"):
                        intent = self._router.route(request.question)
                        trace.get_current_span().set_attribute("intent", intent)
                    
                    if intent == QueryIntent.CHITCHAT:
                        answer_text = self._llm.generate(f"Respond politely to: {request.question}")
                        API_REQUEST_LATENCY.labels(method="ask", endpoint="/ask").observe(time.time() - start_time)
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
                    with trace.get_tracer("rag").start_as_current_span("retrieval_grading"):
                        grade = self._critique.grade_retrieval(request.question, reranked)
                        trace.get_current_span().set_attribute("grade", grade)
                    
                    # Fallback to Web Search if irrelevant and we have a tool
                    if grade == "irrelevant" and self._search:
                        log.info("triggering_web_search", query=request.question)
                        web_results = self._search.search(request.question)
                        
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
                
                with trace.get_tracer("rag").start_as_current_span("llm_generation"):
                    trace.get_current_span().set_attribute("prompt_length", len(prompt))
                    answer_text = self._llm.generate(prompt, temperature=0.1)
                    trace.get_current_span().set_attribute("answer_length", len(answer_text))
                
                # Track token usage (estimate based on text length)
                TOKEN_USAGE.labels(model="default", type="prompt").inc(len(prompt) // 4)
                TOKEN_USAGE.labels(model="default", type="completion").inc(len(answer_text) // 4)
                
                if self._critique:
                    with trace.get_tracer("rag").start_as_current_span("answer_grading"):
                        grade = self._critique.grade_answer(request.question, answer_text, reranked)
                        trace.get_current_span().set_attribute("grade", grade)
                    
                    if grade == "hallucination":
                        strict_prompt = prompt + "\n\nSTRICT: Answer ONLY using provided facts."
                        with trace.get_tracer("rag").start_as_current_span("llm_regeneration"):
                            answer_text = self._llm.generate(strict_prompt, temperature=0.0)
                            trace.get_current_span().set_attribute("mode", "strict")
                        TOKEN_USAGE.labels(model="default", type="completion").inc(len(answer_text) // 4)
                
                # Step 5: Restore Privacy (De-redaction)
                if self._privacy:
                    with trace.get_tracer("rag").start_as_current_span("privacy_restoration"):
                        answer_text = self._privacy.restore(answer_text)
                        self._privacy.clear()

                API_REQUEST_LATENCY.labels(method="ask", endpoint="/ask").observe(time.time() - start_time)
                trace.get_current_span().set_attribute("sources_count", len(reranked))
                
                return Answer(
                    text=answer_text,
                    sources=[c.id for c in reranked],
                )
            except Exception as e:
                API_REQUEST_LATENCY.labels(method="ask", endpoint="/ask").observe(time.time() - start_time)
                trace.get_current_span().record_exception(e)
                raise

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
        
        Metrics tracked: cache hits, retrieval scores (vector/keyword), rerank time.
        Tracing: Full distributed trace of retrieval operations.
        
        If expand_query is True, it generates related queries and searches for all.
        """
        import time
        import logging
        log = logging.getLogger(__name__)
        
        with trace.get_tracer("rag").start_as_current_span("retrieval_pipeline"):
            # 1. Query Expansion
            queries = [question]
            if expand_query and self._expansion:
                with trace.get_tracer("rag").start_as_current_span("query_expansion"):
                    queries = self._expansion.expand(question)
                    trace.get_current_span().set_attribute("original_query", question)
                    trace.get_current_span().set_attribute("expanded_queries_count", len(queries))
                QUERY_EXPANSION_COUNT.labels(expanded="true").inc()
            
            trace.get_current_span().set_attribute("total_queries", len(queries))
            
            all_vector_hits: list[ScoredChunk] = []
            all_keyword_hits: list[ScoredChunk] = []
            
            for i, q in enumerate(queries):
                with trace.get_tracer("rag").start_as_current_span(f"query_{i}"):
                    trace.get_current_span().set_attribute("query", q)
                    
                    # Step 1: Embed query (cached)
                    with self._rag_tracer.trace_embedding(q, cached=False) as embed_span:
                        question_vector = self._embeddings.embed_one(q)
                        
                        # Track cache hit/miss (assume embed_one has _was_cached attribute)
                        if hasattr(question_vector, '_was_cached'):
                            if question_vector._was_cached:
                                EMBEDDING_CACHE_HIT.labels(result="hit").inc()
                                embed_span.set_attribute("cached", "true")
                            else:
                                EMBEDDING_CACHE_HIT.labels(result="miss").inc()
                                embed_span.set_attribute("cached", "false")
                    
                    # Step 2: Vector search
                    with self._rag_tracer.trace_vector_search(tenant_id.value, k_vec, 0):
                        vector_results = self._vector.search_scored(
                            query_vector=question_vector,
                            tenant_id=tenant_id,
                            top_k=k_vec,
                            document_id=document_id,
                        )
                        trace.get_current_span().set_attribute("results_count", len(vector_results))
                    
                    # Track retrieval scores for vector search
                    for result in vector_results:
                        RETRIEVAL_SCORE.labels(method="vector").observe(result.score)
                    
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
                    with trace.get_tracer("rag").start_as_current_span("hydration"):
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
                    with self._rag_tracer.trace_keyword_search(tenant_id.value, q, 0):
                        keyword_chunks = self._keyword.search(
                            query=q,
                            tenant_id=tenant_id,
                            top_k=k_kw,
                            document_id=document_id,
                        )
                        trace.get_current_span().set_attribute("results_count", len(keyword_chunks))
                    
                    # Track keyword hits as perfect matches (score 1.0)
                    for _ in keyword_chunks:
                        RETRIEVAL_SCORE.labels(method="keyword").observe(1.0)
                    
                    all_keyword_hits.extend([
                        ScoredChunk(chunk=c, score=1.0)
                        for c in keyword_chunks
                    ])
            
            # 5. RRF Fusion (handles duplicates automatically)
            with trace.get_tracer("rag").start_as_current_span("rrf_fusion"):
                fused = rrf_fusion(
                    vector_hits=all_vector_hits,
                    keyword_hits=all_keyword_hits,
                    out_limit=fused_limit,
                )
                trace.get_current_span().set_attribute("vector_hits", len(all_vector_hits))
                trace.get_current_span().set_attribute("keyword_hits", len(all_keyword_hits))
                trace.get_current_span().set_attribute("fused_count", len(fused))
            
            fused_chunks = [s.chunk for s in fused]
            
            # 6. Rerank with timing
            with self._rag_tracer.trace_rerank(len(fused_chunks), rerank_top_n):
                rerank_start = time.time()
                reranked = self._reranker.rerank(
                    query=question,  # Original question for reranking
                    chunks=fused_chunks,
                    top_n=rerank_top_n,
                )
                rerank_time = time.time() - rerank_start
                RERANK_DURATION.labels(method="cross_encoder").observe(rerank_time)
                trace.get_current_span().set_attribute("rerank_time_ms", int(rerank_time * 1000))
            
            trace.get_current_span().set_attribute("final_results", len(list(reranked)))
            
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
