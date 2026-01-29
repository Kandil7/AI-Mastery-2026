"""
Ask Question Hybrid Use Case
=============================
RAG with hybrid search (vector + keyword) and reranking.

حالة استخدام السؤال مع البحث الهجين وإعادة الترتيب
"""

from dataclasses import dataclass

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


class AskQuestionHybridUseCase:
    """
    Use case for answering questions with hybrid retrieval.
    
    Flow:
    1. Embed the question (cached)
    2. Vector search (Qdrant) - semantic similarity
    3. Hydrate vector results (get text from Postgres)
    4. Keyword search (Postgres FTS) - lexical match
    5. RRF fusion - merge results
    6. Rerank (Cross-Encoder) - precision boost
    7. Build prompt with guardrails
    8. Generate answer (LLM)
    9. Return answer with sources
    
    Design Decision: Hybrid search for better recall:
    - Vector catches semantic similarity
    - Keyword catches exact matches (names, numbers, dates)
    - RRF fusion combines without score calibration
    - Reranking improves precision significantly
    
    قرار التصميم: البحث الهجين لاستدعاء أفضل
    
    Example:
        >>> uc = AskQuestionHybridUseCase(...)
        >>> answer = uc.execute(AskHybridRequest(
        ...     tenant_id="user123",
        ...     question="What are the main features?",
        ...     k_vec=30,
        ...     rerank_top_n=8,
        ... ))
        >>> answer.text  # Generated answer
        >>> answer.sources  # Chunk IDs used
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
    ) -> None:
        """
        Initialize with all required ports.
        
        All dependencies are injected through ports.
        كل التبعيات محقونة من خلال المنافذ
        """
        self._embeddings = cached_embeddings
        self._vector = vector_store
        self._keyword = keyword_store
        self._text_reader = chunk_text_reader
        self._reranker = reranker
        self._llm = llm
    
    def execute(self, request: AskHybridRequest) -> Answer:
        """
        Execute the hybrid RAG pipeline.
        
        Args:
            request: Question request with parameters
            
        Returns:
            Answer with text and source chunk IDs
        """
        tenant = TenantId(request.tenant_id)
        
        # Step 1: Embed question (cached)
        question_vector = self._embeddings.embed_one(request.question)
        
        # Step 2: Vector search
        vector_results = self._vector.search_scored(
            query_vector=question_vector,
            tenant_id=tenant,
            top_k=request.k_vec,
            document_id=request.document_id,
        )
        
        # Convert to Chunk objects (no text yet)
        vector_chunks = [
            Chunk(
                id=r.chunk_id,
                tenant_id=tenant,
                document_id=DocumentId(r.document_id),
                text="",  # Will be hydrated
            )
            for r in vector_results
        ]
        
        # Step 3: Hydrate vector results (get text from DB)
        hydrated_vec_chunks = hydrate_chunk_texts(
            tenant_id=tenant,
            chunks=vector_chunks,
            reader=self._text_reader,
        )
        
        # Build ScoredChunk list for fusion
        vector_hits = [
            ScoredChunk(chunk=c, score=r.score)
            for c, r in zip(hydrated_vec_chunks, vector_results)
        ]
        
        # Step 4: Keyword search (already has text)
        keyword_chunks = self._keyword.search(
            query=request.question,
            tenant_id=tenant,
            top_k=request.k_kw,
            document_id=request.document_id,
        )
        
        # Convert to ScoredChunk (use rank-based scores for RRF)
        keyword_hits = [
            ScoredChunk(chunk=c, score=1.0)  # Score doesn't matter for RRF
            for c in keyword_chunks
        ]
        
        # Step 5: RRF Fusion
        fused = rrf_fusion(
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
            out_limit=request.fused_limit,
        )
        
        fused_chunks = [s.chunk for s in fused]
        
        # Step 6: Rerank
        reranked = self._reranker.rerank(
            query=request.question,
            chunks=fused_chunks,
            top_n=request.rerank_top_n,
        )
        
        # Step 7: Build prompt
        prompt = build_rag_prompt(
            question=request.question,
            chunks=reranked,
        )
        
        # Step 8: Generate answer
        answer_text = self._llm.generate(
            prompt,
            temperature=0.2,
            max_tokens=700,
        )
        
        # Step 9: Return with sources
        sources = [c.id for c in reranked]
        
        return Answer(text=answer_text, sources=sources)
