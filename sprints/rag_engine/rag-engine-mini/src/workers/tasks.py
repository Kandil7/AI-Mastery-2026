"""
Celery Tasks - Document Indexing
=================================
Background task for document processing pipeline.

مهمة خلفية لخط أنابيب معالجة المستندات
"""

import hashlib

import structlog

from src.workers.celery_app import celery_app
from src.core.bootstrap import get_container
from src.domain.entities import TenantId, DocumentId, Chunk, ChunkSpec
from src.application.services.chunking import chunk_text_token_aware

log = structlog.get_logger()


def _chunk_hash(text: str) -> str:
    """Generate SHA256 hash of normalized text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@celery_app.task(
    name="index_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
)
def index_document(self, *, tenant_id: str, document_id: str) -> dict:
    """
    Index a document: extract → chunk → embed → store.
    
    Flow:
    1. Load document metadata
    2. Extract text from file
    3. Chunk text (token-aware)
    4. Deduplicate chunks (hash-based)
    5. Batch embed chunks
    6. Store in chunk_store (Postgres)
    7. Upsert to vector store (Qdrant)
    8. Update document status
    
    فهرسة مستند: استخراج → تقطيع → تضمين → تخزين
    """
    container = get_container()
    
    # Get dependencies from container
    document_repo = container["document_repo"]
    document_reader = container["document_reader"]
    text_extractor = container["text_extractor"]
    cached_embeddings = container["cached_embeddings"]
    chunk_dedup_repo = container["chunk_dedup_repo"]
    vector_store = container["vector_store"]
    
    tenant = TenantId(tenant_id)
    doc_id = DocumentId(document_id)
    
    # Update status to processing
    document_repo.set_status(
        tenant_id=tenant,
        document_id=doc_id,
        status="processing",
    )
    
    try:
        # Step 1: Get stored file info
        stored_file = document_reader.get_stored_file(
            tenant_id=tenant,
            document_id=doc_id,
        )
        
        if not stored_file:
            raise ValueError(f"Document not found: {document_id}")
        
        log.info(
            "indexing_started",
            tenant_id=tenant_id,
            document_id=document_id,
            filename=stored_file.filename,
        )
        
        # Step 2: Extract text
        extracted = text_extractor.extract(
            stored_file.path,
            stored_file.content_type,
        )
        
        if not extracted.text.strip():
            raise ValueError("No text extracted from document")
        
        log.info(
            "text_extracted",
            document_id=document_id,
            text_length=len(extracted.text),
            metadata=extracted.metadata,
        )
        
        # Step 3: Chunk text
        chunks_text = chunk_text_token_aware(
            extracted.text,
            spec=ChunkSpec(max_tokens=512, overlap_tokens=50),
        )
        
        if not chunks_text:
            raise ValueError("No chunks produced from text")
        
        log.info(
            "chunks_created",
            document_id=document_id,
            chunks_count=len(chunks_text),
        )
        
        # Step 4: Deduplicate by hash
        hashes = [_chunk_hash(t) for t in chunks_text]
        
        # Get unique texts for batch embedding
        unique_by_hash: dict[str, str] = {}
        for h, t in zip(hashes, chunks_text):
            unique_by_hash.setdefault(h, t)
        
        unique_hashes = list(unique_by_hash.keys())
        unique_texts = [unique_by_hash[h] for h in unique_hashes]
        
        # Step 5: Batch embed unique texts
        unique_vectors = cached_embeddings.embed_many(unique_texts)
        vec_by_hash = dict(zip(unique_hashes, unique_vectors))
        
        log.info(
            "embeddings_generated",
            document_id=document_id,
            unique_chunks=len(unique_texts),
            total_chunks=len(chunks_text),
        )
        
        # Step 6: Store chunks in Postgres (with dedup)
        chunk_ids_in_order: list[str] = []
        for h, t in zip(hashes, chunks_text):
            chunk_id = chunk_dedup_repo.upsert_chunk_store(
                tenant_id=tenant,
                chunk_hash=h,
                text=t,
            )
            chunk_ids_in_order.append(chunk_id)
        
        # Update document → chunks mapping
        chunk_dedup_repo.replace_document_chunks(
            tenant_id=tenant,
            document_id=doc_id.value,
            chunk_ids_in_order=chunk_ids_in_order,
        )
        
        # Step 7: Upsert to vector store
        vector_store.ensure_collection()
        
        vectors_in_order = [vec_by_hash[h] for h in hashes]
        vector_store.upsert_points(
            ids=chunk_ids_in_order,
            vectors=vectors_in_order,
            tenant_id=tenant.value,
            document_id=doc_id.value,
        )
        
        # Step 8: Mark as indexed
        document_repo.set_status(
            tenant_id=tenant,
            document_id=doc_id,
            status="indexed",
        )
        
        log.info(
            "indexing_completed",
            tenant_id=tenant_id,
            document_id=document_id,
            chunks_count=len(chunk_ids_in_order),
        )
        
        return {
            "ok": True,
            "chunks": len(chunk_ids_in_order),
            "unique_chunks": len(unique_texts),
        }
        
    except Exception as e:
        # Mark as failed
        document_repo.set_status(
            tenant_id=tenant,
            document_id=doc_id,
            status="failed",
            error=str(e),
        )
        
        log.exception(
            "indexing_failed",
            tenant_id=tenant_id,
            document_id=document_id,
            error=str(e),
        )
        
        raise
