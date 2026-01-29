"""
Celery Tasks - Document Indexing
=================================
Background task for document processing pipeline.
Enhanced for Stage 2: Hierarchical & Contextual RAG.
"""

import hashlib
import structlog

from src.workers.celery_app import celery_app
from src.core.bootstrap import get_container
from src.domain.entities import TenantId, DocumentId, Chunk, ChunkSpec
from src.application.services.chunking import chunk_text_token_aware, chunk_hierarchical

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
    Index a document with Stage 2 features:
    1. Extract text
    2. Generate Document Context (Contextual Retrieval)
    3. Hierarchical Chunking (Parent-Child)
    4. Link context to chunks
    5. Store and index
    """
    cached_embeddings = container["cached_embeddings"]
    chunk_dedup_repo = container["chunk_dedup_repo"]
    vector_store = container["vector_store"]
    llm = container["llm"]
    graph_extractor = container["graph_extractor"]
    graph_repo = container["graph_repo"]
    
    tenant = TenantId(tenant_id)
    doc_id = DocumentId(document_id)
    
    document_repo.set_status(tenant_id=tenant, document_id=doc_id, status="processing")
    
    try:
        # 0. Cleanup old graph data if re-indexing
        graph_repo.delete_by_document(tenant, doc_id.value)

        # 1. Extract text
        stored_file = document_reader.get_stored_file(tenant_id=tenant, document_id=doc_id)
        if not stored_file: raise ValueError(f"Document not found: {document_id}")
        
        extracted = text_extractor.extract(stored_file.path, stored_file.content_type)
        full_text = extracted.text
        if not full_text.strip(): raise ValueError("No text extracted from document")

        # 2. Generate Document Summary (for Contextual Retrieval)
        summary_prompt = (
            "Summarize the following document content in 1-2 sentences to provide context for RAG chunks. "
            "Output ONLY the summary sentences:\n\n"
            f"{full_text[:12000]}"
        )
        doc_summary = llm.generate(summary_prompt, max_tokens=150)
        
        log.info("contextual_summary_generated", document_id=document_id, length=len(doc_summary))

        # 3. Hierarchical Chunking
        spec = ChunkSpec(strategy="hierarchical", parent_size=2048, child_size=512)
        hierarchy = chunk_hierarchical(full_text, spec)
        
        log.info("hierarchical_chunks_created", document_id=document_id, count=len(hierarchy))

        # 4. Batch Embedding (Child chunks)
        child_texts = [h["child_text"] for h in hierarchy]
        unique_child_texts = list(set(child_texts))
        unique_vectors = cached_embeddings.embed_many(unique_child_texts)
        vec_map = dict(zip(unique_child_texts, unique_vectors))

        # 5. Store in DB with links & Graph Extraction
        chunk_ids_in_order: list[str] = []
        
        # Keep track of handled parents to avoid redundant graph extraction
        handled_parents = set()

        for item in hierarchy:
            c_text = item["child_text"]
            p_text = item["parent_text"]
            
            # Record Parent (Contextual)
            p_hash = _chunk_hash(p_text)
            p_id = chunk_dedup_repo.upsert_chunk_store(
                tenant_id=tenant,
                chunk_hash=p_hash,
                text=p_text,
                chunk_context=doc_summary,
            )
            
            # Stage 3: Extract Knowledge Graph from unique parent chunks
            if p_id not in handled_parents:
                triplets = graph_extractor.extract_triplets(p_text)
                if triplets:
                    graph_repo.save_triplets(tenant, doc_id.value, p_id, triplets)
                handled_parents.add(p_id)

            # Record Child (Searchable) linked to Parent
            c_id = chunk_dedup_repo.upsert_chunk_store(
                tenant_id=tenant,
                chunk_hash=_chunk_hash(c_text),
                text=c_text,
                parent_id=p_id,
                chunk_context=doc_summary,
            )
        
        chunk_ids_in_order.append(c_id)
        
        # Add to Vector Store (Search child)
        vector_store.upsert_points(
            ids=[c_id],
            vectors=[vec_map[c_text]],
            tenant_id=tenant.value,
            document_id=doc_id.value,
        )

    # Step 6: Mark as Complete
    chunk_dedup_repo.replace_document_chunks(
        tenant_id=tenant,
        document_id=doc_id.value,
        chunk_ids_in_order=chunk_ids_in_order,
    )
    
    document_repo.set_status(tenant_id=tenant, document_id=doc_id, status="indexed")
    
    log.info("indexing_completed", document_id=document_id, chunks=len(chunk_ids_in_order))
    
    return {
        "ok": True,
        "document_id": document_id,
        "chunks": len(chunk_ids_in_order),
        "contextual": True
    }
    
except Exception as e:
    document_repo.set_status(tenant_id=tenant, document_id=doc_id, status="failed", error=str(e))
    log.exception("indexing_failed", document_id=document_id, error=str(e))
    raise
