"""
Celery Tasks - Document Indexing
=================================
Background task for document processing pipeline.
Enhanced for Stage 4: Multi-Modal (Images) & Structural (Tables) RAG.
"""

import hashlib
import structlog
import fitz  # PyMuPDF

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
def index_document(
    self,
    *,
    tenant_id: str,
    document_id: str,
    force_rechunk: bool = False,
    force_reembed: bool = False,
    use_new_strategy: bool = False,
) -> dict:
    """
    Index a document with Stage 4 features:
    1. Extract Text & Tables (High Precision)
    2. Extract & Describe Images (LLM-Vision)
    3. Hierarchical & Contextual linking
    4. Graph Triplet extraction
    """
    container = get_container()
    
    document_repo = container["document_repo"]
    document_reader = container["document_reader"]
    text_extractor = container["text_extractor"]
    cached_embeddings = container["cached_embeddings"]
    chunk_dedup_repo = container["chunk_dedup_repo"]
    vector_store = container["vector_store"]
    llm = container["llm"]
    graph_extractor = container["graph_extractor"]
    graph_repo = container["graph_repo"]
    vision_service = container["vision_service"]
    
    tenant = TenantId(tenant_id)
    doc_id = DocumentId(document_id)
    
    document_repo.set_status(tenant_id=tenant, document_id=doc_id, status="processing")
    
    try:
        # Step 0: Cleanup
        graph_repo.delete_by_document(tenant, doc_id.value)

        # Step 1: Extract text & Tables (Structural extraction)
        stored_file = document_reader.get_stored_file(tenant_id=tenant, document_id=doc_id)
        if not stored_file:
            raise ValueError(f"Document not found: {document_id}")
        
        extracted = text_extractor.extract(stored_file.path, stored_file.content_type)
        full_text = extracted.text
        if not full_text.strip():
            raise ValueError("No text extracted from document")

        # Step 2: Multi-Modal Extraction (Images)
        image_chunks = []
        if stored_file.content_type == "application/pdf":
            pdf_doc = fitz.open(stored_file.path)
            for page_idx in range(len(pdf_doc)):
                for img in pdf_doc.get_page_images(page_idx):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_doc, xref)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    description = vision_service.describe_image(pix.tobytes())
                    image_chunks.append({
                        "text": f"[Visual Content from Page {page_idx + 1}]: {description}",
                        "context": "Visual image description"
                    })
            pdf_doc.close()
            log.info("images_described", count=len(image_chunks))

        # Step 3: Generate Document Summary
        summary_prompt = (
            "Summarize the following document content in 1-2 sentences to provide context for RAG chunks. "
            "Output ONLY the summary sentences:\n\n"
            f"{full_text[:12000]}"
        )
        doc_summary = llm.generate(summary_prompt, max_tokens=150)

        # Step 4: Hierarchical Chunking
        spec = ChunkSpec(strategy="hierarchical", parent_size=2048, child_size=512)
        hierarchy = chunk_hierarchical(full_text, spec)
        
        # Merge descriptions as top-level chunks
        for img_c in image_chunks:
            hierarchy.append({
                "child_text": img_c["text"],
                "parent_text": img_c["text"]
            })

        # Step 5: Batch Embedding
        child_texts = [h["child_text"] for h in hierarchy]
        unique_child_texts = list(set(child_texts))
        unique_vectors = cached_embeddings.embed_many(unique_child_texts)
        vec_map = dict(zip(unique_child_texts, unique_vectors))

        # Step 6: Detailed Storage & Graph extraction
        chunk_ids_in_order: list[str] = []
        handled_parents = set()

        for item in hierarchy:
            c_text = item["child_text"]
            p_text = item["parent_text"]
            
            p_hash = _chunk_hash(p_text)
            p_id = chunk_dedup_repo.upsert_chunk_store(
                tenant_id=tenant,
                chunk_hash=p_hash,
                text=p_text,
                chunk_context=doc_summary,
            )
            
            if p_id not in handled_parents:
                triplets = graph_extractor.extract_triplets(p_text)
                if triplets:
                    graph_repo.save_triplets(tenant, doc_id.value, p_id, triplets)
                handled_parents.add(p_id)

            c_id = chunk_dedup_repo.upsert_chunk_store(
                tenant_id=tenant,
                chunk_hash=_chunk_hash(c_text),
                text=c_text,
                parent_id=p_id,
                chunk_context=doc_summary,
            )
            chunk_ids_in_order.append(c_id)
            
            vector_store.upsert_points(
                ids=[c_id],
                vectors=[vec_map[c_text]],
                tenant_id=tenant.value,
                document_id=doc_id.value,
            )

        # Step 7: Finalize
        chunk_dedup_repo.replace_document_chunks(
            tenant_id=tenant,
            document_id=doc_id.value,
            chunk_ids_in_order=chunk_ids_in_order,
        )
        
        document_repo.set_status(tenant_id=tenant, document_id=doc_id, status="indexed")
        log.info("indexing_complete", chunks=len(chunk_ids_in_order))
        
        return {"ok": True, "chunks": len(chunk_ids_in_order), "multi_modal": True}
        
    except Exception as e:
        document_repo.set_status(tenant_id=tenant, document_id=doc_id, status="failed", error=str(e))
        log.exception("indexing_failed", error=str(e))
        raise
