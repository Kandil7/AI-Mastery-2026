"""
Celery Tasks - Document Indexing
================================
Background task for document processing pipeline.
Enhanced for Stage 4: Multi-Modal (Images) & Structural (Tables) RAG.
"""

import hashlib
import logging
import structlog
import fitz  # PyMuPDF
from typing import List, Dict, Any

from src.workers.celery_app import celery_app
from src.core.bootstrap import get_container
from src.domain.entities import TenantId, DocumentId, Chunk, ChunkSpec
from src.application.services.chunking import chunk_text_token_aware, chunk_hierarchical

log = structlog.get_logger()
logger = logging.getLogger(__name__)


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
                    image_chunks.append(
                        {
                            "text": f"[Visual Content from Page {page_idx + 1}]: {description}",
                            "context": "Visual image description",
                        }
                    )
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
            hierarchy.append({"child_text": img_c["text"], "parent_text": img_c["text"]})

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
        document_repo.set_status(
            tenant_id=tenant, document_id=doc_id, status="failed", error=str(e)
        )
        log.exception("indexing_failed", error=str(e))
        raise


@celery_app.task(
    name="bulk_upload_documents",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def bulk_upload_documents(
    self,
    *,
    tenant_id: str,
    files: List[Dict[str, Any]],
) -> dict:
    """
    Process multiple document uploads in bulk.

    Args:
        tenant_id: Tenant/user ID
        files: List of file dicts with keys:
            - filename: Original filename
            - content_type: MIME type
            - content: File bytes

    Returns:
        Dict with success/failure counts and results

    معالجة رفع مستندات متعددة بالجملة
    """
    container = get_container()
    document_reader = container["document_reader"]
    document_repo = container["document_repo"]

    tenant = TenantId(tenant_id)
    results = []
    success_count = 0
    failure_count = 0

    for file_info in files:
        try:
            # Store file
            stored = await document_reader.save_upload(
                tenant_id=tenant,
                upload_filename=file_info["filename"],
                content_type=file_info["content_type"],
                data=file_info["content"],
            )

            # Create document record
            file_hash = hashlib.sha256(file_info["content"]).hexdigest()
            document_id = document_repo.create_document(
                tenant_id=tenant,
                stored_file=stored,
                file_sha256=file_hash,
            )

            # Queue indexing task
            index_document.delay(
                tenant_id=tenant_id,
                document_id=document_id.value,
            )

            results.append(
                {
                    "filename": file_info["filename"],
                    "document_id": document_id.value,
                    "status": "queued",
                }
            )
            success_count += 1

        except Exception as e:
            logger.error(f"Bulk upload failed for {file_info.get('filename')}", error=str(e))
            results.append(
                {
                    "filename": file_info.get("filename", "unknown"),
                    "status": "failed",
                    "error": str(e),
                }
            )
            failure_count += 1

    logger.info(
        "bulk_upload_complete",
        total=len(files),
        success=success_count,
        failures=failure_count,
    )

    return {
        "total": len(files),
        "success": success_count,
        "failures": failure_count,
        "results": results,
    }


@celery_app.task(
    name="bulk_delete_documents",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def bulk_delete_documents(
    self,
    *,
    tenant_id: str,
    document_ids: List[str],
) -> dict:
    """
    Delete multiple documents in bulk.

    Args:
        tenant_id: Tenant/user ID
        document_ids: List of document IDs to delete

    Returns:
        Dict with success/failure counts

    حذف مستندات متعددة بالجملة
    """
    container = get_container()
    document_repo = container["document_repo"]
    vector_store = container["vector_store"]
    chunk_repo = container["chunk_repo"]

    tenant = TenantId(tenant_id)
    results = []
    success_count = 0
    failure_count = 0

    for doc_id_str in document_ids:
        try:
            doc_id = DocumentId(doc_id_str)

            # Delete from vector store
            vector_store.delete_points(
                tenant_id=tenant,
                document_id=doc_id,
            )

            # Delete chunks from database
            chunk_repo.delete_by_document(tenant_id=tenant, document_id=doc_id)

            # Delete document record
            document_repo.delete_document(tenant_id=tenant, document_id=doc_id)

            results.append(
                {
                    "document_id": doc_id_str,
                    "status": "deleted",
                }
            )
            success_count += 1

        except Exception as e:
            logger.error(f"Bulk delete failed for {doc_id_str}", error=str(e))
            results.append(
                {
                    "document_id": doc_id_str,
                    "status": "failed",
                    "error": str(e),
                }
            )
            failure_count += 1

    logger.info(
        "bulk_delete_complete",
        total=len(document_ids),
        success=success_count,
        failures=failure_count,
    )

    return {
        "total": len(document_ids),
        "success": success_count,
        "failures": failure_count,
        "results": results,
    }


@celery_app.task(
    name="merge_pdfs",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def merge_pdfs(
    self,
    *,
    tenant_id: str,
    source_document_ids: List[str],
    merged_filename: str,
    target_document_id: str | None = None,
) -> dict:
    """
    Merge multiple PDF documents into one.

    Args:
        tenant_id: Tenant/user ID
        source_document_ids: List of source document IDs
        merged_filename: Filename for merged PDF
        target_document_id: Optional target document ID (creates new if None)

    Returns:
        Dict with merged document info

    دمج عدة مستندات PDF في مستند واحد
    """
    import PyPDF2
    from io import BytesIO

    container = get_container()
    document_reader = container["document_reader"]
    document_repo = container["document_repo"]
    cached_embeddings = container["cached_embeddings"]

    tenant = TenantId(tenant_id)

    # Retrieve source documents
    source_docs = []
    for doc_id_str in source_document_ids:
        doc_id = DocumentId(doc_id_str)
        stored_file = document_reader.get_stored_file(tenant_id=tenant, document_id=doc_id)

        # Read PDF content
        with open(stored_file.path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            source_docs.append(
                {
                    "reader": pdf_reader,
                    "filename": stored_file.filename,
                }
            )

    # Create merged PDF
    merged_writer = PyPDF2.PdfWriter()

    for source in source_docs:
        for page_num in range(len(source["reader"].pages)):
            merged_writer.add_page(source["reader"].pages[page_num])

    # Write merged PDF to bytes
    merged_bytes = BytesIO()
    merged_writer.write(merged_bytes)
    merged_content = merged_bytes.getvalue()

    # Store merged file
    stored = await document_reader.save_upload(
        tenant_id=tenant,
        upload_filename=merged_filename,
        content_type="application/pdf",
        data=merged_content,
    )

    # Create or update document
    file_hash = hashlib.sha256(merged_content).hexdigest()

    if target_document_id:
        # Update existing document
        target_id = DocumentId(target_document_id)
        updated_doc = document_repo.update_document(
            tenant_id=tenant,
            document_id=target_id,
            stored_file=stored,
            file_sha256=file_hash,
        )
        merged_document_id = target_document_id
    else:
        # Create new document
        new_id = document_repo.create_document(
            tenant_id=tenant,
            stored_file=stored,
            file_sha256=file_hash,
        )
        merged_document_id = new_id.value

    # Queue indexing task
    index_document.delay(
        tenant_id=tenant_id,
        document_id=merged_document_id,
    )

    logger.info(
        "pdf_merge_complete",
        source_count=len(source_document_ids),
        merged_document_id=merged_document_id,
    )

    return {
        "merged_document_id": merged_document_id,
        "source_count": len(source_document_ids),
        "filename": merged_filename,
        "size_bytes": len(merged_content),
    }


@celery_app.task(
    name="generate_chat_title",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
)
def generate_chat_title(
    self,
    *,
    tenant_id: str,
    session_id: str,
) -> dict:
    """
    Generate a title for a chat session using LLM.

    Args:
        tenant_id: Tenant/user ID
        session_id: Chat session ID

    Returns:
        Dict with generated title

    توليد عنوان لجلسة المحادثة باستخدام LLM
    """
    container = get_container()
    chat_repo = container["chat_repo"]
    llm = container["llm"]

    tenant = TenantId(tenant_id)

    # Get chat turns
    turns = chat_repo.get_session_turns(
        tenant_id=tenant,
        session_id=session_id,
        limit=5,
    )

    if not turns:
        return {"title": "New Chat", "status": "skipped"}

    # Build context from turns
    context_parts = []
    for i, turn in enumerate(turns[:3]):  # First 3 turns
        context_parts.append(f"Q{i + 1}: {turn.question}")
        context_parts.append(f"A{i + 1}: {turn.answer[:200]}")  # Truncate answers

    context = "\n\n".join(context_parts)

    # Generate title using LLM
    prompt = f"""Generate a concise title (max 50 characters) for this chat session:

{context}

Guidelines:
- Use the main topic discussed
- Keep it short and descriptive
- Focus on what the user was asking about
- Do NOT include the word "Chat" or "Session"
- Provide ONLY the title, nothing else

Title:"""

    try:
        title = llm.generate(prompt, temperature=0.3, max_tokens=50)

        # Clean up title
        title = title.strip()
        title = title.replace('"', "").replace("'", "")
        title = title[:50]

        # Update session with title (TODO: implement update_session in chat_repo)
        # chat_repo.update_session_title(tenant_id=tenant, session_id=session_id, title=title)

        logger.info("chat_title_generated", session_id=session_id, title=title)

        return {"title": title, "status": "success"}

    except Exception as e:
        logger.error("chat_title_generation_failed", session_id=session_id, error=str(e))
        return {"title": "New Chat", "status": "failed", "error": str(e)}


@celery_app.task(
    name="summarize_chat_session",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
)
def summarize_chat_session(
    self,
    *,
    tenant_id: str,
    session_id: str,
) -> dict:
    """
    Generate a summary for a completed chat session.

    Args:
        tenant_id: Tenant/user ID
        session_id: Chat session ID

    Returns:
        Dict with summary, topics, sentiment

    تلخيص جلسة محادثة مكتملة
    """
    container = get_container()
    chat_repo = container["chat_repo"]
    llm = container["llm"]

    tenant = TenantId(tenant_id)

    # Get all chat turns
    turns = chat_repo.get_session_turns(
        tenant_id=tenant,
        session_id=session_id,
        limit=100,
    )

    if not turns:
        return {
            "summary": "Empty session",
            "topics": [],
            "sentiment": "neutral",
            "status": "skipped",
        }

    # Build Q&A context
    qa_text = "\n\n".join(
        [
            f"Q{i + 1}: {turn.question}\nA{i + 1}: {turn.answer[:300]}"
            for i, turn in enumerate(turns[-5:])  # Last 5 turns
        ]
    )

    # Generate summary using LLM
    prompt = f"""Analyze this chat session and provide:

1. A concise summary (200 chars max)
2. Main topics discussed (comma-separated, max 5)
3. Overall sentiment (positive/neutral/negative)

Chat Session:
{qa_text}

Format your response as:
Summary: [summary]
Topics: [topic1, topic2, ...]
Sentiment: [positive/neutral/negative]"""

    try:
        response = llm.generate(prompt, temperature=0.2, max_tokens=200)

        # Parse response
        summary = ""
        topics = []
        sentiment = "neutral"

        for line in response.split("\n"):
            line = line.strip()
            if line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
            elif line.lower().startswith("topics:"):
                topics_str = line.split(":", 1)[1].strip()
                topics = [t.strip() for t in topics_str.split(",")][:5]
            elif line.lower().startswith("sentiment:"):
                sentiment = line.split(":", 1)[1].strip().lower()

        # Validate sentiment
        if sentiment not in ["positive", "neutral", "negative"]:
            sentiment = "neutral"

        # Update session with summary (TODO: implement update_session in chat_repo)
        # chat_repo.update_session_summary(tenant_id=tenant, session_id=session_id, summary=...)

        logger.info(
            "chat_summarized",
            session_id=session_id,
            summary_length=len(summary),
            topics=topics,
            sentiment=sentiment,
        )

        return {
            "summary": summary,
            "topics": topics,
            "sentiment": sentiment,
            "question_count": len(turns),
            "status": "success",
        }

    except Exception as e:
        logger.error("chat_summary_failed", session_id=session_id, error=str(e))
        return {
            "summary": "Summary generation failed",
            "topics": ["General"],
            "sentiment": "neutral",
            "status": "failed",
            "error": str(e),
        }
