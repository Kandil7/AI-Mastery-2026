"""
Chunk Hydration Service
========================
Pure service for filling in chunk texts from database.

خدمة ملء نصوص القطع من قاعدة البيانات
"""

from typing import Sequence

from src.application.ports.chunk_text_reader import ChunkTextReaderPort
from src.domain.entities import Chunk, TenantId


def hydrate_chunk_texts(
    *,
    tenant_id: TenantId,
    chunks: Sequence[Chunk],
    reader: ChunkTextReaderPort,
) -> list[Chunk]:
    """
    Fill in chunk texts from database.
    
    Args:
        tenant_id: Owner tenant (for isolation)
        chunks: Chunks with empty or placeholder text
        reader: Database reader for text lookup
    
    Returns:
        New list of chunks with text filled in
    
    Design Decision: Text hydration is needed because:
    - Vector store (Qdrant) stores minimal payload (no text)
    - Reduces Qdrant storage costs significantly
    - Keeps text in Postgres as single source of truth
    - Easier text updates without re-embedding
    
    قرار التصميم: ملء النص مطلوب لأن مخزن المتجهات لا يخزن النص
    
    Example:
        >>> hydrated = hydrate_chunk_texts(
        ...     tenant_id=tenant,
        ...     chunks=vec_results,  # text is empty
        ...     reader=postgres_reader
        ... )
        >>> hydrated[0].text  # Now has text
        "The actual content..."
    """
    if not chunks:
        return []
    
    # Get all chunk IDs
    chunk_ids = [c.id for c in chunks]
    
    # Batch fetch texts
    texts_map = reader.get_texts_by_ids(
        tenant_id=tenant_id,
        chunk_ids=chunk_ids,
    )
    
    # Build new chunks with text
    result: list[Chunk] = []
    for chunk in chunks:
        text = texts_map.get(chunk.id, "")
        result.append(
            Chunk(
                id=chunk.id,
                tenant_id=chunk.tenant_id,
                document_id=chunk.document_id,
                text=text,
            )
        )
    
    return result
