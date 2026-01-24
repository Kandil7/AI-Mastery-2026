import asyncio
import logging
from typing import Iterable, List, Optional

from src.retrieval import Document
from src.retrieval.vector_store import vector_manager
from src.ingestion.mongo_storage import mongo_storage
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


async def index_documents(
    rag_pipeline: RAGPipeline,
    documents: Iterable[Document],
    *,
    persist_vectors: bool = True,
    persist_documents: bool = True,
) -> None:
    documents_list: List[Document] = list(documents)
    if not documents_list:
        return

    # Always index into the in-process retriever to keep query path consistent.
    rag_pipeline.index(documents_list)

    # Persist documents to MongoDB (metadata store).
    if persist_documents and mongo_storage is not None:
        try:
            await mongo_storage.store_documents(documents_list)
        except Exception as exc:
            logger.warning("MongoDB persistence failed: %s", exc)

    # Persist vectors to the vector store (if configured).
    if persist_vectors and vector_manager is not None:
        try:
            encoder = rag_pipeline.retriever.dense_retriever.encoder
        except Exception as exc:
            logger.warning("Dense encoder unavailable for vector persistence: %s", exc)
            return

        try:
            embeddings = encoder.encode(
                [doc.content for doc in documents_list],
                convert_to_numpy=True,
            )
        except Exception as exc:
            logger.warning("Embedding generation failed: %s", exc)
            return

        tasks = []
        for doc, embedding in zip(documents_list, embeddings):
            metadata = {
                **(doc.metadata or {}),
                "document_id": doc.id,
                "source": doc.source,
                "doc_type": doc.doc_type,
                "dimension": len(embedding),
            }
            tasks.append(
                vector_manager.add_document_vector(
                    document_id=doc.id,
                    vector=embedding.tolist(),
                    text_content=doc.content,
                    metadata=metadata,
                )
            )

        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["index_documents"]
