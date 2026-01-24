from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rag.chunking import Chunk
from rag.retriever import RetrievedChunk
from storage.vectordb_base import VectorStore


class PgVectorStore(VectorStore):
    def __init__(self, dsn: str, table: str, embedding_dim: int) -> None:
        self._dsn = dsn
        self._table = table
        self._embedding_dim = embedding_dim
        self._ensure_table()

    def _connect(self):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        conn = psycopg2.connect(self._dsn)
        register_vector(conn)
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({self._embedding_dim}) NOT NULL,
                        metadata JSONB
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx
                    ON {self._table} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                    """
                )

    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        if not chunks:
            return
        rows = [
            (chunk.chunk_id, chunk.doc_id, chunk.text, embeddings[i], metadata)
            for i, chunk in enumerate(chunks)
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                from psycopg2.extras import execute_values

                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self._table} (chunk_id, doc_id, text, embedding, metadata)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET doc_id = EXCLUDED.doc_id,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                    """,
                    rows,
                )

    def query_by_vector(
        self,
        vector: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[RetrievedChunk]:
        where_sql = ""
        params: List[Any] = [vector]
        if filters:
            where_sql = "WHERE metadata @> %s"
            params.append(filters)

        query = f"""
            SELECT chunk_id, doc_id, text, metadata, 1 - (embedding <=> %s) AS score
            FROM {self._table}
            {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """
        params.append(vector)
        params.append(top_k)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows: List[Tuple[str, str, str, Dict[str, Any], float]] = cur.fetchall()
        return [
            RetrievedChunk(
                chunk_id=row[0],
                doc_id=row[1],
                text=row[2],
                metadata=row[3] or {},
                score=float(row[4]),
            )
            for row in rows
        ]
