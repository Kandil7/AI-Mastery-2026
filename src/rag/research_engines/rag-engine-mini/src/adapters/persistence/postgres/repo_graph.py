"""
Graph Repository Implementation
===============================
PostgreSQL implementation for knowledge graph storage.

تنفيذ مستودع الرسم البياني في PostgreSQL
"""

from typing import List, Sequence
from sqlalchemy import insert, select, delete
from src.adapters.persistence.postgres.db import SessionLocal
from src.adapters.persistence.postgres.models_graph import GraphTripletRow
from src.domain.entities import TenantId

class PostgresGraphRepo:
    """
    PostgreSQL implementation for storing and querying knowledge triplets.
    """

    def save_triplets(
        self,
        tenant_id: TenantId,
        document_id: str,
        chunk_id: str,
        triplets: List[dict],
    ) -> None:
        """
        Batch save triplets to the database.
        """
        if not triplets:
            return

        with SessionLocal() as db:
            rows = [
                {
                    "user_id": tenant_id.value,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "subject": t["subject"],
                    "relation": t["relation"],
                    "obj": t["obj"],
                }
                for t in triplets
            ]
            db.execute(insert(GraphTripletRow), rows)
            db.commit()

    def get_triplets_by_entity(
        self,
        tenant_id: TenantId,
        entity_name: str,
    ) -> List[dict]:
        """
        Find all triplets where entity is subject or object.
        """
        with SessionLocal() as db:
            stmt = select(GraphTripletRow).where(
                GraphTripletRow.user_id == tenant_id.value,
                (GraphTripletRow.subject.ilike(f"%{entity_name}%")) |
                (GraphTripletRow.obj.ilike(f"%{entity_name}%"))
            )
            rows = db.execute(stmt).scalars().all()
            return [
                {
                    "subject": r.subject,
                    "relation": r.relation,
                    "obj": r.obj,
                }
                for r in rows
            ]

    def delete_by_document(self, tenant_id: TenantId, document_id: str) -> int:
        """Delete all triplets for a document."""
        with SessionLocal() as db:
            stmt = delete(GraphTripletRow).where(
                GraphTripletRow.user_id == tenant_id.value,
                GraphTripletRow.document_id == document_id,
            )
            result = db.execute(stmt)
            db.commit()
            return result.rowcount
