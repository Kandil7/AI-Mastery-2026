"""Create chunk_store and document_chunks tables

Revision ID: 002_add_chunk_store
Revises: 001_create_users_documents
Create Date: 2026-01-29

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "002_add_chunk_store"
down_revision = "001_create_users_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # chunk_store table
    op.create_table(
        "chunk_store",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_hash", sa.String(64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        # tsvector column - will be made GENERATED via raw SQL
        sa.Column("tsv", postgresql.TSVECTOR(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("user_id", "chunk_hash", name="uq_chunk_store_user_hash"),
    )
    op.create_index("ix_chunk_store_user_id", "chunk_store", ["user_id"])

    # Make tsv a GENERATED ALWAYS column
    op.execute("""
        ALTER TABLE chunk_store DROP COLUMN IF EXISTS tsv;
    """)
    op.execute("""
        ALTER TABLE chunk_store
        ADD COLUMN tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED;
    """)
    op.execute("""
        CREATE INDEX ix_chunk_store_tsv ON chunk_store USING gin (tsv);
    """)

    # document_chunks mapping table
    op.create_table(
        "document_chunks",
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("ord", sa.Integer(), primary_key=True),
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("chunk_store.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_chunk_id", "document_chunks", ["chunk_id"])


def downgrade() -> None:
    op.drop_index("ix_document_chunks_chunk_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")

    op.execute("DROP INDEX IF EXISTS ix_chunk_store_tsv;")
    op.drop_index("ix_chunk_store_user_id", table_name="chunk_store")
    op.drop_table("chunk_store")
