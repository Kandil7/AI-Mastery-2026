"""Create chat sessions and turns tables

Revision ID: 003_add_chat_tables
Revises: 002_add_chunk_store
Create Date: 2026-01-29

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "003_add_chat_tables"
down_revision = "002_add_chunk_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # chat_sessions table
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])

    # chat_turns table
    op.create_table(
        "chat_turns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("sources", postgresql.ARRAY(sa.String(36)), nullable=False),
        sa.Column("retrieval_k", sa.Integer(), nullable=False, server_default="0"),
        # Observability fields
        sa.Column("embed_ms", sa.Integer(), nullable=True),
        sa.Column("search_ms", sa.Integer(), nullable=True),
        sa.Column("llm_ms", sa.Integer(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_chat_turns_session_id", "chat_turns", ["session_id"])
    op.create_index("ix_chat_turns_user_id", "chat_turns", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_turns_user_id", table_name="chat_turns")
    op.drop_index("ix_chat_turns_session_id", table_name="chat_turns")
    op.drop_table("chat_turns")

    op.drop_index("ix_chat_sessions_user_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")
