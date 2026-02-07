"""Add Stage 2 hierarchical and contextual retrieval columns

Revision ID: 705d0a4c791f
Revises: 003_add_chat_tables
Create Date: 2026-01-29 18:50:19.360992

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '705d0a4c791f'
down_revision = '003_add_chat_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add columns to chunk_store
    op.add_column('chunk_store', sa.Column('parent_id', sa.String(length=36), nullable=True))
    op.add_column('chunk_store', sa.Column('chunk_context', sa.Text(), nullable=True))
    
    # Add foreign key constraint for self-reference
    op.create_foreign_key(
        'fk_chunk_store_parent',
        'chunk_store', 'chunk_store',
        ['parent_id'], ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    op.drop_constraint('fk_chunk_store_parent', 'chunk_store', type_='foreignkey')
    op.drop_column('chunk_store', 'chunk_context')
    op.drop_column('chunk_store', 'parent_id')
