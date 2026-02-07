"""Add graph triplets table for Stage 3

Revision ID: 0c169705ce73
Revises: 705d0a4c791f
Create Date: 2026-01-29 19:18:06.871348

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0c169705ce73'
down_revision = '705d0a4c791f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create graph_triplets table
    op.create_table(
        'graph_triplets',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column('chunk_id', sa.String(length=36), nullable=False),
        sa.Column('subject', sa.String(length=256), nullable=False),
        sa.Column('relation', sa.String(length=256), nullable=False),
        sa.Column('obj', sa.String(length=256), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['chunk_id'], ['chunk_store.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_graph_document', 'graph_triplets', ['document_id'], unique=False)
    op.create_index('ix_graph_object', 'graph_triplets', ['obj'], unique=False)
    op.create_index('ix_graph_subject', 'graph_triplets', ['subject'], unique=False)
    op.create_index('ix_graph_user_id', 'graph_triplets', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_graph_user_id', table_name='graph_triplets')
    op.drop_index('ix_graph_subject', table_name='graph_triplets')
    op.drop_index('ix_graph_object', table_name='graph_triplets')
    op.drop_index('ix_graph_document', table_name='graph_triplets')
    op.drop_table('graph_triplets')
