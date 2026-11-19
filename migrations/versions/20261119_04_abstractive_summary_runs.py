"""Stage 4 append-only local Gemma audit. Does not alter News/Cluster/Post.

Revision ID: 20261119_04
Revises: 20261118_03
"""
from alembic import op
import sqlalchemy as sa

revision = "20261119_04"
down_revision = "20261118_03"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if "summary_runs" in inspector.get_table_names():
        raise RuntimeError("summary_runs already exists; inspect schema before migration")
    op.create_table(
        "summary_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("cluster_id", sa.Integer(), sa.ForeignKey("clusters.id"), nullable=False),
        sa.Column("source_sha256", sa.String(64), nullable=False),
        sa.Column("source_materials", sa.JSON(), nullable=False),
        sa.Column("extracted_facts", sa.JSON(), nullable=False),
        sa.Column("textrank_summary", sa.Text(), nullable=False),
        sa.Column("gemma_summary", sa.Text(), nullable=False),
        sa.Column("model", sa.String(80), nullable=False),
        sa.Column("model_parameters", sa.JSON(), nullable=False),
        sa.Column("validation", sa.JSON(), nullable=False),
        sa.Column("attempts", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_summary_runs_cluster_id", "summary_runs", ["cluster_id"])
    op.create_index("ix_summary_runs_created_at", "summary_runs", ["created_at"])


def downgrade():
    raise RuntimeError("Stage 4 audit records are append-only; destructive downgrade disabled")
