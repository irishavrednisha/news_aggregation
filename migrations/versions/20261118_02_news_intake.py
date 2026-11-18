"""Stage 2: audit raw articles and exact duplicates without altering legacy news.

Revision ID: 20261118_02
Revises: 20261112_01
Create Date: 2026-11-18
"""
from alembic import op
import sqlalchemy as sa

revision = "20261118_02"
down_revision = "20261112_01"
branch_labels = None
depends_on = None


def upgrade():
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "news_intake_records" in tables:
        raise RuntimeError("news_intake_records already exists; inspect migration state before retry")
    if not {"sources", "news", "clusters", "posts"}.issubset(tables):
        raise RuntimeError("Legacy tables missing; check DATABASE_URL and backup. Nothing created.")
    op.create_table(
        "news_intake_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id"), nullable=False),
        sa.Column("news_id", sa.Integer(), sa.ForeignKey("news.id"), nullable=True),
        sa.Column("duplicate_of_news_id", sa.Integer(), sa.ForeignKey("news.id"), nullable=True),
        sa.Column("raw_url", sa.Text(), nullable=False),
        sa.Column("raw_title", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("raw_published_at", sa.DateTime(), nullable=True),
        sa.Column("text_source", sa.String(length=40), nullable=True),
        sa.Column("canonical_url", sa.Text(), nullable=True),
        sa.Column("url_sha256", sa.String(length=64), nullable=False),
        sa.Column("payload_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("reason", sa.String(length=80), nullable=False),
        sa.Column("decision_stage", sa.String(length=20), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("seen_count", sa.Integer(), nullable=False, server_default="1"),
        sa.UniqueConstraint("source_id", "url_sha256", "payload_sha256",
                            name="uq_intake_source_url_payload"),
    )
    op.create_index("ix_intake_url_sha256", "news_intake_records", ["url_sha256"])
    op.create_index("ix_intake_status", "news_intake_records", ["status"])
    op.create_index("ix_intake_duplicate", "news_intake_records", ["duplicate_of_news_id"])


def downgrade():
    raise RuntimeError("Destructive downgrade disabled: raw audit history must be retained.")
