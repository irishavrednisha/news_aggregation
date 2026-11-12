"""Stage 2: add image, Telegram moderation, and publication attempt tables.

Revision ID: 20261112_01
Revises:
Create Date: 2026-11-12

ADDITIVE ONLY. Existing news, sources, clusters, posts are never altered.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20261112_01"
down_revision = None
branch_labels = None
depends_on = None

_EXISTING = {"sources", "news", "clusters", "posts"}
_NEW = {
    "news_images",
    "moderation_items",
    "moderation_actions",
    "publication_attempts",
}


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    missing = _EXISTING - tables
    if missing:
        raise RuntimeError(
            "Найдена пустая/не та БД: отсутствуют исходные таблицы: "
            + ", ".join(sorted(missing))
            + ". Сначала проверьте DB_NAME и резервную копию."
        )
    already_present = _NEW & tables
    if already_present:
        raise RuntimeError(
            "Новые таблицы уже существуют, миграция остановлена: "
            + ", ".join(sorted(already_present))
            + ". Не применяйте stamp без проверки."
        )

    op.create_table(
        "news_images",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("news_id", sa.Integer(), sa.ForeignKey("news.id"), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("local_path", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("credit", sa.Text(), nullable=True),
        sa.Column("license_url", sa.Text(), nullable=True),
        sa.Column(
            "rights_status",
            sa.String(length=24),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("news_id", name="uq_news_images_news_id"),
    )
    op.create_table(
        "moderation_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("post_id", sa.Integer(), sa.ForeignKey("posts.id"), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="pending"),
        sa.Column("snapshot_sha256", sa.String(length=64), nullable=False),
        sa.Column("image_id", sa.Integer(), sa.ForeignKey("news_images.id"), nullable=True),
        sa.Column(
            "image_rights_decision",
            sa.String(length=24),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("moderator_telegram_id", sa.BigInteger(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("telegram_chat_id", sa.BigInteger(), nullable=True),
        sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("post_id", name="uq_moderation_items_post_id"),
    )
    op.create_index(
        "ix_moderation_items_status", "moderation_items", ["status"]
    )
    op.create_table(
        "moderation_actions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "moderation_item_id",
            sa.Integer(),
            sa.ForeignKey("moderation_items.id"),
            nullable=False,
        ),
        sa.Column("moderator_telegram_id", sa.BigInteger(), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_moderation_actions_item_id", "moderation_actions", ["moderation_item_id"]
    )
    op.create_table(
        "publication_attempts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("post_id", sa.Integer(), sa.ForeignKey("posts.id"), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="started"),
        sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("max_message_id", sa.String(length=255), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_publication_attempts_post_id", "publication_attempts", ["post_id"]
    )


def downgrade() -> None:
    raise RuntimeError(
        "Автоматический откат отключён: он удалил бы журнал модерации и "
        "изображения. Восстановление возможно только из проверенной копии."
    )
