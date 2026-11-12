"""Stage 2 data models. Created via Alembic, never via startup create_tables().

These tables are additive and do not modify the legacy Source/News/Cluster/Post
schema. Images are stored as local files; only metadata and paths live in DB.
"""
from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)

from database.db import Base


class NewsImage(Base):
    """At most one main image is retained for each collected article."""
    __tablename__ = "news_images"
    __table_args__ = (
        UniqueConstraint("news_id", name="uq_news_images_news_id"),
    )

    id = Column(Integer, primary_key=True)
    news_id = Column(Integer, ForeignKey("news.id"), nullable=False)
    source_url = Column(Text, nullable=False)
    local_path = Column(Text, nullable=True)
    mime_type = Column(String(100), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    sha256 = Column(String(64), nullable=True)
    caption = Column(Text, nullable=True)
    credit = Column(Text, nullable=True)
    license_url = Column(Text, nullable=True)
    # Unknown is NOT permission for republication.
    rights_status = Column(String(24), nullable=False, default="unknown", server_default="unknown")
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class ModerationItem(Base):
    """One moderation request per frozen Post snapshot (revision via new Post)."""
    __tablename__ = "moderation_items"
    __table_args__ = (
        UniqueConstraint("post_id", name="uq_moderation_items_post_id"),
        Index("ix_moderation_items_status", "status"),
    )

    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    status = Column(String(30), nullable=False, default="pending", server_default="pending")
    # SHA-256 of the exact post text + selected image identifier, for future gate.
    snapshot_sha256 = Column(String(64), nullable=False)
    image_id = Column(Integer, ForeignKey("news_images.id"), nullable=True)
    # For image_id != NULL: explicitly require 'cleared' or 'remove_image'.
    image_rights_decision = Column(
        String(24), nullable=False, default="pending", server_default="pending"
    )
    moderator_telegram_id = Column(BigInteger, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    telegram_chat_id = Column(BigInteger, nullable=True)
    telegram_message_id = Column(BigInteger, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ModerationAction(Base):
    """Append-only moderation audit trail; no overwriting human decisions."""
    __tablename__ = "moderation_actions"
    __table_args__ = (
        Index("ix_moderation_actions_item_id", "moderation_item_id"),
    )

    id = Column(Integer, primary_key=True)
    moderation_item_id = Column(
        Integer, ForeignKey("moderation_items.id"), nullable=False
    )
    moderator_telegram_id = Column(BigInteger, nullable=False)
    action = Column(String(40), nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class PublicationAttempt(Base):
    """Track MAX attempts independently of the publication's current status."""
    __tablename__ = "publication_attempts"
    __table_args__ = (
        Index("ix_publication_attempts_post_id", "post_id"),
    )

    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    status = Column(String(30), nullable=False, default="started", server_default="started")
    started_at = Column(DateTime, nullable=False, server_default=func.now())
    finished_at = Column(DateTime, nullable=True)
    max_message_id = Column(String(255), nullable=True)
    error_text = Column(Text, nullable=True)
