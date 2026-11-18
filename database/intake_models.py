"""Stage 2 append-only snapshots of fetched articles, including rejects and copies.

Additive migration 20261118_02 installs this table. No changes to old news.
"""
from sqlalchemy import (
    Column, DateTime, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, func,
)
from database.db import Base


class NewsIntakeRecord(Base):
    __tablename__ = "news_intake_records"
    __table_args__ = (
        UniqueConstraint("source_id", "url_sha256", "payload_sha256",
                         name="uq_intake_source_url_payload"),
        Index("ix_intake_url_sha256", "url_sha256"),
        Index("ix_intake_status", "status"),
        Index("ix_intake_duplicate", "duplicate_of_news_id"),
        Index("ix_intake_duplicate_raw", "duplicate_of_intake_id"),
    )

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    news_id = Column(Integer, ForeignKey("news.id"), nullable=True)
    duplicate_of_news_id = Column(Integer, ForeignKey("news.id"), nullable=True)
    duplicate_of_intake_id = Column(Integer, ForeignKey("news_intake_records.id"), nullable=True)
    raw_url = Column(Text, nullable=False)
    raw_title = Column(Text, nullable=False)
    raw_text = Column(Text, nullable=False)
    raw_published_at = Column(DateTime, nullable=True)
    text_source = Column(String(40), nullable=True)
    canonical_url = Column(Text, nullable=True)
    url_sha256 = Column(String(64), nullable=False)
    payload_sha256 = Column(String(64), nullable=False)
    status = Column(String(20), nullable=False)
    reason = Column(String(80), nullable=False)
    decision_stage = Column(String(20), nullable=False)
    first_seen_at = Column(DateTime, nullable=False, server_default=func.now())
    last_seen_at = Column(DateTime, nullable=False, server_default=func.now())
    seen_count = Column(Integer, nullable=False, server_default="1", default=1)
