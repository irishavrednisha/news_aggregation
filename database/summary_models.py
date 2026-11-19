"""Append-only Stage 4 audit records; NO changes to the publication schema."""
from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, JSON, String, Text, func
from database.db import Base


class SummaryRun(Base):
    __tablename__ = "summary_runs"
    __table_args__ = (
        Index("ix_summary_runs_cluster_id", "cluster_id"),
        Index("ix_summary_runs_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    source_sha256 = Column(String(64), nullable=False)
    source_materials = Column(JSON, nullable=False)
    extracted_facts = Column(JSON, nullable=False)
    textrank_summary = Column(Text, nullable=False)
    gemma_summary = Column(Text, nullable=False)
    model = Column(String(80), nullable=False)
    model_parameters = Column(JSON, nullable=False)
    validation = Column(JSON, nullable=False)
    attempts = Column(JSON, nullable=False)
    status = Column(String(30), nullable=False)  # summary_draft | summary_review
    created_at = Column(DateTime, nullable=False, server_default=func.now())
