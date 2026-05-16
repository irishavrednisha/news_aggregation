import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    func,
    JSON
)
from sqlalchemy.orm import relationship

from database.db import Base


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    url = Column(Text, nullable=False)
    rss_url = Column(Text, nullable=True)
    source_type = Column(String(50), default="rss")  # rss или html
    is_active = Column(Boolean, default=True)

    news = relationship("News", back_populates="source")


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)

    summary_text = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)

    cluster_size = Column(Integer, default=0)
    status = Column(String(50), default="new")

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    embedding = Column(JSON, nullable=True)
    title = Column(Text)
    news = relationship("News", back_populates="cluster")
    posts = relationship("Post", back_populates="cluster")


class News(Base):
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, index=True)

    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)

    title = Column(Text, nullable=False)
    text = Column(Text, nullable=True)
    url = Column(Text, nullable=False, unique=True)

    published_at = Column(DateTime, nullable=True)
    collected_at = Column(DateTime, server_default=func.now())
    embedding = Column(JSON, nullable=True)
    is_primary = Column(Boolean, default=False)

    source = relationship("Source", back_populates="news")
    cluster = relationship("Cluster", back_populates="news")


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)

    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)

    post_text = Column(Text, nullable=False)
    published_at = Column(DateTime, nullable=True)

    status = Column(String(50), default="draft")
    max_message_id = Column(String(255), nullable=True)

    cluster = relationship("Cluster", back_populates="posts")