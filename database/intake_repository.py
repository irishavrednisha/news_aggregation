"""Atomic stage 2 ingestion; old News table remains the sole ML input.

Rejected/reviewed/duplicate raw articles stay in news_intake_records. Every
accepted article has both an intake snapshot and one News row. No publisher.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os

from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from database.models import News, Source
from database.intake_models import NewsIntakeRecord
from processing.exact_duplicates import (
    ExactDuplicateIndex, checksum, normalize_text, normalize_title, normalize_url,
)
from processing.news_intake import classify_intake


def require_intake_schema(session):
    if not inspect(session.get_bind()).has_table("news_intake_records"):
        raise RuntimeError(
            "Stage 2 schema missing. Back up PostgreSQL and run "
            "'python -m alembic upgrade head' before collecting. "
            "No automatic table creation or legacy fallback."
        )


@dataclass
class IntakeStats:
    saved: int = 0
    counts: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    def as_dict(self):
        return {"saved": self.saved, "counts": dict(self.counts), "errors": self.errors}


def _snapshot_hash(article):
    # Raw snapshots are immutable. A revised article with the same URL creates
    # another snapshot; an identical rerun updates only seen_count.
    published = getattr(article, "published_at", None)
    timestamp = published.isoformat() if hasattr(published, "isoformat") else str(published or "")
    return checksum("\n".join((
        str(getattr(article, "title", "") or ""),
        str(getattr(article, "text", "") or ""),
        timestamp,
        str(getattr(article, "text_source", "") or ""),
    )))


def _store_image_after_commit(session, article, news):
    # Preserve opt-in image collection from stage 1, without letting a photo
    # error undo the atomically committed text+intake snapshot.
    if os.getenv("NEWS_IMAGES_ENABLED", "false").strip().casefold() not in (
        "1", "true", "yes", "on"
    ):
        return
    candidates = getattr(article, "image_candidates", None) or []
    if not candidates:
        return
    try:
        from database.moderation_models import NewsImage
        from processing.image_downloader import download_main_image
        image = download_main_image(
            candidates, storage_dir=os.getenv("NEWS_IMAGE_STORAGE_DIR", "data/news_images")
        )
        if image is None:
            return
        session.add(NewsImage(
            news_id=news.id, source_url=image.source_url,
            local_path=image.local_path, mime_type=image.mime_type,
            width=image.width, height=image.height, sha256=image.sha256,
            caption=image.caption, credit=image.credit, license_url=image.license_url,
            rights_status="unknown",
        ))
        session.commit()
    except Exception:
        session.rollback()


def save_intake_batch(session, articles, *, now=None, max_age_days=30) -> IntakeStats:
    """All reads/writes use caller session. Never clears rows or rebuilds DB.

    This implementation prioritizes correctness for the existing small database.
    A future bulk fingerprint index/backfill can optimize large installations.
    """
    require_intake_schema(session)
    stats = IntakeStats()
    index = ExactDuplicateIndex()
    # Legacy rows must be considered; old data have no intake snapshots.
    for existing in session.query(News).all():
        index.add(existing)

    for article in articles:
        try:
            source = session.query(Source).filter(
                Source.name == getattr(article, "source", "")
            ).first()
            if source is None:
                stats.counts["unknown_source"] += 1
                stats.errors.append("unknown_source:" + str(getattr(article, "source", "")))
                continue
            raw_url = str(getattr(article, "url", "") or "")
            canonical = normalize_url(raw_url)
            url_sha = checksum(canonical or raw_url)
            payload_sha = _snapshot_hash(article)
            previous = session.query(NewsIntakeRecord).filter_by(
                source_id=source.id, url_sha256=url_sha,
                payload_sha256=payload_sha,
            ).first()
            if previous is not None:
                previous.seen_count += 1
                previous.last_seen_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.commit()
                stats.counts["already_seen"] += 1
                continue

            decision = classify_intake(article, now=now, max_age_days=max_age_days)
            match = index.find(article) if decision.status == "accepted" else None
            news = None
            status, reason, stage = decision.status, decision.reason, decision.stage
            if match is not None:
                status, reason, stage = "duplicate", match.reason, "exact_duplicate"
            elif decision.status == "accepted":
                news = News(
                    source_id=source.id, title=article.title, text=article.text,
                    url=article.url, published_at=article.published_at,
                    cluster_id=None, is_primary=False,
                )
                session.add(news)
                session.flush()

            record = NewsIntakeRecord(
                source_id=source.id,
                news_id=news.id if news else None,
                duplicate_of_news_id=match.news_id if match else None,
                raw_url=raw_url, raw_title=str(getattr(article, "title", "") or ""),
                raw_text=str(getattr(article, "text", "") or ""),
                raw_published_at=getattr(article, "published_at", None),
                text_source=getattr(article, "text_source", None),
                canonical_url=canonical, url_sha256=url_sha,
                payload_sha256=payload_sha,
                status=status, reason=reason, decision_stage=stage,
            )
            session.add(record)
            session.commit()
            if news is not None:
                index.add(news)
                stats.saved += 1
                _store_image_after_commit(session, article, news)
            stats.counts[status] += 1
            stats.counts[reason] += 1
        except IntegrityError as exc:
            session.rollback()
            # Concurrent write/unique URL collision cannot silently lose the
            # source; record the race and retry it on the next run.
            stats.counts["db_conflict"] += 1
            stats.errors.append("db_conflict:" + str(getattr(article, "url", "")))
        except Exception as exc:
            session.rollback()
            stats.counts["db_error"] += 1
            stats.errors.append(
                f"{getattr(article, 'url', '')}: {type(exc).__name__}: {exc}"
            )
    return stats
