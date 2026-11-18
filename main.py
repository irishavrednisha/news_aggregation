"""Independent entry points for the five pipeline roles.

Default invocation is collection only. Neither import nor the collect subcommand
imports ML libraries, creates tables, changes configured sources, or publishes.
Run each role in its own OS process (Windows collector, optional WSL ML worker).
"""
from __future__ import annotations

import argparse
import logging
import os

LOG = logging.getLogger("news_pipeline")

# The operator explicitly accepts a browser-style certificate warning for
# these three official sources. No other sources or HTTPS requests inherit it.
CERT_WARNING_SOURCES = frozenset({"Ростех", "Роснефть", "Газпром"})


def _browser_cert_warning_enabled() -> bool:
    """Set NEWS_BROWSER_CERT_WARNING_FALLBACK=0 to opt out immediately."""
    return os.getenv("NEWS_BROWSER_CERT_WARNING_FALLBACK", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def collect(limit: int = 10) -> int:
    from collectors.data_collector import collect_news
    from database.db import get_session
    from database.repository import get_active_sources
    from database.intake_repository import require_intake_schema, save_intake_batch

    session = get_session()
    diagnostics = {}
    saved = 0
    try:
        require_intake_schema(session)
        sources = get_active_sources(session)
        LOG.info("Collecting from %s active sources (database is authoritative)", len(sources))
        for source in sources:
            try:
                # Per-source isolation: one failed website cannot interrupt others.
                articles = collect_news(
                    [source], limit_per_source=limit, session=None,
                    diagnostics=diagnostics, apply_filters=False,
                    allow_unverified_tls=(
                        _browser_cert_warning_enabled()
                        and source.name in CERT_WARNING_SOURCES
                    ),
                )
                intake = save_intake_batch(session, articles)
                saved_now = intake.saved
                saved += saved_now
                stat = diagnostics.get(source.name, {})
                if stat.get("unverified_tls_urls"):
                    LOG.warning(
                        "%s: Chromium accepted an untrusted certificate for %d "
                        "official-site URLs (source-specific browser fallback); "
                        "MAX publication remains disabled",
                        source.name, len(stat["unverified_tls_urls"]),
                    )
                LOG.info("%s: candidates=%s raw=%s triage=%s saved=%s status=%s",
                         source.name, stat.get("candidates", "?"), len(articles),
                         intake.as_dict(), saved_now, stat.get("status", "unknown"))
            except Exception:
                session.rollback()
                LOG.exception("Source %s failed; continuing", source.name)
        LOG.info("Collection complete: %s new articles; ML not imported; MAX disabled", saved)
        return saved
    finally:
        session.close()


def process(hours: int = 48) -> None:
    # The only role that imports PyTorch / scikit-learn-dependent modules.
    from database.db import get_session
    from processing.clustering import cluster_recent_news
    from processing.summarizer import summarize_new_clusters
    from processing.tagger import tag_new_clusters

    session = get_session()
    try:
        cluster_recent_news(session=session, hours=hours, threshold=0.7)
        summarize_new_clusters(session=session, sentence_count=4, max_chars=800)
        tag_new_clusters(session=session, min_score=0.35)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def prepare(limit: int = 30) -> int:
    # Draft snapshots require migration 20261112_01 and do NOT send to MAX.
    from database.db import get_session
    from moderation.service import create_pending_items

    session = get_session()
    try:
        count = create_pending_items(session, limit=limit)
        LOG.info("Created %s moderation drafts; nothing published", count)
        return count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def status() -> dict:
    """Read existing PostgreSQL work states; no new schema or ML imports."""
    from sqlalchemy import func
    from database.db import get_session
    from database.models import Source, News, Cluster, Post
    from database.moderation_models import ModerationItem

    session = get_session()
    try:
        stats = {
            "sources_total": session.query(Source).count(),
            "sources_active": session.query(Source).filter(Source.is_active.is_(True)).count(),
            "news_awaiting_model": session.query(News).filter(News.cluster_id.is_(None)).count(),
            "cluster_statuses": dict(session.query(Cluster.status, func.count(Cluster.id)).group_by(Cluster.status).all()),
            "post_statuses": dict(session.query(Post.status, func.count(Post.id)).group_by(Post.status).all()),
            "moderation_statuses": dict(session.query(ModerationItem.status, func.count(ModerationItem.id)).group_by(ModerationItem.status).all()),
        }
        import json
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return stats
    finally:
        session.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Independent news pipeline workers")
    parser.add_argument(
        "stage", nargs="?", default="collect",
        choices=("collect", "process", "prepare", "telegram", "max", "status"),
        help="Each stage is a separate process; collect is the safe default",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--hours", type=int, default=48)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.hours < 1:
        parser.error("--limit and --hours must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "collect":
        collect(args.limit)
    elif args.stage == "process":
        process(args.hours)
    elif args.stage == "prepare":
        prepare(args.limit)
    elif args.stage == "status":
        status()
    elif args.stage == "telegram":
        from moderation.bot import main as run_bot
        run_bot()
    elif args.stage == "max":
        # Hard stop for this stabilization stage, even if the local env is misconfigured.
        raise SystemExit("Stage 1: MAX publication is disabled. No requests sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
