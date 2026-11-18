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


def collect(limit: int = 10) -> int:
    from collectors.data_collector import collect_news
    from database.db import get_session
    from database.repository import get_active_sources, save_articles
    from processing.topic_filter import filter_articles

    session = get_session()
    diagnostics = {}
    saved = 0
    try:
        sources = get_active_sources(session)
        LOG.info("Collecting from %s active sources (database is authoritative)", len(sources))
        for source in sources:
            try:
                # Per-source isolation: one failed website cannot interrupt others.
                articles = collect_news(
                    [source], limit_per_source=limit, session=session,
                    diagnostics=diagnostics, allow_unverified_tls=False,
                )
                filtered = filter_articles(articles)
                saved_now = save_articles(session, filtered)
                saved += saved_now
                LOG.info("%s: candidates=%s articles=%s filtered=%s saved=%s status=%s",
                         source.name, diagnostics[source.name]["candidates"],
                         len(articles), len(filtered), saved_now,
                         diagnostics[source.name]["status"])
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Independent news pipeline workers")
    parser.add_argument(
        "stage", nargs="?", default="collect",
        choices=("collect", "process", "prepare", "telegram", "max"),
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
    elif args.stage == "telegram":
        from moderation.bot import main as run_bot
        run_bot()
    elif args.stage == "max":
        # Hard stop for this stabilization stage, even if the local env is misconfigured.
        raise SystemExit("Stage 1: MAX publication is disabled. No requests sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
