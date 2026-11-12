"""MAX publication adapter with mandatory, fail-closed moderation.

IMPORTANT:
- Cluster.status == "ready" is not permission to publish.
- Only Post.status == "approved" with a frozen post_text snapshot is eligible.
- MAX_PUBLISHING_ENABLED is false by default.
- Until reliable delivery confirmation exists, an attempted send is marked
  "uncertain" and is NEVER automatically retried (prevents duplicate posts).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from database.db import get_session
from database.models import Cluster, News, Post
from publishing.moderation_gate import (
    is_approved_post,
    max_publishing_enabled,
    split_post_snapshot,
)


CHANNEL_URL = os.getenv("MAX_CHANNEL_URL", "https://web.max.ru/-74750848532727")
PROFILE_DIR = os.getenv("MAX_PROFILE_DIR", "max_browser_profile")


def get_primary_news(session, cluster_id: int) -> News | None:
    return (
        session.query(News)
        .filter(News.cluster_id == cluster_id)
        .order_by(
            News.is_primary.desc(),
            News.published_at.asc().nullslast(),
            News.collected_at.asc(),
        )
        .first()
    )


def get_approved_posts(session, limit: int = 100) -> list[Post]:
    """Select Post snapshots, NOT ready clusters. Invalid rows are skipped."""
    candidates = (
        session.query(Post)
        .filter(Post.status == "approved")
        .filter(Post.post_text.isnot(None))
        .order_by(Post.id.asc())
        .limit(limit)
        .all()
    )
    return [post for post in candidates if is_approved_post(post)]


def _claim_post(session, post_id: int) -> bool:
    """Reserve an approved post atomically, so two workers cannot both send it."""
    updated = (
        session.query(Post)
        .filter(Post.id == post_id, Post.status == "approved")
        .update({Post.status: "sending"}, synchronize_session=False)
    )
    session.commit()
    return updated == 1


def _send_to_max(page, title: str, body: str) -> None:
    """Low-level browser operation. Invoke ONLY from publish_approved_posts."""
    input_box = page.locator("p.paragraph").last
    input_box.click()

    page.keyboard.press("Control+B")
    page.keyboard.insert_text(title)
    page.keyboard.press("Control+B")

    page.keyboard.down("Shift")
    page.keyboard.press("Enter")
    page.keyboard.press("Enter")
    page.keyboard.up("Shift")
    page.keyboard.insert_text(body)

    time.sleep(1)
    page.locator('button[aria-label="Отправить сообщение"]').click()

    # A click is NOT proof that MAX accepted the message. Never call it
    # "published" without a verified message identifier/confirmation.


def publish_approved_posts(limit: int = 100, *, dry_run: bool = False) -> int:
    """Attempt only approved snapshots, or preview safely with dry_run.

    No automation should retry a "sending" or "uncertain" row. Those states
    need reconciliation before any resend. This is intentional for safety.
    """
    if not dry_run and not max_publishing_enabled():
        print("MAX отключён (MAX_PUBLISHING_ENABLED=false). Ничего не отправлено.")
        return 0

    session = get_session()
    try:
        posts = get_approved_posts(session, limit=limit)
        print(f"Одобренных публикаций с полным текстом: {len(posts)}")

        if dry_run:
            for post in posts:
                title, body = split_post_snapshot(post.post_text)
                print(f"[DRY RUN] Post #{post.id}, cluster #{post.cluster_id}: {title}")
                print(body[:250])
            print("Тестовый режим: отправки в MAX и изменения БД не выполнялись.")
            return len(posts)

        if not posts:
            return 0

        Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
        attempted = 0

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch_persistent_context(
                user_data_dir=PROFILE_DIR,
                headless=False,
            )
            try:
                page = browser.new_page()
                page.goto(CHANNEL_URL)
                print("Ожидание загрузки MAX...")
                time.sleep(10)

                for post in posts:
                    # Use the approved snapshot, never a mutable Cluster summary.
                    title, body = split_post_snapshot(post.post_text)
                    post_id = post.id
                    if not _claim_post(session, post_id):
                        print(f"Post #{post_id}: уже зарезервирован другим процессом.")
                        continue
                    attempted += 1

                    try:
                        _send_to_max(page, title, body)
                    except Exception as error:
                        session.rollback()
                        print(f"Post #{post_id}: ошибка отправки: {error}")
                    finally:
                        # Safe even if click succeeded but a later step failed.
                        # Do not auto retry; delivery verification is a later stage.
                        post_row = session.get(Post, post_id)
                        if post_row is not None and post_row.status == "sending":
                            post_row.status = "uncertain"
                            session.commit()
                            print(
                                f"Post #{post_id}: статус uncertain; "
                                "проверьте доставку вручную."
                            )

                    time.sleep(5)
            finally:
                browser.close()

        return attempted
    finally:
        session.close()


def publish_ready_clusters(limit: int = 100, *, dry_run: bool = False) -> int:
    """Deprecated entry point; STILL requires approved Post + explicit opt-in."""
    return publish_approved_posts(limit=limit, dry_run=dry_run)


def build_post_parts(session, cluster: Cluster) -> tuple[str, str]:
    """Preview helper; output from this function is NOT approved for sending."""
    primary_news = get_primary_news(session, cluster.id)
    title = cluster.title or "Новость промышленности"
    summary = cluster.summary_text or ""
    tags = cluster.tags or ""
    url = primary_news.url if primary_news else ""

    hashtags = " ".join(
        "#" + tag.strip().replace(" ", "_").replace("-", "_")
        for tag in tags.split(",")
        if tag.strip()
    )
    body_parts = [summary]
    if url:
        body_parts.extend(["", f"Источник: {url}"])
    if hashtags:
        body_parts.extend(["", hashtags])
    return title, "\n".join(body_parts).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="MAX: approved posts only.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview approved post snapshots without browser/DB writes.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    publish_approved_posts(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
