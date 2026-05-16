import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import time
from pathlib import Path
from html import escape

from playwright.sync_api import sync_playwright

from database.db import get_session
from database.models import Cluster, News


CHANNEL_URL = "https://web.max.ru/-74750848532727"

PROFILE_DIR = "max_browser_profile"


def get_primary_news(session, cluster_id: int) -> News | None:
    return (
        session.query(News)
        .filter(News.cluster_id == cluster_id)
        .order_by(
            News.is_primary.desc(),
            News.published_at.asc().nullslast(),
            News.collected_at.asc()
        )
        .first()
    )


def format_cluster_post(session, cluster: Cluster) -> str:
    primary_news = get_primary_news(session, cluster.id)

    title = escape(cluster.title or "Новость промышленности")
    summary = escape(cluster.summary_text or "")
    tags = cluster.tags or ""

    url = primary_news.url if primary_news else ""

    hashtags = ""

    if tags:
        hashtags = " ".join(
            "#" + tag.strip().replace(" ", "_").replace("-", "_")
            for tag in tags.split(",")
            if tag.strip()
        )

    parts = [
        title,
        "",
        summary,
    ]

    if url:
        parts.extend([
            "",
            f"Источник: {escape(url)}"
        ])

    if hashtags:
        parts.extend([
            "",
            escape(hashtags)
        ])

    return "\n".join(parts).strip()

def get_ready_clusters(session, limit: int = 100) -> list[Cluster]:
    return (
        session.query(Cluster)
        .filter(Cluster.status == "ready")
        .filter(Cluster.summary_text.isnot(None))
        .order_by(Cluster.created_at.asc())
        .limit(limit)
        .all()
    )


def publish_post(page, title: str, body: str):
    input_box = page.locator("p.paragraph").last
    input_box.click()

    # Включаем жирный
    page.keyboard.press("Control+B")
    page.keyboard.insert_text(title)
    page.keyboard.press("Control+B")

    # Обычный текст
    page.keyboard.down("Shift")
    page.keyboard.press("Enter")
    page.keyboard.press("Enter")
    page.keyboard.up("Shift")
    page.keyboard.insert_text(body)

    time.sleep(1)

    send_button = page.locator(
        'button[aria-label="Отправить сообщение"]'
    )

    send_button.click()

    time.sleep(3)


def publish_ready_clusters(limit: int = 100):
    session = get_session()

    try:
        clusters = get_ready_clusters(session, limit=limit)

        print(f"Кластеров ready: {len(clusters)}")

        if not clusters:
            return

        Path(PROFILE_DIR).mkdir(exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(
                user_data_dir=PROFILE_DIR,
                headless=False,
            )

            page = browser.new_page()

            page.goto(CHANNEL_URL)

            print("Ожидание загрузки MAX...")
            time.sleep(10)

            for cluster in clusters:
                print("-" * 100)
                print(f"Публикация кластера #{cluster.id}")
                print(cluster.title)

                title, body = build_post_parts(session, cluster)

                try:
                    publish_post(page, title, body)

                    cluster.status = "published"

                    session.commit()

                    print("Опубликовано.")

                except Exception as error:
                    print("Ошибка публикации:")
                    print(error)

                time.sleep(5)

            browser.close()

    finally:
        session.close()


def build_post_parts(session, cluster: Cluster) -> tuple[str, str]:
    primary_news = get_primary_news(session, cluster.id)

    title = cluster.title or "Новость промышленности"
    summary = cluster.summary_text or ""
    tags = cluster.tags or ""
    url = primary_news.url if primary_news else ""

    hashtags = ""

    if tags:
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

    body = "\n".join(body_parts).strip()

    return title, body


if __name__ == "__main__":
    publish_ready_clusters(limit=100)