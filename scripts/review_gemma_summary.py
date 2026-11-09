"""Manually check and approve one local Gemma draft for MAX.

Run:
    .\.venv\Scripts\python.exe scripts\review_gemma_summary.py --cluster-id 123

Compare each factual claim with the displayed original source texts.
Only an explicit typed confirmation makes a reviewed, tagged draft ready.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import get_session
from database.models import Cluster, News


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-id", type=int, required=True)
    args = parser.parse_args()
    session = get_session()
    try:
        cluster = session.query(Cluster).filter(Cluster.id == args.cluster_id).first()
        if not cluster or cluster.status != "review_required":
            parser.error("Кластер не найден или не ожидает ручной проверки")
        articles = (
            session.query(News)
            .filter(News.cluster_id == cluster.id)
            .order_by(News.is_primary.desc())
            .all()
        )
        print(f"Кластер #{cluster.id}: {cluster.title}")
        print(f"Темы: {cluster.tags or '(ещё не определены)'}")
        print(f"\nСводка Gemma:\n{cluster.summary_text or '(пусто)'}\n")
        for item in articles:
            print(f"\nИсточник: {item.url}")
            print(f"Заголовок: {item.title}")
            print(f"Текст:\n{item.text or '(пусто)'}")
        if not cluster.tags or not cluster.tags.strip() or not cluster.summary_text:
            print("Подтверждение невозможно: отсутствуют теги или сводка.")
            return
        print("\nПроверь смысл, отрицания, планы и завершённые действия, "
              "все цифры, организации и технические подробности по первоисточникам.")
        answer = input("Чтобы разрешить публикацию, введи 'ПОДТВЕРЖДАЮ': ").strip()
        if answer != "ПОДТВЕРЖДАЮ":
            print("Статус не изменён; публикация заблокирована.")
            return
        cluster.status = "ready"
        session.commit()
        print("Сводка одобрена. Кластер готов к обычной публикации.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
