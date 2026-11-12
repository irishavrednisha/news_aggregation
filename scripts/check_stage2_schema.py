"""Read-only report of existing and stage-2 tables; does not mutate PostgreSQL."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow both "python scripts/check_stage2_schema.py" and "python -m scripts.check_stage2_schema".
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import inspect, text

from database.db import engine

LEGACY = ("sources", "news", "clusters", "posts")
STAGE_2 = ("news_images", "moderation_items", "moderation_actions", "publication_attempts")


def main() -> None:
    with engine.connect() as connection:
        existing = set(inspect(connection).get_table_names())
        print("Исходные таблицы:")
        for table in LEGACY:
            if table not in existing:
                print(f"  {table}: ОТСУТСТВУЕТ")
                continue
            # Fixed, hard-coded table names only; no user input in SQL.
            count = connection.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()
            print(f"  {table}: {count} строк")
        print("Таблицы модерации:")
        for table in STAGE_2:
            print(f"  {table}: {'OK' if table in existing else 'ещё не создана'}")
        print(f"Alembic: {'OK' if 'alembic_version' in existing else 'не применялся'}")


if __name__ == "__main__":
    main()
