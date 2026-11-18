"""Read-only before/after migration count and schema audit.

Use before migration:
  python scripts/check_stage2_intake_schema.py --allow-missing --output collector_test_results/stage2_before.json
Use immediately after migration, BEFORE running collection:
  python scripts/check_stage2_intake_schema.py --baseline collector_test_results/stage2_before.json --output collector_test_results/stage2_after.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect, text
from database.db import get_session

LEGACY = ("sources", "news", "clusters", "posts")
INTAKE = "news_intake_records"
REQUIRED_COLUMNS = {
    "id", "source_id", "news_id", "duplicate_of_news_id", "raw_url",
    "raw_title", "raw_text", "raw_published_at", "canonical_url",
    "url_sha256", "payload_sha256", "status", "reason", "decision_stage",
    "seen_count", "first_seen_at", "last_seen_at", "text_source",
}


def inspect_schema(session):
    tables = set(inspect(session.get_bind()).get_table_names())
    missing = set(LEGACY) - tables
    if missing:
        raise RuntimeError("Wrong database; missing legacy tables: " + ", ".join(sorted(missing)))
    old_counts = {
        table: session.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
        for table in LEGACY
    }
    installed = INTAKE in tables
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True, "legacy_counts": old_counts,
        "intake_schema_installed": installed,
        "missing_intake_columns": [],
        "intake_total": None,
        "intake_statuses": None,
    }
    if installed:
        have = {column["name"] for column in inspect(session.get_bind()).get_columns(INTAKE)}
        report["missing_intake_columns"] = sorted(REQUIRED_COLUMNS - have)
        report["intake_total"] = session.execute(
            text("SELECT count(*) FROM news_intake_records")
        ).scalar_one()
        report["intake_statuses"] = dict(session.execute(
            text("SELECT status, count(*) FROM news_intake_records GROUP BY status")
        ).all())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    session = get_session()
    try:
        report = inspect_schema(session)
    finally:
        session.close()
    issues = []
    if not report["intake_schema_installed"] and not args.allow_missing:
        issues.append("migration_not_applied")
    if report["missing_intake_columns"]:
        issues.append("incomplete_intake_schema")
    if args.baseline:
        original = json.loads(args.baseline.read_text(encoding="utf-8"))
        for name in LEGACY:
            if original["legacy_counts"][name] != report["legacy_counts"][name]:
                issues.append(f"legacy_count_changed:{name}")
    report["issues"] = issues
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
