"""Optional integration audit against an isolated PostgreSQL database named stage4_ci.

No tables are created, updated, inserted or deleted. Everything, including
approved posts and immutable source articles, is compared before/after a
Qwen dry-run with a fake request (no Ollama needed).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.compare_stage4_qwen import run


class QwenReadOnlyPostgresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dsn = os.getenv("STAGE4_TEST_URL")
        if not dsn or os.getenv("STAGE4_TEST_WRITE_ACK") != "ephemeral-only":
            raise unittest.SkipTest("Only run on explicit disposable stage4_ci copy")
        from sqlalchemy.engine import make_url
        parsed = make_url(dsn)
        if parsed.get_backend_name() != "postgresql" or parsed.database != "stage4_ci":
            raise RuntimeError("Refusing database other than disposable stage4_ci")
        from sqlalchemy import create_engine
        cls.engine = create_engine(dsn)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            cls.engine.dispose()

    @classmethod
    def snapshot(cls):
        from sqlalchemy import text
        tables = ("news", "clusters", "posts")
        hashes = {}
        with cls.engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            for table in tables:
                rows = conn.execute(text(
                    f"SELECT row_to_json(x)::text FROM "
                    f"(SELECT * FROM {table} ORDER BY id) x"
                )).scalars().all()
                digest = hashlib.sha256()
                for row in rows:
                    digest.update(row.encode("utf-8"))
                    digest.update(b"\n")
                hashes[table] = {"rows": len(rows), "sha256": digest.hexdigest()}
            conn.rollback()
        return hashes

    def test_snapshots_unchanged_by_qwen_runner(self):
        before = self.snapshot()
        article = {
            "id": "postgres-test", "title": "Производственная линия",
            "text": "Завод планирует установить новую линию. " * 7,
            "source": "Тестовый", "url": "https://example.org/read-only",
            "is_primary": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inp, out = root / "source.json", root / "result.json"
            inp.write_text(json.dumps({"stage": 4, "clusters": [{
                "cluster_id": 1, "source_materials": [article],
                "gemma_summary": "Тестовая сводка",
            }]}, ensure_ascii=False), encoding="utf-8")

            def fake_request(messages, schema, *, model, options):
                return {"main_event": "План установки линии",
                        "is_news_event": True, "summary": "",
                        "claims": [], "uncertainties": ["test mode"],
                        "requires_review": True}, {}

            report = run(inp, out, limit=1, measure_gpu=False, request=fake_request)
            self.assertEqual(report["clusters"][0]["status"], "summary_review")
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
