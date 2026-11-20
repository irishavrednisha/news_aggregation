"""Offline one-shot audit runner tests: no model, SQL, network or MAX."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from processing.qwen_abstractive_summary import canonical_materials, source_digest
from scripts.stage4_audit_once import run_audit


class OneShotAuditTests(unittest.TestCase):
    def fixture(self):
        docs = canonical_materials([{
            "id": "s1", "title": "Завод провёл испытания оборудования",
            "source": "Тест", "url": "https://example.org/1",
            "text": ("На заводе завершились испытания оборудования. "
                     "Результаты были переданы инженерной службе. "
                     "Новая установка работает на производственном участке."),
            "published_at": "2026-11-20", "is_primary": True,
        }])
        return {
            "stage": 4, "created_utc": "2026-11-20T00:00:00Z",
            "clusters": [{
                "cluster_id": 1, "source_materials": docs,
                "source_sha256": source_digest(docs),
                "qwen_summary": "", "attempts": [],
                "validation": {"issues": ["no_generated_summary"]},
            }],
        }

    def test_atomic_json_contains_offline_guards_and_no_publication(self):
        with TemporaryDirectory() as root:
            base = Path(root)
            incoming = base / "original.json"
            outgoing = base / "audit.json"
            incoming.write_text(json.dumps(self.fixture(), ensure_ascii=False),
                                encoding="utf-8")
            immutable_bytes = incoming.read_bytes()
            result = run_audit(incoming, outgoing, skip_tests=True)
            self.assertTrue(result["successful"])
            self.assertEqual(incoming.read_bytes(), immutable_bytes)
            reread = json.loads(outgoing.read_text(encoding="utf-8"))
            self.assertTrue(reread["source_report_is_immutable"])
            self.assertFalse(reread["ollama_called"])
            self.assertFalse(reread["database_writes"])
            self.assertEqual(reread["max_publications"], 0)
            self.assertFalse(reread["independent_accuracy_measured"])
            self.assertEqual(reread["revalidation"]["metrics"]["rows"], 1)

    def test_source_sha_mismatch_is_recorded_as_failure(self):
        with TemporaryDirectory() as root:
            base = Path(root)
            incoming = base / "bad.json"
            outgoing = base / "error.json"
            report = self.fixture()
            report["clusters"][0]["source_sha256"] = "not the real hash"
            incoming.write_text(json.dumps(report, ensure_ascii=False),
                                encoding="utf-8")
            result = run_audit(incoming, outgoing, skip_tests=True)
            self.assertFalse(result["successful"])
            self.assertIn("Source digest mismatch", result["revalidation_error"])
            self.assertTrue(outgoing.exists())

    def test_never_overwrites_input(self):
        with TemporaryDirectory() as root:
            f = Path(root) / "same.json"
            f.write_text(json.dumps(self.fixture(), ensure_ascii=False),
                         encoding="utf-8")
            with self.assertRaises(ValueError):
                run_audit(f, f, skip_tests=True)


if __name__ == "__main__":
    unittest.main()
