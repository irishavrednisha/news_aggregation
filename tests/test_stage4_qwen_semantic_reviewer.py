"""No GPU/network/SQL tests for optional independent local semantic referee."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from processing.qwen_abstractive_summary import canonical_materials, source_digest
from processing.qwen_semantic_reviewer import review_cluster, FORMAT, REVIEW_MODEL
from scripts.stage4_semantic_audit import run


def row():
    materials = canonical_materials([{
        "id": "s1", "source": "Тест", "url": "https://example.org/a",
        "title": "Завод начал выпуск промышленных насосов",
        "text": ("Завод начал выпуск промышленных насосов для добычи. "
                 "Первые изделия переданы заказчику. "
                 "Мощность новой установки составляет 20 кВт."),
        "published_at": "2026-11-20", "is_primary": True
    }])
    return {
        "cluster_id": 100, "source_materials": materials,
        "source_sha256": source_digest(materials),
        "qwen_summary": ("Завод запустил выпуск насосов для добычи. "
                         "Первые изделия переданы заказчику. "
                         "Мощность установки составляет 20 кВт."),
        "main_event": "Выпуск насосов",
    }


class FakeReviewer:
    def __init__(self, verdict="supported", language=False, invalid=False):
        self.calls = 0
        self.verdict = verdict
        self.language = language
        self.invalid = invalid

    def __call__(self, messages, schema, *, model, options):
        self.calls += 1
        assert model == REVIEW_MODEL
        assert "claims" in schema["properties"]
        incoming = json.loads(messages[-1]["content"])
        claims = [{
            "index": entry["index"], "verdict": self.verdict,
            "reason": "Краткое объяснение на основе цитаты",
            "quote_indices": [0] if entry["quotes"] else [],
        } for entry in incoming["claims"]]
        if self.invalid:
            claims[0]["quote_indices"] = [900]
        return {"claims": claims,
                "language": {"needs_edit": self.language, "notes": []}}, {
                    "eval_count": 100, "tokens_per_second": 50,
                }


class SemanticRefereeTests(unittest.TestCase):
    def test_valid_verdict_still_never_confirms_accuracy_or_publication(self):
        fake = FakeReviewer()
        result = review_cluster(row(), request=fake)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(result["status"], "advisory_no_issues_detected")
        self.assertFalse(result["semantic_accuracy_certified"])
        self.assertFalse(result["publication_allowed"])
        self.assertTrue(result["human_approval_required"])

    def test_contradiction_always_requires_editor(self):
        result = review_cluster(row(), request=FakeReviewer(
            verdict="contradicted", language=True))
        self.assertEqual(result["status"], "review_required")
        self.assertIn("claim_1_semantic_contradicted", result["issues"])
        self.assertIn("model_flagged_russian_language", result["issues"])

    def test_unverifiable_claim_requires_editor(self):
        result = review_cluster(row(), request=FakeReviewer(
            verdict="insufficient"))
        self.assertIn("claim_1_semantic_insufficient", result["issues"])

    def test_invalid_reference_never_accepted(self):
        result = review_cluster(row(), request=FakeReviewer(invalid=True))
        self.assertIn("invalid_claim_1_verdict", result["issues"])
        self.assertIsNone(result["answer"])

    def test_digest_mismatch_blocks_review(self):
        altered = row()
        altered["source_sha256"] = "garbled"
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            review_cluster(altered, request=FakeReviewer())

    def test_missing_summary_does_not_call_model(self):
        empty = row()
        empty["qwen_summary"] = ""
        fake = FakeReviewer()
        result = review_cluster(empty, request=fake)
        self.assertEqual(fake.calls, 0)
        self.assertIn("no_summary", result["issues"])

    def test_one_command_report_checkpoints_and_resume_without_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            incoming, out = base / "input.json", base / "result.json"
            input_document = {"stage": 4, "clusters": [row()]}
            incoming.write_text(json.dumps(input_document, ensure_ascii=False),
                                encoding="utf-8")
            fake = FakeReviewer()
            first = run(incoming, out, request=fake)
            self.assertTrue(first["complete"])
            self.assertEqual(fake.calls, 1)
            self.assertTrue(out.is_file())
            second = run(incoming, out, request=fake, resume=True)
            self.assertEqual(fake.calls, 1)
            self.assertTrue(second["complete"])
            self.assertFalse(second["auto_publication"])
            self.assertFalse(second["editorial_accuracy_measured"])

    def test_connection_failure_checkpoints_and_stops(self):
        class Down:
            def __call__(self, *args, **kwargs):
                raise ConnectionError("Ollama offline")
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            incoming, out = base / "input.json", base / "out.json"
            incoming.write_text(
                json.dumps({"stage": 4, "clusters": [row(), dict(row(), cluster_id=101)]},
                           ensure_ascii=False), encoding="utf-8")
            report = run(incoming, out, request=Down())
            self.assertFalse(report["complete"])
            self.assertEqual(report["stopped_reason"], "local_reviewer_unavailable")
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(report["metrics"]["processed"], 1)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
