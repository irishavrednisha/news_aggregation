"""CPU-only regression tests for LoRA diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_news_lora import surviving_reply_tokens
from scripts.evaluate_news_lora import select_rows, summarize, f1


class DiagnosticsTests(unittest.TestCase):
    def test_reply_survival(self):
        self.assertEqual(surviving_reply_tokens(1000, 1060, 1024), 24)
        self.assertEqual(surviving_reply_tokens(1100, 1160, 1024), 0)
        self.assertEqual(surviving_reply_tokens(500, 600, 1024), 100)

    def test_stratified_limit_reproducible(self):
        rows = [
            {"task": "fact_tagging", "story_id": "a"},
            {"task": "fact_tagging", "story_id": "b"},
            {"task": "abstractive_summary", "story_id": "a"},
            {"task": "abstractive_summary", "story_id": "c"},
            {"task": "fact_tagging", "story_id": "c"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "validation.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            selected = select_rows(path, 1)
        self.assertEqual(
            [(r["task"], r["story_id"]) for r in selected],
            [("fact_tagging", "a"), ("abstractive_summary", "a")],
        )

    def test_summary_reports_failures_and_skips(self):
        rows = [
            {"task": "fact_tagging", "grade": {
                "valid_json": True, "valid_contract": True,
                "relevant_correct": True, "tags_exact": True,
            }},
            {"task": "abstractive_summary", "skip_reason": "too_long"},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["skipped_long_prompt"], 1)
        self.assertEqual(summary["relevance_correct"], 1)
        self.assertEqual(summary["summary_valid_contract"], 0)
        self.assertEqual(f1(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
