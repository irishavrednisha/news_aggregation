"""No network, DB, Ollama or publish: verified-gold comparison contract."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_real_tagging import (
    compare, read_audit, read_verified_gold, score,
)


def make_gold(url: str, relevant: bool, tags: list[str], split: str = "validation"):
    return {
        "url": url,
        "source": "Источник",
        "title": "Промышленная новость",
        "split": split,
        "review_status": "verified",
        "origin": "real",
        "relevant": relevant,
        "tags": tags,
    }


class VerifiedComparisonTests(unittest.TestCase):
    def test_holdout_rejects_drafts_and_synthetic(self):
        one = make_gold("https://test.local/1", True, ["машиностроение"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gold.json"
            for changed in ({"origin": "synthetic"}, {"review_status": "draft"}):
                path.write_text(json.dumps({"examples": [{**one, **changed}]}),
                                encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "не подтверждённую"):
                    read_verified_gold(path)
            path.write_text(json.dumps({"examples": [one, {**one}]}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "повторяющийся"):
                read_verified_gold(path)

    def test_split_is_fixed_and_never_train(self):
        a = make_gold("https://test.local/a", True, ["машиностроение"])
        b = make_gold("https://test.local/b", False, [], split="test")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gold.json"
            path.write_text(json.dumps({"examples": [a, b]}), encoding="utf-8")
            self.assertEqual(set(read_verified_gold(path)), {a["url"]})
            self.assertEqual(set(read_verified_gold(path, "test")), {b["url"]})
            self.assertEqual(len(read_verified_gold(path, "all")), 2)
            path.write_text(json.dumps({"examples": [{**a, "split": "train"}]}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "фиксированный"):
                read_verified_gold(path)

    def test_read_audit_rejects_duplicate_urls_and_published(self):
        path_row = {"url": "https://test.local/a", "tags": ["машиностроение"]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.json"
            path.write_text(json.dumps({
                "read_only": True, "publishing_performed": False,
                "results": [path_row],
            }), encoding="utf-8")
            self.assertEqual(read_audit(path)[path_row["url"]], path_row)
            path.write_text(json.dumps({
                "read_only": True, "results": [path_row, path_row],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "повторяющийся"):
                read_audit(path)
            path.write_text(json.dumps({
                "read_only": True, "publishing_performed": True,
                "results": [path_row],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "публикацией"):
                read_audit(path)

    def test_coverage_and_shared_scoring_do_not_treat_errors_as_negative(self):
        a = make_gold("https://test.local/a", True, ["машиностроение"])
        b = make_gold("https://test.local/b", False, [])
        c = make_gold("https://test.local/c", True, ["химическая промышленность"])
        gold = {row["url"]: row for row in (a, b, c)}
        baseline = {
            a["url"]: {"tags": ["машиностроение"]},
            b["url"]: {"tags": []},
            c["url"]: {"tags": ["атомная отрасль"]},
        }
        qwen = {
            a["url"]: {"tags": ["машиностроение"], "relevant": True},
            b["url"]: {"tags": [], "relevant": None,
                       "error": "invalid quote", "warnings": ["inference_or_validation_failed"]},
            c["url"]: {"tags": ["химическая промышленность"], "relevant": True},
        }
        out = compare(gold, {"legacy": baseline, "qwen": qwen})
        self.assertEqual(out["shared_resolved_articles"], 2)
        self.assertEqual(out["audits"]["qwen"]["coverage"], 0.6667)
        self.assertEqual(out["audits"]["legacy"]["coverage"], 1.0)
        self.assertEqual(out["audits"]["qwen"]["metrics_on_resolved"]["micro_f1"], 1.0)
        self.assertEqual(out["audits"]["legacy"]["metrics_on_shared_resolved"]["micro_f1"], 0.5)
        self.assertEqual(out["audits"]["qwen"]["unresolved_reasons"],
                         {"inference_or_validation_failed": 1})
        self.assertEqual(out["audits"]["qwen"]["metrics_on_resolved"]["relevance_scored"], 2)
        self.assertIsNone(out["audits"]["legacy"]["metrics_on_resolved"]["relevance_accuracy"])

    def test_abstentions_never_inflate_exact_match(self):
        gold = {
            "https://test.local/a": make_gold("https://test.local/a", False, []),
        }
        audit = {
            "https://test.local/a": {
                "tags": [], "relevant": None,
                "warnings": ["skipped:unsafe_source"],
            },
        }
        res = compare(gold, {"qwen": audit})
        self.assertEqual(res["audits"]["qwen"]["resolved"], 0)
        self.assertIsNone(res["audits"]["qwen"]["metrics_on_resolved"]["micro_f1"])
        self.assertIsNone(res["audits"]["qwen"]["metrics_on_resolved"]["exact_match"])


if __name__ == "__main__":
    unittest.main()
