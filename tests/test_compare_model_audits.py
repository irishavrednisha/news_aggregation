"""Offline contracts for the exploratory Qwen/SBERT review queue."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_model_audits import compare, write_csv


def data():
    rows = [
        ("1", "На заводе запустили линию", ["заводы"], ["заводы"], True, ["заводы"]),
        ("2", "На заводе провели концерт", ["заводы"], [], False, []),
        ("3", "Построили судно", [], ["судостроение"], True, ["судостроение"]),
        ("4", "Компания нарушила закон", [], [], False, []),
        ("5", "На верфи испытали катер", [], [], None, []),
    ]
    sr = []
    hr = []
    qr = []
    br = []
    for url, title, st, ht, relevant, qt in rows:
        uri = "https://example.org/" + url
        sr.append({"url": uri, "title": title, "tags": st, "source": "Источник"})
        hr.append({"url": uri, "title": title, "tags": ht, "source": "Источник"})
        qr.append({"url": uri, "title": title, "tags": qt,
                   "relevant": relevant, "source": "Источник",
                   "error": "failed" if relevant is None else None})
        br.append({
            "url": uri, "title": title, "source": "Источник",
            "unverified_tls": url == "3",
            "primary_filter_accepted": url not in ("3", "4"),
            "primary_filter_reason": (
                "foreign_without_russian_connection" if url == "3"
                else "offtopic" if url == "4" else "accepted"
            ),
        })
    return (
        {"tagger_mode": "legacy", "threshold": 0.35, "max_tags": 2, "results": sr},
        {"tagger_mode": "hybrid", "threshold": 0.35, "max_tags": 2, "results": hr},
        {"tags_only": True, "results": qr},
        {"articles": br},
    )


class ModelComparisonTests(unittest.TestCase):
    def test_joins_by_url_and_never_creates_gold_or_publication(self):
        s, h, q, b = data()
        q["results"].reverse()
        report = compare(s, h, q, b)
        summary = report["summary"]
        self.assertEqual(summary["articles"], 5)
        self.assertEqual(summary["qwen_positive"], 2)
        self.assertEqual(summary["qwen_negative"], 2)
        self.assertEqual(summary["qwen_abstained"], 1)
        self.assertEqual(summary["binary_model_disagreement"], 2)
        self.assertFalse(report["publishing_performed"])
        self.assertFalse(report["gold_labels_provided"])
        self.assertEqual(report["human_verified_count"], 0)
        by_url = {x["url"]: x for x in report["results"]}
        self.assertEqual(by_url["https://example.org/1"]["priority"], "medium")
        self.assertEqual(by_url["https://example.org/2"]["priority"], "high")
        self.assertEqual(by_url["https://example.org/3"]["priority"], "high")
        self.assertIn(
            "foreign_scope_conflict", by_url["https://example.org/3"]["reasons"]
        )
        self.assertIn(
            "source_tls_unverified", by_url["https://example.org/3"]["reasons"]
        )
        self.assertEqual(by_url["https://example.org/4"]["priority"], "low")
        self.assertIn(
            "qwen_abstained_or_failed", by_url["https://example.org/5"]["reasons"]
        )
        for row in report["results"]:
            self.assertEqual(row["human_review_status"], "draft")
            self.assertEqual(row["human_tags"], "")

    def test_refuses_mismatched_or_duplicate_urls_and_parameters(self):
        for case in ("different_url", "duplicate_url", "bad_threshold", "bad_qwen"):
            s, h, q, b = data()
            if case == "different_url":
                q["results"][0]["url"] = "https://example.org/unknown"
            if case == "duplicate_url":
                h["results"][0]["url"] = h["results"][1]["url"]
            if case == "bad_threshold":
                h["threshold"] = 0.5
            if case == "bad_qwen":
                q["tags_only"] = False
            with self.subTest(case=case), self.assertRaises(ValueError):
                compare(s, h, q, b)

    def test_unresolved_is_not_negative_and_csv_is_editable(self):
        report = compare(*data())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            write_csv(report, path)
            self.assertIn("human_review_status", path.read_text(encoding="utf-8-sig"))
            self.assertIn("qwen_abstained_or_failed", path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
