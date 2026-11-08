"""Regression tests for offline tagging evaluation and threshold tuning."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluate_tagging import (
    read_references, labeled_rows, metrics, optimize, threshold_tags,
)


class TaggingEvaluationTests(unittest.TestCase):
    def test_micro_and_exact_match_and_negative_false_tags(self):
        samples = [
            ({"tags": ["судостроение"]},
             {"url": "https://ex.test/1", "expected_tags": ["судостроение"]}),
            ({"tags": ["металлургия"]},
             {"url": "https://ex.test/2", "expected_tags": []}),
            ({"tags": []},
             {"url": "https://ex.test/3", "expected_tags": ["атомная отрасль"]}),
            ({"tags": []},
             {"url": "https://ex.test/4", "expected_tags": []}),
        ]
        result = metrics(samples)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["micro_f1"], 0.5)
        self.assertEqual(result["exact_match"], 0.5)
        self.assertEqual(result["false_tag_rate_on_untagged"], 0.5)
        self.assertEqual(len(result["errors"]), 2)

    def test_drafts_excluded_unless_opted_in(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["url", "expected_tags", "review_status"],
                )
                writer.writeheader()
                writer.writerow({
                    "url": "https://ex.test/a",
                    "expected_tags": "судостроение; инжиниринг",
                    "review_status": "draft",
                })
                writer.writerow({
                    "url": "https://ex.test/b",
                    "expected_tags": "",
                    "review_status": "verified",
                })
            verified = read_references(path)
            all_data = read_references(path, include_drafts=True)
            self.assertEqual(len(verified), 1)
            self.assertEqual(len(all_data), 2)
            self.assertEqual(
                all_data["https://ex.test/a"]["expected_tags"],
                ["судостроение", "инжиниринг"],
            )

    def test_weight_gap_and_threshold_control_selection(self):
        row = {"ranked": [
            {
                "tag": "судостроение",
                "focus_similarity": 0.75,
                "body_similarity": 0.10,
                "keyword_boost": 0.00,
                "keyword_penalty": 0.00,
            },
            {
                "tag": "инжиниринг",
                "focus_similarity": 0.15,
                "body_similarity": 0.95,
                "keyword_boost": 0.00,
                "keyword_penalty": 0.00,
            },
        ]}
        self.assertEqual(
            threshold_tags(row, 0.78, 0.35, 0.05, 2),
            ["судостроение"],
        )
        self.assertEqual(
            threshold_tags(row, 0.20, 0.35, 0.05, 2),
            ["инжиниринг"],
        )
        self.assertEqual(
            threshold_tags(row, 0.78, 0.95, 0.05, 2),
            [],
        )

    def test_incompatible_scores_return_safe_status(self):
        row = (
            {"tags": ["машиностроение"]},
            {"url": "https://ex.test/a", "expected_tags": ["машиностроение"]},
        )
        self.assertEqual(
            optimize([row] * 40)["status"], "incompatible_audit"
        )

    def test_labeled_rows_matches_by_url_not_title(self):
        preds = {
            "https://ex.test/1": {"url": "https://ex.test/1", "tags": []},
            "https://ex.test/2": {"url": "https://ex.test/2", "tags": ["заводы"]},
        }
        refs = {
            "https://ex.test/2": {
                "url": "https://ex.test/2",
                "expected_tags": ["заводы"],
            },
        }
        self.assertEqual(len(labeled_rows(preds, refs)), 1)
        self.assertEqual(metrics(labeled_rows(preds, refs))["exact_match"], 1)


if __name__ == "__main__":
    unittest.main()
