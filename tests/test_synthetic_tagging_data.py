"""Sanity checks for the independent fictional tagging stress benchmark.

This test checks dataset integrity, not Sentence-BERT's accuracy.
The dataset is NOT training data and MUST NOT be combined with real news
to claim real-world precision/recall.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "synthetic_tagging_120.json"
LABELS = ROOT / "benchmarks" / "synthetic_tagging_labels_120.csv"

TAGS = {
    "авиация", "автомобильная промышленность", "атомная отрасль",
    "горнодобывающая промышленность", "железнодорожное машиностроение",
    "заводы", "инжиниринг", "инновации", "машиностроение",
    "металлургия", "нефтегазовая промышленность", "оборонная промышленность",
    "судостроение", "технологии", "технологическое производство",
    "химическая промышленность", "электротехническое направление", "ТЭК",
}


class SyntheticTaggingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATA.read_text(encoding="utf-8"))
        with LABELS.open(encoding="utf-8-sig", newline="") as stream:
            cls.references = list(csv.DictReader(stream))

    def test_120_distinct_unpublished_articles(self):
        rows = self.dataset["articles"]
        self.assertEqual(len(rows), 120)
        self.assertTrue(self.dataset["synthetic"])
        self.assertTrue(self.dataset["not_for_publication"])
        self.assertEqual(len({r["url"] for r in rows}), 120)
        self.assertEqual(len({r["title"] for r in rows}), 120)
        self.assertTrue(all(r["url"].startswith("synthetic://") for r in rows))
        self.assertTrue(all(r["synthetic"] for r in rows))
        self.assertTrue(all(r["text_source"] == "synthetic" for r in rows))
        self.assertTrue(all(len(r["text"]) >= 100 for r in rows))

    def test_case_mix_and_labels(self):
        rows = self.dataset["articles"]
        self.assertEqual(Counter(r["scenario"] for r in rows), {
            "positive": 72, "negative": 24, "challenge": 24,
        })
        self.assertEqual(len(self.references), 120)
        self.assertEqual(
            {r["url"] for r in rows},
            {r["url"] for r in self.references},
        )
        expected_coverage = Counter()
        for r in self.references:
            self.assertEqual(r["review_status"], "draft")
            self.assertEqual(r["label_origin"], "synthetic")
            tags = [x.strip() for x in r["expected_tags"].split(";") if x.strip()]
            self.assertEqual(len(tags), len(set(tags)))
            self.assertTrue(set(tags).issubset(TAGS))
            if r["scenario"] == "positive":
                self.assertEqual(len(tags), 1)
                expected_coverage[tags[0]] += 1
            if r["scenario"] == "negative":
                self.assertEqual(tags, [])
        self.assertEqual(set(expected_coverage), TAGS)
        self.assertTrue(all(expected_coverage[tag] == 4 for tag in TAGS))

    def test_model_input_does_not_contain_label_hints(self):
        for record in self.dataset["articles"]:
            body = record["text"].lower()
            for leaking_phrase in (
                "нет подходящего тематического тега",
                "классификация должна учитывать",
                "это искусственный пример",
                "правильный ответ",
            ):
                self.assertNotIn(leaking_phrase, body)

    def test_independent_from_real_news_urls(self):
        # The model gets only title + text. Reference CSV is never
        # imported by check_focused_tagging.py.
        self.assertTrue(all(
            "http://" not in r["url"] and "https://" not in r["url"]
            for r in self.dataset["articles"]
        ))


if __name__ == "__main__":
    unittest.main()
