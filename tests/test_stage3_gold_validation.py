"""Gold-pair audit must never report inflated or invalid manual review counts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.audit_stage3 import compare_pair_labels


class GoldValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [
            {"url": f"https://example.test/{n}", "title": f"Статья {n}",
             "text": "Производственная новость", "status": "accepted"}
            for n in range(3)
        ]
        self.source = self.root / "real.json"
        self.source.write_text(json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")
        self.decisions = [
            {"url": row["url"], "cluster": f"event:{i}"} for i, row in enumerate(self.rows)
        ]
        self.gold = self.root / "labels.json"

    def check(self, pairs):
        self.gold.write_text(json.dumps({"pairs": pairs}, ensure_ascii=False), encoding="utf-8")
        return compare_pair_labels(self.rows, self.decisions, self.decisions,
                                   self.gold, self.source)

    def test_unique_pairs_are_counted(self):
        report = self.check([{"a_index": 0, "b_index": 1, "same_event": False}])
        self.assertEqual(report["revised"]["counts"]["labeled_pairs"], 1)

    def test_duplicate_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate/reversed"):
            self.check([
                {"a_index": 0, "b_index": 1, "same_event": False},
                {"a_index": 0, "b_index": 1, "same_event": False},
            ])

    def test_reversed_pair_is_rejected_even_with_conflicting_label(self):
        with self.assertRaisesRegex(ValueError, "Duplicate/reversed"):
            self.check([
                {"a_index": 0, "b_index": 1, "same_event": False},
                {"a_index": 1, "b_index": 0, "same_event": True},
            ])

    def test_invalid_index_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "invalid accepted article"):
            self.check([{"a_index": 0, "b_index": 100, "same_event": False}])

    def test_pair_of_same_article_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            self.check([{"a_index": 0, "b_index": 0, "same_event": True}])

    def test_non_boolean_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.check([{"a_index": 0, "b_index": 1, "same_event": "false"}])


if __name__ == "__main__":
    unittest.main()
