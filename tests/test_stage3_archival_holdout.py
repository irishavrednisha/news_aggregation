"""Source-grounded archival holdout integrity and read-only replay.

This is NOT a second human annotator. Claims about new unseen production-news
recall remain provisional until separately reviewed.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.audit_stage3_archival_holdout import _verify, run

FIXTURE = Path("tests/fixtures/stage3_archival_holdout.json")


class Stage3ArchivalHoldoutTests(unittest.TestCase):
    def test_distinct_archive_stories_and_labels(self):
        data=json.loads(FIXTURE.read_text(encoding="utf-8"))
        articles, pairs=_verify(data)
        positives=[p for p in pairs if p["same_event"]]
        negatives=[p for p in pairs if not p["same_event"]]
        self.assertEqual(len(positives),5)
        self.assertGreaterEqual(len(negatives),20)
        self.assertGreaterEqual(len(articles),17)
        for p in positives:
            self.assertNotEqual(articles[p["a_index"]]["url"],articles[p["b_index"]]["url"])
            self.assertNotEqual(articles[p["a_index"]]["source"],articles[p["b_index"]]["source"])
            self.assertTrue(p["evidence"])
        self.assertIn("NOT_independently_human_verified", data["annotation_status"])

    def test_duplicate_or_reversed_pair_rejected(self):
        data=json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["pairs"].append(dict(data["pairs"][0],a_index=data["pairs"][0]["b_index"],
                                  b_index=data["pairs"][0]["a_index"]))
        with self.assertRaisesRegex(ValueError, "Repeated"):
            _verify(data)

    def test_nonboolean_label_rejected(self):
        data=json.loads(FIXTURE.read_text(encoding="utf-8"))
        data["pairs"][0]["same_event"]="true"
        with self.assertRaisesRegex(ValueError, "Invalid label"):
            _verify(data)

    def test_lexical_smoke_repeatable_without_database(self):
        before=FIXTURE.read_bytes()
        a=run(FIXTURE,backend="lexical")
        b=run(FIXTURE,backend="lexical")
        self.assertEqual(a,b)
        self.assertEqual(before,FIXTURE.read_bytes())
        self.assertTrue(a["read_only"])
        self.assertEqual(a["db_writes"],0)
        self.assertEqual(a["positive_pairs"],5)
        self.assertGreaterEqual(a["negative_pairs"],20)


if __name__=="__main__":
    unittest.main()
