"""Real archived articles: hard-positive and hard-negative event-identity gates.

The similarity values were observed in the first independent ARCHIVE SBERT
evaluation (not invented), before the new corroboration logic was written.
The separate full SBERT holdout job remeasures all case labels on every push.
"""
import json
from pathlib import Path
import unittest

from processing.event_matching import same_event

DATA=json.loads(Path("tests/fixtures/stage3_archival_holdout.json").read_text(encoding="utf-8"))
BY_ID={item["id"]:item for item in DATA["articles"]}


class UnseenRealArticleRegression(unittest.TestCase):
    def test_identical_manturovo_news_on_two_sites(self):
        result=same_event(BY_ID[4],BY_ID[10],0.92833)
        self.assertTrue(result.merge, result)
        self.assertEqual(result.reason,"near_identical_full_report")

    def test_manturovo_differently_worded_305m_project(self):
        for left,right,score in ((4,145,0.76175),(10,145,0.82291)):
            with self.subTest(left=left,right=right):
                result=same_event(BY_ID[left],BY_ID[right],score)
                self.assertTrue(result.merge, result)
                self.assertEqual(result.reason,"same_company_shared_measures_and_report")

    def test_morshansk_exact_copy_different_site(self):
        result=same_event(BY_ID[32],BY_ID[193],0.91058)
        self.assertTrue(result.merge, result)
        self.assertEqual(result.reason,"near_identical_full_report")

    def test_uralkhimplast_exact_repost(self):
        self.assertTrue(same_event(BY_ID[18],BY_ID[104],1.0).merge)

    def test_same_osk_distinct_education_events(self):
        result=same_event(BY_ID[37],BY_ID[65],0.76556)
        self.assertFalse(result.merge)
        self.assertEqual(result.reason,"different_object")

    def test_same_sveza_different_factory_action_remains_separate(self):
        for left,right in ((4,153),(4,198),(145,198),(127,153)):
            with self.subTest(left=left,right=right):
                self.assertFalse(same_event(BY_ID[left],BY_ID[right],0.85).merge)

    def test_identical_headline_different_real_body_cannot_bypass_hard_veto(self):
        other=dict(BY_ID[4], text=BY_ID[153]["text"],
                   published_at="2026-10-01")
        self.assertFalse(same_event(BY_ID[4],other,0.99).merge)


if __name__=="__main__":
    unittest.main()
