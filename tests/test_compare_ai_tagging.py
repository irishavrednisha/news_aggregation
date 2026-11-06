"""Offline model comparison must reject differing inputs and never call models."""
import copy
import unittest

from compare_ai_tagging import compare_reports


def report(model="MiniLM", tags=None):
    return {
        "model": model,
        "primary_filter": True,
        "as_of": "2026-11-06",
        "summary_mode": "textrank",
        "results": [{
            "source": "Производства.рф",
            "title": "Модернизация завода",
            "url": "https://example.org/plant/1",
            "summary_used": "Завод установил новое оборудование.",
            "tags": tags or [],
            "reason": "tagged" if tags else "no_confident_industrial_tag",
            "event_type": "industrial_production",
            "ranked_tags": [{"tag": "заводы", "score": 0.3}],
        }],
    }


class CompareAIReportsTests(unittest.TestCase):
    def test_identical_news_with_improved_tags(self):
        base = report(tags=["ТЭК"])
        challenger = report(model="mDeBERTa", tags=["технологическое производство"])
        comp = compare_reports(base, challenger)
        self.assertEqual(comp["count"], 1)
        self.assertEqual(comp["changed_tag_sets"], 1)
        self.assertEqual(comp["same_tag_sets"], 0)
        self.assertEqual(comp["changes"][0]["new_tags"], ["технологическое производство"])

    def test_same_tags_with_different_order_treated_equal(self):
        base = report(tags=["заводы", "технологическое производство"])
        challenger = report(model="mDeBERTa", tags=["технологическое производство", "заводы"])
        comp = compare_reports(base, challenger)
        self.assertEqual(comp["same_tag_sets"], 1)
        self.assertEqual(comp["changed_tag_sets"], 0)

    def test_different_summaries_cannot_be_compared(self):
        base = report(tags=["ТЭК"])
        challenger = report(model="mDeBERTa", tags=["заводы"])
        challenger["results"][0]["summary_used"] = "Совсем другой текст."
        with self.assertRaisesRegex(ValueError, "Разный текст"):
            compare_reports(base, challenger)

    def test_different_source_sets_cannot_be_compared(self):
        base = report(tags=["ТЭК"])
        challenger = report(model="mDeBERTa", tags=["заводы"])
        challenger["results"][0]["url"] = "https://example.org/plant/2"
        with self.assertRaisesRegex(ValueError, "Наборы новостей различаются"):
            compare_reports(base, challenger)

    def test_different_filter_config_cannot_be_compared(self):
        base = report(tags=["ТЭК"])
        challenger = report(model="mDeBERTa", tags=["заводы"])
        challenger["primary_filter"] = False
        with self.assertRaisesRegex(ValueError, "Несовпадение параметра"):
            compare_reports(base, challenger)

    def test_no_tags_are_counted_and_not_confused_with_quality(self):
        base = report(tags=[])
        challenger = report(model="mDeBERTa", tags=["заводы"])
        comp = compare_reports(base, challenger)
        self.assertEqual(comp["only_challenger_tagged"], 1)
        self.assertEqual(comp["baseline_tagged"], 0)
        self.assertEqual(comp["challenger_tagged"], 1)
        self.assertEqual(comp["both_tagged_different"], 0)


if __name__ == "__main__":
    unittest.main()
