"""Offline checks for the read-only integrated audit (no external traffic)."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from collectors.image_extractor import ImageCandidate
from processing.grounded_tagger import TagDecision
from scripts.audit_news_pipeline import audit_article, build_report


def article(source: str, url: str, title: str):
    return SimpleNamespace(
        source=source,
        title=title,
        url=url,
        published_at=None,
        text=("Промышленное предприятие выпустило новую партию техники. "
              "Она предназначена для модернизации действующего оборудования. ") * 5,
        text_source="html",
        image_candidates=[ImageCandidate("https://example.org/photo.jpg")],
    )


class PipelineAuditTests(unittest.TestCase):
    def test_reports_primary_filter_acceptance_but_untagged_event(self):
        row = article(
            "Военное.рф", "https://example.org/1",
            "В Петербурге проходят обыски по делу экс-замминистра обороны Булгакова",
        )
        with patch("scripts.audit_news_pipeline.build_textrank_summary",
                   return_value="Правоохранительные органы проводят обыски."):
            with patch("scripts.audit_news_pipeline.select_tags",
                       return_value=TagDecision((), "review", "no_industrial_event")):
                result = audit_article(row)
        self.assertTrue(result["filter"]["accepted"])
        self.assertEqual(result["tagger"]["outcome"], "review")
        self.assertIn("primary_filter_accepted_but_tagger_not_tagged", result["warnings"])
        self.assertEqual(len(result["image_candidates"]), 1)

    def test_mirrored_headlines_flagged_but_not_silently_merged(self):
        rows = [
            article("ФлотПром", "https://flotprom.ru/2026/example", "ОСК выпустила новый корабль"),
            article("ФлотКом", "https://flot.com/2026/example", "ОСК выпустила новый корабль"),
        ]
        with patch("scripts.audit_news_pipeline.build_textrank_summary",
                   return_value="Компания ОСК выпустила новый корабль на предприятии."):
            with patch("scripts.audit_news_pipeline.select_tags",
                       return_value=TagDecision(("судостроение",), "tagged", "test")):
                report = build_report(rows, {}, ["ФлотПром", "ФлотКом"])
        self.assertTrue(report["read_only"])
        self.assertFalse(report["db_modified"])
        self.assertFalse(report["max_publishing_performed"])
        self.assertEqual(report["collected_total"], 2)
        self.assertEqual(len(report["repeated_titles"]), 1)
        self.assertTrue(all(
            "duplicate_title_different_url" in x["warnings"]
            for x in report["articles"]
        ))

    def test_empty_summary_is_audited_not_invented(self):
        row = article("ТАСС", "https://tass.ru/123", "Российское предприятие выпустило станок")
        with patch("scripts.audit_news_pipeline.build_textrank_summary", return_value=""):
            with patch("scripts.audit_news_pipeline.select_tags",
                       return_value=TagDecision((), "review", "no_summary")):
                result = audit_article(row)
        self.assertIn("empty_summary", result["warnings"])
        self.assertEqual(result["summary"], "")


if __name__ == "__main__":
    unittest.main()
