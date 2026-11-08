"""No-network tests for the read-only AI tagging audit workflow."""
import sys
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

from check_ai_tagging import prepare_text_for_ai, selected_articles


class AITaggingAuditTests(unittest.TestCase):
    def test_textrank_audit_matches_one_news_cluster_summary_input(self):
        summarizer = SimpleNamespace(
            build_textrank_summary=Mock(return_value="Подлинный TextRank итог.")
        )
        title = "Завод внедрил новую технологическую линию"
        text = "Промышленный материал. " * 170
        with patch.dict(sys.modules, {"processing.summarizer": summarizer}):
            summary, lead = prepare_text_for_ai(title, text, "textrank")
        self.assertEqual(summary, "Подлинный TextRank итог.")
        self.assertEqual(lead, text[:330])
        summarizer.build_textrank_summary.assert_called_once_with(
            text=title + "\n" + text[:2500],
            title=title, sentence_count=4, max_chars=800,
        )

    def test_lead_mode_reproduces_previous_45_news_experiment(self):
        title = "Модернизация завода"
        text = "А" * 780 + "Б" * 400
        summary, lead = prepare_text_for_ai(title, text, "lead")
        self.assertEqual(summary, "А" * 780)
        self.assertEqual(lead, "Б" * 330)

    def test_invalid_summary_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_text_for_ai("Название", "Текст", "unknown")

    def test_primary_filter_optional_and_reasons_retained(self):
        body = (
            "На заводе установили новое оборудование и запустили "
            "производственную линию. Инженеры испытали систему, "
            "которая повышает производительность и безопасность. "
        ) * 3
        keep = {
            "source": "Заводы.рф",
            "title": "Завод внедрил станки для выпуска оборудования",
            "url": "https://example.org/news/new-equipment",
            "text": body, "published_at": "2026-11-05",
        }
        stale = {**keep, "title": "На заводе внедрили второй станок",
                 "published_at": "2025-03-10"}
        promo = {**keep, "title": "На АЗС запущена акция — скидка на топливо"}
        all_articles = [keep, stale, promo]
        self.assertEqual(selected_articles(all_articles)[0], all_articles)
        accepted, rejected = selected_articles(
            all_articles, primary_filter=True, as_of=date(2026, 11, 6),
        )
        self.assertEqual(accepted, [keep])
        self.assertEqual(
            [x["reason"] for x in rejected],
            ["stale_article", "offtopic_consumer_promotion"],
        )

    def test_source_filter_applied_before_primary_filter(self):
        rows = [
            {"source": "Заводы.рф", "title": "Статья 1"},
            {"source": "НедраДВ", "title": "Статья 2"},
        ]
        accepted, rejected = selected_articles(rows, source="НедраДВ")
        self.assertEqual(accepted, [rows[1]])
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
