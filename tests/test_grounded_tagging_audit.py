"""Offline tests for collector JSON -> grounded audit and CSV-safe outputs."""
import unittest
from types import SimpleNamespace
from scripts.check_grounded_tagging import audit, load_articles, enrich_summaries, save_report


class GroundedAuditTests(unittest.TestCase):
    def test_uses_collector_articles_without_model_or_database(self):
        payload = {"articles": [
            {"title": "На воду спустили речное судно", "url": "https://a/1", "source": "A"},
            {"title": "2,3 тонны золота приобретет Гохран России", "url": "https://a/2"},
            {"title": "Сталь не вывозит", "url": "https://a/3"},
        ]}
        report = audit(payload, limit=230)
        self.assertEqual(report["counts"]["articles"], 3)
        self.assertEqual(report["counts"]["tagged"], 1)
        self.assertEqual(report["counts"]["review"], 1)
        self.assertEqual(report["counts"]["rejected"], 1)
        self.assertEqual(report["results"][0]["tags"], ["судостроение"])
        self.assertFalse(report["publishing_performed"])
        self.assertFalse(report["database_modified"])

    def test_audit_summary_has_priority_and_body_is_not_read(self):
        row = {
            "title": "На заводе запустили производство промышленных насосов",
            "summary": "На заводе запустили производство трансформаторов.",
            "url": "https://a/summary",
        }
        report = audit({"articles": [
            {**row, "text": "На верфи построили корабль." * 300}
        ]}, limit=230)
        without_body = audit({"articles": [row]}, limit=230)
        self.assertEqual(report["results"], without_body["results"])
        self.assertEqual(report["counts"]["summaries_present"], 1)
        self.assertEqual(
            report["results"][0]["tags"][0], "электротехническое направление"
        )
        self.assertTrue(report["results"][0]["summary_present"])

    def test_headline_only_audit_reports_missing_summaries(self):
        report = audit({"articles": [{
            "title": "Предприятие запустило новую линию",
            "text": "Компания выпустила самолет и корабль.",
            "url": "https://a/no-summary",
        }]}, limit=230)
        self.assertEqual(report["counts"]["summaries_present"], 0)
        self.assertFalse(report["results"][0]["summary_present"])
        self.assertNotIn("авиация", report["results"][0]["tags"])

    def test_textrank_then_whole_gemma_feeds_tagging_and_json(self):
        calls = []
        def fake_textrank(text, *, title, sentence_count, max_chars):
            calls.append(("textrank", text, title, sentence_count, max_chars))
            return "Выпущены промышленные трансформаторы."
        def fake_rewrite(extractive, *, backend, model, rewrite_mode):
            calls.append(("gemma", extractive, backend, model, rewrite_mode))
            return SimpleNamespace(
                text="Предприятие выпустило промышленные трансформаторы.",
                rewritten=True, warnings=[]
            )
        data = {"articles": [{
            "title": "На предприятии запустили новое оборудование",
            "text": "Полный текст предприятия и других отраслей.",
            "source": "ИСТОЧНИК", "url": "https://a/summ"
        }]}
        original = data["articles"][0].copy()
        enriched = enrich_summaries(
            data, limit=5, summarizer=fake_textrank,
            rewrite_draft=fake_rewrite, audit_alignment=lambda _t, _s: [],
        )
        self.assertEqual(data["articles"][0], original)
        self.assertEqual(calls[0][0], "textrank")
        self.assertEqual(calls[1][-1], "whole")
        self.assertEqual(calls[1][2], "gemma")
        report = audit(enriched, limit=5)
        row = report["results"][0]
        self.assertEqual(
            row["summary"],
            "Предприятие выпустило промышленные трансформаторы."
        )
        self.assertEqual(
            row["extractive_summary"],
            "Выпущены промышленные трансформаторы."
        )
        self.assertEqual(row["summary_method"], "textrank_gemma_whole")
        self.assertIn("электротехническое направление", row["tags"])
        self.assertTrue(row["summary_review_required"])
        self.assertEqual(report["counts"]["summaries_present"], 1)
        self.assertEqual(report["counts"]["summaries_rewritten"], 1)

    def test_suspicious_gemma_edit_falls_back_to_textrank(self):
        result = enrich_summaries(
            {"articles": [{
                "title": "На заводе создали трансформатор",
                "text": "На предприятии разработали новый трансформатор."
            }]}, limit=1,
            summarizer=lambda *args, **kwargs:
                "На предприятии разработали новый трансформатор.",
            rewrite_draft=lambda *args, **kwargs: SimpleNamespace(
                text="На предприятии разработали новый трансформатор.",
                rewritten=False, warnings=["changed_numbers_or_dates"]
            ),
            audit_alignment=lambda *_: [],
        )
        row = audit(result, limit=1)["results"][0]
        self.assertEqual(row["summary_method"], "textrank_fallback")
        self.assertEqual(row["summary"], row["extractive_summary"])
        self.assertIn("changed_numbers_or_dates", row["summary_warnings"])
        self.assertTrue(row["summary_review_required"])

    def test_no_body_and_existing_summary_handled_without_model(self):
        data = {"articles": [
            {"title": "На заводе выпустили станок", "summary": "На заводе выпустили станок."},
            {"title": "Выпущена новая система", "text": ""},
        ]}
        def forbidden(*args, **kwargs):
            raise AssertionError("Model and TextRank must not be called")
        result = enrich_summaries(
            data, limit=2, summarizer=forbidden,
            rewrite_draft=forbidden, audit_alignment=forbidden,
        )
        rows = audit(result, limit=2)["results"]
        self.assertEqual(rows[0]["summary_method"], "provided")
        self.assertEqual(rows[1]["summary"], "")
        self.assertEqual(rows[1]["summary_method"], "missing_source_text")
        self.assertEqual(rows[1]["summary_warnings"], ["missing_text_for_summary"])

    def test_ollama_exception_falls_back_to_original_textrank(self):
        def broken(*args, **kwargs):
            raise RuntimeError("Ollama не запущена")
        result = enrich_summaries(
            {"articles": [{"title": "Завод построил цех", "text": "Очень длинный текст"}]},
            limit=1, summarizer=lambda *args, **kwargs: "Завод построил цех.",
            rewrite_draft=broken, audit_alignment=lambda *_: [],
        )
        row = audit(result, limit=1)["results"][0]
        self.assertEqual(row["summary"], "Завод построил цех.")
        self.assertEqual(row["summary_method"], "textrank_fallback")
        self.assertIn("rewrite_error:RuntimeError", row["summary_warnings"][0])

    def test_rejects_unsupported_schema_and_duplicate_urls(self):
        with self.assertRaises(ValueError):
            load_articles({"results": {}})
        with self.assertRaises(ValueError):
            audit({"articles": [
                {"title": "На заводе запустили станок", "url": "same"},
                {"title": "Выпустили теплоход", "url": "same"},
            ]}, limit=10)

    def test_handles_missing_title_as_review(self):
        report = audit({"articles": [{"url": "https://a/empty"}]}, limit=10)
        self.assertEqual(report["results"][0]["outcome"], "review")
        self.assertEqual(report["results"][0]["reason"], "missing_title")


if __name__ == "__main__":
    unittest.main()
