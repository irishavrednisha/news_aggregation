"""Offline tests for collector JSON -> grounded audit and CSV-safe outputs."""
import unittest
from scripts.check_grounded_tagging import audit, load_articles


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
