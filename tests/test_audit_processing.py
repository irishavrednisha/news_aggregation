"""Offline checks for the read-only collection processing audit."""
import unittest
from datetime import date

from scripts.audit_processing import audit_payload, title_key


TEXT = (
    "На заводе в Туле запущена линия сборки насосов для промышленной "
    "водоочистки. Предприятие освоило выпуск новых агрегатов и провело "
    "заводские испытания. Инженеры проверили рабочие характеристики "
    "опытной серии. Новые станки установили в производственном цехе. "
    "Оборудование прошло техническую приёмку."
)


def row(title, url, source, *, published_at="2026-11-08", tls=False):
    return {
        "title": title, "url": url, "source": source,
        "published_at": published_at, "unverified_tls": tls,
        "text": TEXT,
    }


class AuditProcessingTests(unittest.TestCase):
    def test_diagnostic_groups_and_replay_same_existing_filter(self):
        duplicates = "На заводе запущено производство промышленных насосов"
        data = {"collected_at": "2026-11-09T03:14:38", "articles": [
            row(duplicates, "https://first.example/n1", "A"),
            row(duplicates, "https://second.example/n1", "B", tls=True),
            row("Главное в отраслевых СМИ за неделю",
                "https://first.example/digest", "A"),
            row("Создано новое станочное оборудование",
                "https://first.example/old", "A", published_at="2025-03-11"),
        ]}
        report, candidates = audit_payload(
            data, as_of=date(2026, 11, 9), max_age_days=30,
        )
        self.assertEqual(report["summary"]["input_articles"], 4)
        self.assertEqual(report["summary"]["primary_filter_accepted"], 2)
        self.assertEqual(report["summary"]["primary_filter_rejected"], 2)
        self.assertEqual(report["summary"]["unverified_tls_articles"], 1)
        self.assertEqual(report["summary"]["possible_cross_source_duplicate_groups"], 1)
        self.assertEqual(report["summary"]["possible_cross_source_duplicate_articles"], 2)
        self.assertIn("stale_article", report["summary"]["by_filter_reason"])
        self.assertIn("editorial_digest_or_service",
                      report["summary"]["by_filter_reason"])
        self.assertEqual(len(candidates["articles"]), 2)
        self.assertTrue(candidates["articles"][1]["unverified_tls"])
        self.assertTrue(report["decisions"][1]["requires_source_verification"])
        self.assertTrue(all(d["possible_cross_source_duplicate"]
                            for d in report["decisions"][:2]))
        self.assertTrue(report["read_only"])
        self.assertFalse(report["publishing_performed"])

    def test_same_source_title_not_marked_as_cross_source_duplicate(self):
        title = "На предприятии создают промышленное оборудование"
        data = {"articles": [
            row(title, "https://a.example/a", "A"),
            row(title, "https://a.example/b", "A"),
        ]}
        report, _ = audit_payload(data, as_of=date(2026, 11, 9))
        self.assertEqual(
            report["summary"]["possible_cross_source_duplicate_groups"], 0
        )

    def test_event_guard_is_diagnostic_only(self):
        article = row(
            "Работники завода прошли обучение в университете",
            "https://a.example/training", "A",
        )
        report, candidates = audit_payload(
            {"articles": [article]}, as_of=date(2026, 11, 9)
        )
        self.assertIsNotNone(report["decisions"][0]["event_guard_reason"])
        # The filter may independently reject training news; the guard itself
        # must not alter the primary filter decision.
        self.assertEqual(
            len(candidates["articles"]),
            int(report["decisions"][0]["accepted_by_primary_filter"]),
        )

    def test_invalid_input_and_conservative_title_key(self):
        with self.assertRaises(ValueError):
            audit_payload({"results": []}, as_of=date(2026, 11, 9))
        self.assertEqual(
            title_key("Новые «насосы» для Тулы!"),
            title_key("НОВЫЕ насосы для Тулы"),
        )


if __name__ == "__main__":
    unittest.main()
