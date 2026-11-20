"""Regressions observed in the 41-item report, plus counterexamples.

The original user diagnostic report is NOT shipped in the repository and is
never used to fit/claim a new independent evaluation.
"""
from __future__ import annotations
import unittest

from processing.qwen_abstractive_summary import attach_evidence, validate_response
from processing.qwen_claim_relations import (
    audit_claim_links, safe_acronym_alias, safe_geography_equivalence,
)
from processing.qwen_language_audit import audit_russian_quality
from processing.qwen_relation_audit import audit_relations


def _docs(body, title="Производственная новость"):
    return [{"id": "evidence:1", "title": title, "text": body,
             "source": "Тест", "url": "https://example.org/one",
             "published_at": "2026-11-20", "is_primary": True}]


def _check(body, summary, title="Производственная новость"):
    docs = _docs(body, title)
    generated = {"main_event": "Промышленная новость",
                 "is_news_event": True, "summary": summary,
                 "uncertainties": [], "requires_review": False}
    return validate_response(attach_evidence(generated, docs), docs)


class EvidenceActorTests(unittest.TestCase):
    def test_900002_commodity_cannot_express_interest_of_ghokhran(self):
        source = ("Драгметалл планируется приобрести до 4 декабря. "
                  "Гохран России выразил заинтересованность в приобретении "
                  "410 кг аффинированного золота в стандартных слитках. "
                  "Поставки оформляют по договорам купли-продажи.")
        bad = ("Драгметалл Гохрана России выразил заинтересованность в покупке "
               "410 кг золота. Закупку планируют завершить в декабре. "
               "Слитки приобретаются по договору.")
        good = ("Гохран России выразил заинтересованность в покупке "
                "410 кг золота. Закупку планируют завершить в декабре. "
                "Слитки приобретаются по договору.")
        check_bad = _check(source, bad)
        self.assertIn("claim_1_actor_mismatch:expressed_interest",
                      check_bad["issues"])
        self.assertIn("language_inanimate_reporting_actor",
                      check_bad["issues"])
        check_good = _check(source, good)
        self.assertNotIn("claim_1_actor_mismatch:expressed_interest",
                         check_good["issues"])
        self.assertNotIn("language_inanimate_reporting_actor",
                         check_good["issues"])

    def test_actor_check_does_not_guess_in_the_absence_of_source_verb(self):
        result = audit_claim_links([{
            "claim": "Предприятие выразило заинтересованность в контракте.",
            "source_ids": ["evidence:1"],
            "evidence_quotes": ["Предприятие провело переговоры по контракту."]
        }], _docs("Предприятие провело переговоры по контракту."))
        self.assertFalse(any("actor_mismatch" in x for x in result["issues"]))
        self.assertFalse(result["factually_verified"])


class EvidenceEventTests(unittest.TestCase):
    def test_900009_catalogue_cannot_be_rewritten_as_new_unveiling(self):
        source = ("Завод выпускает силовые гибкие кабели ELKAFLEX для "
                  "горнодобывающих предприятий. Инженер рассказал о кабелях. "
                  "Предприятие проводит испытания продукции по заявкам.")
        bad = ("Завод представил новые гибкие кабели ELKAFLEX для карьеров. "
               "Материалы рассчитаны на промышленную эксплуатацию. "
               "Предприятие предлагает заказчикам испытания.")
        check = _check(source, bad, "Как использовать кабели в добыче")
        self.assertIn("claim_1_event_not_supported:new_product_presentation",
                      check["issues"])

    def test_real_product_unveiling_is_not_rejected(self):
        source = ("Завод впервые представил новые гибкие кабели ELKAFLEX "
                  "на отраслевой выставке. В этом году начался серийный выпуск. "
                  "Изделия предназначены для горнодобывающих предприятий.")
        good = ("Завод представил новые гибкие кабели ELKAFLEX. "
                "Изделия предназначены для карьеров. "
                "Производство запущено в этом году.")
        check = _check(source, good)
        self.assertNotIn("claim_1_event_not_supported:new_product_presentation",
                         check["issues"])


class FinancialModalityTests(unittest.TestCase):
    def test_900014_cost_of_hypothetical_three_sites_not_actual(self):
        source = (
            "Рассмотрим условный расчёт. Если самосвал перевозит 20 тонн "
            "материала, 50 рейсов эквивалентны 1 млн рублей. "
            "Видеоаналитика фиксирует маршруты транспорта. "
            "В расчётном примере бессрочная лицензия на одну базу "
            "стоит 650 000 рублей, внедрение — 350 000 рублей. "
            "Первоначальные затраты составляют 1 млн рублей. "
            "Сценарий описывает три базы заказчика."
        )
        bad = ("Компания внедрила видеоаналитику на трёх площадках. "
               "Внедрение на трёх площадках обошлось в 1 млн рублей. "
               "Система фиксирует маршруты самосвалов.")
        correct = ("Для расчётного примера указаны затраты в 1 млн рублей. "
                   "Иллюстрация охватывает три площадки. "
                   "Видеоаналитика отслеживает маршруты машин.")
        self.assertIn("conditional_amount_presented_as_spending:1",
                      audit_relations(bad, _docs(source))["issues"])
        self.assertNotIn("conditional_amount_presented_as_spending:1",
                         audit_relations(correct, _docs(source))["issues"])


class AliasAndQualityTests(unittest.TestCase):
    def test_900029_government_rf_equals_government_russia(self):
        source = ("Правительство России внесло титан в перечень редких "
                  "металлов, следует из распоряжения. Перечень обновили. "
                  "Ранее перечень включал четырнадцать видов.")
        summary = ("Правительство РФ включило титан в перечень редких металлов. "
                   "Изменение закреплено распоряжением. "
                   "Перечень ранее включал другие материалы.")
        result = _check(source, summary)
        self.assertNotIn("unsupported_acronym:РФ", result["issues"])
        self.assertNotIn("claim_1_actor_acronym_not_in_evidence",
                         result["issues"])

    def test_rf_not_globally_accepted_for_unrelated_or_missing_source(self):
        source = ("Компания выпускает электротехнику для промышленных "
                  "предприятий. Состоялись испытания изделий.")
        summary = ("Компания РФ выпускает электротехнику. "
                   "Испытания изделий завершились. "
                   "Проект будет расширяться.")
        result = _check(source, summary)
        self.assertIn("unsupported_acronym:РФ", result["issues"])
        self.assertFalse(safe_acronym_alias("РФ", summary, source))

    def test_900027_russian_industry_adjective_supports_country_referent(self):
        self.assertTrue(safe_geography_equivalence(
            "России", "российские металлурги сократили выпуск стали"))
        self.assertFalse(safe_geography_equivalence(
            "России", "зарубежные металлурги сократили выпуск стали"))

    def test_russian_language_is_an_independent_review_axis(self):
        good = audit_russian_quality(
            "Гохран России выразил заинтересованность в покупке золота. "
            "Закупку планируют провести до декабря. Договор будет подготовлен.")
        bad = audit_russian_quality(
            "Драгметалл Гохрана России выразил заинтересованность в покупке. "
            "Закупку планируют провести до декабря. Договор будет подготовлен.")
        self.assertNotIn("language_inanimate_reporting_actor", good["issues"])
        self.assertIn("language_inanimate_reporting_actor", bad["issues"])
        self.assertFalse(good["grammar_verified"])
        self.assertFalse(good["naturalness_verified"])
        self.assertTrue(bad["editorial_language_review_required"])

    def test_long_sentences_are_just_flags(self):
        sample = " ".join(["Промышленное предприятие продолжает выпуск"]*18)+"."
        result = audit_russian_quality(sample)
        self.assertIn("language_sentence_too_long_review", result["flags"])
        self.assertEqual(result["issues"], [])


if __name__ == "__main__":
    unittest.main()
