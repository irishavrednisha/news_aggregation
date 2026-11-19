"""Regression cases extracted from the real accepted stage-2 news corpus.

SBERT scores come from the first stage-3 report for the same real 156 articles.
These tests exercise factual verification, not SBERT itself.
"""
import unittest
from processing.event_matching import same_event


def article(title, lead, published_at):
    return {"title": title, "text": lead, "published_at": published_at}


TOR_A = article(
    "Пять предприятий по разработке медных месторождений войдут в новую ТОР «Колыма»",
    'Все проекты принадлежат компании «Нордголд». Постановление о создании территории '
    'опережающего развития (ТОР) «Колыма» подписано. В границы ТОР войдут земельные '
    'участки площадью около 159 тыс. гектаров. РЮГК — 23,3 млрд рублей; '
    'Аэрис — 6,5 млрд рублей; Недра-М — 35 млрд рублей. '
    'В границах ТОР появится дата-центр мощностью 150 МВт.',
    "2026-11-16",
)
TOR_B = article(
    "В Магаданской области проекты цветной металлургии войдут в ТОР",
    'В регионе появится территория опережающего развития «Колыма». '
    'Площадь составит около 159 тыс. гектаров. Все пять предприятий '
    'принадлежат «Нордголд». РЮГК направит 23,3 млрд рублей, '
    'Аэрис вложит 6,5 млрд рублей, Недра-М — 35 млрд рублей. '
    'Кроме них будет дата-центр мощностью 150 МВт.',
    "2026-11-17",
)
REG_A = article(
    "Законопроект об усилении контроля за россыпной золотодобычей вынесли на общественное обсуждение",
    'Выразить свое мнение можно до 27 ноября 2026 года. Правительство подготовило '
    'постановление об изменениях государственного геологического контроля. '
    'Россыпные месторождения золота предлагается относить к категории высокого риска. '
    'Проект постановления проходит общественное обсуждение.',
    "2026-11-16",
)
REG_B = article(
    "Минприроды предлагает повысить категории риска для россыпей золота",
    'Предложено повысить категории риска для россыпных месторождений золота '
    'до высокой. Проект постановления правительства опубликован для '
    'общественного обсуждения, которое продлится до 27 ноября.',
    "2026-11-17",
)


class RealEvidenceRecallTests(unittest.TestCase):
    def test_real_tor_same_program_across_different_title_categories(self):
        result = same_event(TOR_A, TOR_B, 0.78735)
        self.assertTrue(result.merge, result)
        self.assertEqual(result.reason, "same_named_program_and_measures")

    def test_real_consultation_same_draft_across_different_headlines(self):
        result = same_event(REG_A, REG_B, 0.74459)
        self.assertTrue(result.merge, result)
        self.assertEqual(result.reason, "same_regulatory_draft_and_consultation")

    def test_same_program_name_without_numbers_is_not_enough(self):
        altered = article(TOR_B["title"], 'Создали ТОР «Колыма» и провели совещание.',
                          "2026-11-17")
        result = same_event(TOR_A, altered, 0.85)
        self.assertFalse(result.merge, result)

    def test_shared_numbers_different_program_does_not_merge(self):
        altered = dict(TOR_B, text=TOR_B["text"].replace("«Колыма»", "«Север»"))
        result = same_event(TOR_A, altered, 0.89)
        self.assertFalse(result.merge, result)

    def test_same_program_different_period_does_not_merge(self):
        altered = dict(TOR_B, published_at="2027-02-12")
        result = same_event(TOR_A, altered, 0.95)
        self.assertFalse(result.merge, result)
        self.assertEqual(result.reason, "publication_gap")

    def test_shared_regulatory_topic_different_consultation_deadline(self):
        altered = dict(REG_B, text=REG_B["text"].replace("27 ноября", "12 декабря"))
        result = same_event(REG_A, altered, 0.78)
        self.assertFalse(result.merge, result)

    def test_same_deadline_unrelated_regulation_does_not_merge(self):
        altered = article(
            "Минприроды предлагает изменить категории риска для угольных шахт",
            'Проект постановления опубликован для общественного обсуждения '
            'до 27 ноября. Предложено повысить категории риска для угольных шахт.',
            "2026-11-17",
        )
        result = same_event(REG_A, altered, 0.77)
        self.assertFalse(result.merge, result)

    def test_no_date_no_exemption(self):
        altered = dict(TOR_B, published_at=None)
        result = same_event(TOR_A, altered, 0.78735)
        self.assertFalse(result.merge, result)

    def test_explicit_different_event_dates_still_veto(self):
        a = dict(TOR_A, title=TOR_A["title"] + " 12 ноября 2026")
        b = dict(TOR_B, title=TOR_B["title"] + " 15 ноября 2026")
        result = same_event(a, b, 0.99)
        self.assertFalse(result.merge, result)
        self.assertEqual(result.reason, "different_event_date")

    def test_original_cross_company_and_action_guard_still_veto(self):
        a = article("ЧМЗ выпустил партию автокранов 25 тонн", "", "2026-11-17")
        b = article("ЧМЗ построил новый цех для автокранов", "", "2026-11-17")
        self.assertFalse(same_event(a, b, 0.99).merge)


if __name__ == "__main__":
    unittest.main()
