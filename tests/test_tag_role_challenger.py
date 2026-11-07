"""Offline role challenger contracts: no SBERT, database or MAX."""
import unittest

from processing.tag_role_challenger import (
    aviation_production_evidence, shipbuilding_evidence,
    production_process_evidence, candidate_tags,
)
from check_tag_role_challenger import compare


def row(title, summary, scores):
    return {
        "title": title,
        "summary_used": summary,
        "url": "https://example.test/" + str(len(title)),
        "tags": [tag for tag, _ in scores[:2]],
        "ranked": [{"tag": tag, "final_score": score}
                   for tag, score in scores],
    }


class TagRoleChallengerTests(unittest.TestCase):
    def test_mention_of_ship_is_not_shipbuilding(self):
        self.assertFalse(shipbuilding_evidence(
            "В Невельске возрождают производство по переработке морепродуктов",
            "Компания управляет 51 судном и морскими терминалами."
        ))
        self.assertFalse(shipbuilding_evidence(
            "Азия станет рынком российского СПГ",
            "Морские поставки требуют доступного флота и газовозов."
        ))
        self.assertTrue(shipbuilding_evidence(
            "Специалисты Северной верфи испытывают оборудование",
            "На судостроительном предприятии проверяют систему."
        ))
        self.assertTrue(shipbuilding_evidence(
            "Представили концепт эсминца",
            "Новый проект эсминца проходит модельные испытания."
        ))

    def test_aviation_requires_production_or_development(self):
        self.assertFalse(aviation_production_evidence(
            "Дроны следят за лесными пожарами",
            "Беспилотники выполняют мониторинг лесов."
        ))
        self.assertFalse(aviation_production_evidence(
            "Испытали двигатель для водородного поезда",
            "Железнодорожный транспорт"
        ))
        self.assertTrue(aviation_production_evidence(
            "Производство самолета МС-21",
            "Компания модернизировала авиационное оборудование."
        ))

    def test_production_process_is_independent_candidate(self):
        self.assertTrue(production_process_evidence(
            "Резидент предприятия наращивает выпуск медицинского оборудования"
        ))
        self.assertFalse(production_process_evidence(
            "Компания выплатила дивиденды по итогам года"
        ))
        r = row(
            "В Невельске возрождают производство по переработке морепродуктов",
            "Компания управляет морскими терминалами и 51 судном.",
            [("судостроение", 0.51), ("заводы", 0.40),
             ("технологическое производство", 0.276)],
        )
        self.assertEqual(candidate_tags(r)["tags"],
                         ["заводы", "технологическое производство"])

    def test_ship_design_kept_and_aviation_suppressed(self):
        r = row(
            "Naval Group представила концепт эсминца",
            "Разрабатывается проект эсминца для нового флота.",
            [("авиация", 0.475), ("оборонная промышленность", 0.463),
             ("судостроение", 0.463)],
        )
        self.assertEqual(candidate_tags(r)["tags"],
                         ["оборонная промышленность", "судостроение"])

    def test_comparison_only_records_changes_without_editing_inputs(self):
        r = row(
            "ОСК провела фестиваль судостроителей",
            "Фестиваль в честь юбилея.",
            [("судостроение", 0.81)],
        )
        data = {"results": [r]}
        output = compare(data)
        self.assertEqual(output["articles"], 1)
        self.assertFalse(output["production_tagger_changed"])
        self.assertEqual(r["tags"], ["судостроение"])
        self.assertEqual(output["results"][0]["candidate_tags"], [])


if __name__ == "__main__":
    unittest.main()
