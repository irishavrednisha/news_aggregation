"""Offline regressions from real Russian industry headlines, no models or DB."""
import unittest

from processing.grounded_tagger import select_tags


class GroundedTaggerTests(unittest.TestCase):
    def test_real_industrial_events_have_specific_main_subject_tags(self):
        cases = [
            (
                'На воду спустили электрическое речное судно "Пожарский" проекта ПКС 40.2',
                "судостроение",
            ),
            ("Тюменский завод ЖБИ восстанавливает работу после паводка", "заводы"),
            ("Ростех запустил серийное производство высокоточных алюминиевых деталей для радиоэлектроники",
             "технологическое производство"),
            ('В ОЭЗ «Технополис Москва» запущено российско-китайское производство SIM-карт',
             "технологическое производство"),
            ('"Октава ДМ" модернизировала микрофонный предусилитель',
             "электротехническое направление"),
            ("«Роснефть» совершенствует технологии ремонта скважин",
             "нефтегазовая промышленность"),
            ("Конструкторы разработали новый электропоезд", "железнодорожное машиностроение"),
            ("На предприятии освоили выпуск промышленных насосов", "машиностроение"),
            ("На российской верфи построили новый ледокол", "судостроение"),
            ("На заводе запустили производство трансформаторов",
             "электротехническое направление"),
            ("В России разработали новый авиационный двигатель для самолета",
             "авиация"),
        ]
        for title, expected_tag in cases:
            with self.subTest(title=title):
                result = select_tags(title, ranked=[
                    {"tag": "авиация", "final_score": 0.99},
                    {"tag": "горнодобывающая промышленность", "final_score": 0.95},
                ])
                self.assertEqual(result.outcome, "tagged")
                self.assertIn(expected_tag, result.tags, (title, result))
                self.assertLessEqual(len(result.tags), 2)

    def test_irrelevant_even_with_extreme_sbert_scores(self):
        cases = [
            'Финляндия поставит донные мины для датского флота',
            'Производители лазеров для радиэлектронной отрасли заплатят пониженные страховые взносы',
            'Золотодобытчиков могут обязать создавать фонды для рекультивации земель',
            '2,3 тонны золота приобретет Гохран России до конца ноября',
            'Акционеры предприятия утвердили новый устав',
            'На нефтяном месторождении случилось ДТП',
        ]
        for title in cases:
            with self.subTest(title=title):
                result = select_tags(title, ranked=[
                    {"tag": "судостроение", "final_score": 0.99},
                    {"tag": "горнодобывающая промышленность", "final_score": 0.98},
                ])
                self.assertEqual(result.tags, ())
                self.assertIn(result.outcome, ("reject", "review"))

    def test_uncertain_industry_does_not_get_random_tags_or_silent_rejection(self):
        cases = [
            "Сталь не вывозит",
            'АО "Птицефабрика Чайковская" перевыполнила план по производству ВТМ',
            "«Норникель» объявляет итоги производственной деятельности за 9М 2026 г.",
            "Ростех представил новую промышленную разработку",
            "Михаил Алексеев: для меня Ил-114-300 — это дело жизни",
        ]
        for title in cases:
            with self.subTest(title=title):
                result = select_tags(title, ranked=[
                    {"tag": "химическая промышленность", "final_score": 0.96},
                    {"tag": "авиация", "final_score": 0.94},
                ])
                self.assertIn(result.outcome, ("review", "reject"))
                self.assertFalse(result.tags)

    def test_sbert_cannot_tag_irrelevant_context(self):
        result = select_tags(
            "Тюменский завод ЖБИ восстанавливает работу после паводка",
            "Восстановление производственных линий, выпуск железобетонных изделий.",
            ranked=[{"tag": "авиация", "final_score": 0.999}],
        )
        self.assertEqual(result.tags, ("заводы",))
        self.assertEqual(result.outcome, "tagged")

    def test_foreign_project_with_explicit_russian_participant_is_not_banned(self):
        result = select_tags(
            "Ростех в Южной Корее запустил производство новых авиационных двигателей"
        )
        self.assertEqual(result.outcome, "tagged")
        self.assertIn("авиация", result.tags)

    def test_foreign_shipbuilding_can_receive_industry_tags(self):
        result = select_tags(
            "В Южной Корее спустили на воду пятую подлодку нового поколения"
        )
        self.assertEqual(result.outcome, "tagged")
        self.assertIn("судостроение", result.tags)

    def test_foreign_shipbuilding_has_industry_regardless_of_geography(self):
        decision = select_tags(
            "Naval Group представила концепт эсминца с 96 ячейками для ракет"
        )
        self.assertEqual(decision.outcome, "tagged")
        self.assertIn("судостроение", decision.tags)

    def test_generic_industrial_factory_without_sector_is_tagged(self):
        result = select_tags("В Челябинске запустили новую линию производства деталей")
        self.assertEqual(result.tags, ("технологическое производство",))


if __name__ == "__main__":
    unittest.main()
