"""Regression cases drawn from the 230 collected real industrial-news titles.

These are manually checked directional examples, NOT a statistically
independent holdout or proof of precision/recall. The classifier must be
able to run without databases, SBERT, Ollama, or network access.
"""
import unittest

from processing.grounded_tagger import CATEGORY, select_tags


class RealNewsEventRegressionTests(unittest.TestCase):
    def test_industry_labels_match_the_production_taxonomy(self):
        import ast
        from pathlib import Path
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / "processing" / "tagger.py")
            .read_text(encoding="utf-8")
        )
        assigned = next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "TAG_KEYWORDS"
                for target in node.targets
            )
        )
        tag_names = {node.value for node in assigned.keys}
        self.assertEqual({name for name, _ in CATEGORY} | {"заводы", "технологическое производство"}, tag_names)

    def test_recovered_industrial_examples(self):
        examples = [
            ("РУСАЛ создает производство ценного редкоземельного металла скандия",
             "металлургия"),
            ("СОЛЛЕРС запускает в России полный цикл выпуска 8-ступенчатых АКПП",
             "автомобильная промышленность"),
            ("Производство новых видов автокомпонентов стартовало в Костроме",
             "автомобильная промышленность"),
            ("«Синара» вывела на обкатку в Петербурге трехсекционный трамвай 71-233А",
             "железнодорожное машиностроение"),
            ("«Роснефть» совершенствует технологии добычи нефти",
             "нефтегазовая промышленность"),
            ("«Роснефть» совершенствует технологии ремонта скважин",
             "нефтегазовая промышленность"),
            ("Лукойл начал промышленную добычу нефти на командиршорской группе месторождений",
             "нефтегазовая промышленность"),
            ("«Роснефть» разработала присадки к моторным маслам для стационарных газовых двигателей",
             "химическая промышленность"),
            ("В РФ создали собственный датчик температуры для химических реакторов",
             "электротехническое направление"),
            ("«Октава ДМ» модернизировала микрофонный предусилитель",
             "электротехническое направление"),
            ("На воду спустили электрическое речное судно «Пожарский» проекта ПКС 40.2",
             "судостроение"),
            ("Тюменский завод ЖБИ восстанавливает работу после паводка",
             "заводы"),
            ("Зенитный боевой модуль «Цитадель» установили на мобильную платформу",
             "оборонная промышленность"),
            ("АЛРОСА впервые добыла голубой алмаз ювелирного качества",
             "горнодобывающая промышленность"),
            ("Недропользователи Ягоднинского округа Колымы извлекли с начала года 5,6 тонны золота",
             "горнодобывающая промышленность"),
            ("Ростех наладил выпуск высокоточных алюминиевых деталей",
             "технологическое производство"),
            ("НТМК перевёл контроль зоны горения в домнах на видеонаблюдение",
             "металлургия"),
            ("ИИ создаёт «Норникелю» 10 тыс. палладиевых структур в месяц",
             "металлургия"),
            ("В Москве открыт кластер электробатарей при участии Путина, Собянина и Лихачёва",
             "электротехническое направление"),
            ("На «Янтаре» автоматизируют линию резки металла",
             "технологическое производство"),
            ("Ростех выводит на рынок новый аппарат для наркоза с поддержкой анестетика пятого поколения",
             "машиностроение"),
        ]
        for title, tag in examples:
            with self.subTest(title=title):
                decision = select_tags(title)
                self.assertEqual(decision.outcome, "tagged", decision)
                self.assertIn(tag, decision.tags, decision)
                self.assertLessEqual(len(decision.tags), 2)

    def test_context_disambiguates_reactor_and_does_not_invent_client_sector(self):
        atomic = select_tags(
            'На "Чукотке" запустили второй реактор',
            summary='На атомном ледоколе "Чукотка" был выполнен физический пуск реактора РИТМ-200.'
        )
        self.assertIn("атомная отрасль", atomic.tags)
        sensor = select_tags(
            "В РФ создали собственный датчик температуры для химических реакторов",
            text="Российский датчик предназначен для реакторов получения метанола."
        )
        self.assertNotIn("химическая промышленность", sensor.tags)
        self.assertIn("электротехническое направление", sensor.tags)
        movie = select_tags(
            "Историческая драма «Рождение империи», созданная при поддержке «Роснефти», выходит в широкий прокат",
            text="Роснефть выступила генеральным партнером фильма. Компания занимается добычей нефти."
        )
        self.assertFalse(movie.tags)

    def test_other_nonindustrial_events_do_not_get_sector_tags(self):
        cases = [
            "Naval Group представила концепт эсминца с 96 ячейками для ракет",
            "Финляндия поставит донные мины для датского флота",
            "Производители лазеров для радиэлектронной отрасли заплатят пониженные страховые взносы",
            "Историческая драма «Рождение империи», созданная при поддержке «Роснефти», выходит в широкий прокат",
            "«Роснефть» открыла новые цифровые лаборатории в Ачинском техникуме нефти и газа",
            "Компания ТРИС – лидер рынка бурового инструмента для BAUER и SANY",
            "Как проверить, есть ли у поставщика собственное производство",
            "Как производителю начать попадать в упоминания нейросетей",
            "2,3 тонны золота приобретет Гохран России до конца ноября",
            "Посольство КНР предупредило граждан об ответственности за незаконную добычу золота в России",
            "Погрузка угля на ДВЖД выросла за январь-октябрь на 3,6%",
            "Дроны следят за пожарами",
        ]
        for title in cases:
            with self.subTest(title=title):
                decision = select_tags(title, text="В статье упоминаются металлургия, заводы и авиация.")
                self.assertEqual(decision.tags, ())
                self.assertIn(decision.outcome, ("reject", "review"))

    def test_summary_outweighs_specific_headline_sector(self):
        title = "На заводе запустили производство промышленных насосов"
        summary = "На заводе запустили производство электрических трансформаторов."
        result = select_tags(title, summary)
        self.assertEqual(result.outcome, "tagged")
        self.assertEqual(result.tags[0], "электротехническое направление")
        self.assertTrue(any("summary=" in item for item in result.evidence))

    def test_full_body_is_completely_ignored(self):
        title = "На предприятии модернизировали производственную линию"
        summary = "На предприятии модернизировали трансформаторные установки."
        unrelated = (
            "На другой верфи построили новый корабль. "
            "Авиационный завод выпустил самолет. "
            "Разработали технологию добычи нефти. "
        ) * 300
        first = select_tags(title, summary, text=unrelated)
        second = select_tags(title, summary, text="")
        self.assertEqual(first, second)
        self.assertEqual(first.tags[0], "электротехническое направление")

    def test_no_summary_cannot_infer_from_body(self):
        title = "На предприятии запустили новое оборудование"
        result = select_tags(
            title,
            text="Выпустили самолет. Построили судно. Изготовили насос."
        )
        self.assertNotIn("авиация", result.tags)
        self.assertNotIn("судостроение", result.tags)
        self.assertNotIn("машиностроение", result.tags)

    def test_summary_resolves_specific_product_when_title_is_generic(self):
        result = select_tags(
            "На предприятии разработали новое оборудование",
            "Компания разработала электродвигатель для промышленной линии."
        )
        self.assertEqual(result.outcome, "tagged")
        self.assertIn("электротехническое направление", result.tags)

    def test_foreign_location_does_not_block_real_industrial_event(self):
        result = select_tags(
            "Начался новый этап создания европейского многоцелевого корвета"
        )
        self.assertEqual(result.outcome, "tagged")
        self.assertIn("судостроение", result.tags)

    def test_only_signed_agreement_is_not_a_completed_atomic_plant(self):
        result = select_tags(
            "Россия и Мьянма подписали межправительственное соглашение о "
            "строительстве атомной станции малой мощности"
        )
        self.assertEqual(result.outcome, "review")
        self.assertFalse(result.tags)

    def test_pipeline_without_oil_or_gas_is_not_assumed_to_be_oil_gas(self):
        result = select_tags(
            "Российские учёные повысили надёжность трубопроводной системы",
            "Исследователи повысили надёжность систем водоснабжения."
        )
        self.assertNotIn("нефтегазовая промышленность", result.tags)

    def test_plans_and_operations_do_not_masquerade_as_completed_manufacturing(self):
        cases = [
            "«Русал» и Гана подписали меморандум о совместном предприятии по добыче бокситов",
            "Калининградские «Продукты питания» хотят восстановить производство",
            "Чебоксарский речной порт запустит новый контейнерный кран в 2027 году",
            "Сталь не вывозит",
        ]
        for title in cases:
            with self.subTest(title=title):
                self.assertFalse(select_tags(title).tags)
    def test_full_text_must_not_turn_unrelated_company_events_into_industry(self):
        cases = [
            (
                "Двигатели ОДК обеспечили успешный старт ракеты «Союз» с космодрома Плесецк",
                "ОДК изготовила двигатели. Грузовой транспортный корабль Прогресс "
                "доставил топливо на космическую станцию.",
            ),
            (
                "Препарат Ростеха одобрили для лечения детей с нарушениями иммунной системы",
                "Ростех произвел лекарственный препарат, который запатентовали.",
            ),
            (
                "ТопФактор: экспертиза в управлении результативностью и развитии человеческого капитала",
                "Компания создала систему автоматизации кадрового документооборота.",
            ),
            (
                "Михаил Алексеев: «Для меня Ил-114-300 — это не просто самолет, а дело жизни»",
                "Конструктор рассказал, как создавался и производится самолет.",
            ),
            (
                "ОСК и Судпроф: открытый разговор о главном",
                "Представители профсоюза обсудили премии. Верфи строят суда.",
            ),
            (
                "Внешнее управление введено на борисоглебском «Борхиммаше»",
                "Завод производит оборудование и машиностроительные агрегаты.",
            ),
            (
                "Atyrau Oil & Gas 2026: диалог бизнеса и государства",
                "На форуме обсуждали нефтегазовое оборудование и добычу нефти.",
            ),
            (
                "В Красноярском крае создают платформу для кооперации производителей и заказчиков",
                "Компания создала портал для местных производителей. "
                "В регионе развивается машиностроение.",
            ),
        ]
        for title, text in cases:
            with self.subTest(title=title):
                result = select_tags(title, text=text)
                self.assertFalse(result.tags, result)
                self.assertNotEqual(result.outcome, "tagged")

    def test_full_article_does_not_tag_customer_or_machine_only_used(self):
        metal_parts = select_tags(
            "Ростех запустил серийное производство высокоточных алюминиевых деталей для радиоэлектроники",
            text="Производство алюминиевых деталей для радиоэлектроники запущено."
        )
        self.assertIn("технологическое производство", metal_parts.tags)
        self.assertNotIn("электротехническое направление", metal_parts.tags)
        automated_line = select_tags(
            'На "Янтаре" автоматизируют линию резки металла',
            text="На предприятии автоматизируют линию резки металла. "
                 "Используют станки и промышленное оборудование."
        )
        self.assertIn("технологическое производство", automated_line.tags)
        self.assertNotIn("машиностроение", automated_line.tags)

    def test_real_laboratory_is_technology_not_incidental_metal(self):
        result = select_tags(
            "«Норникель» представил в Китае ИИ-лабораторию новых материалов",
            text="Компания представила ИИ-лабораторию. "
                 "В дальнейшем исследуют палладиевые материалы."
        )
        self.assertEqual(result.outcome, "tagged")
        self.assertIn("технологии", result.tags)
        self.assertNotIn("металлургия", result.tags)

    def test_howto_with_colon_is_not_new_industrial_product(self):
        result = select_tags(
            "Автоматизация буровых работ: как заменить Excel на low-code",
            text="В статье рассказали про автоматизацию работы и ПО."
        )
        self.assertFalse(result.tags)

    def test_cannot_invent_sector_from_extreme_sbert_similarity(self):
        ranking = [
            {"tag": "авиация", "final_score": 999.0},
            {"tag": "горнодобывающая промышленность", "final_score": 999.0},
        ]
        decision = select_tags(
            "Тюменский завод ЖБИ восстанавливает работу после паводка",
            ranked=ranking,
        )
        self.assertEqual(decision.tags, ("заводы",))


if __name__ == "__main__":
    unittest.main()
