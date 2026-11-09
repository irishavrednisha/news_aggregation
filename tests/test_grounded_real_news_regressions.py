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
            text='На атомном ледоколе "Чукотка" был выполнен физический пуск реактора РИТМ-200.'
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

    def test_summary_and_full_body_are_both_used_without_secondary_topic_noise(self):
        # The relevant equipment is identified beyond the former 550-char
        # lead window. An unrelated aircraft paragraph is not a second tag.
        title = "Предприятие модернизировало технологический процесс"
        summary = "Специалисты изменили производственный процесс предприятия."
        body = (
            "Предприятие модернизировало технологический процесс. " +
            "Подготовлены новые производственные инструкции. " * 25 +
            "Предприятие модернизировало электродвигатель оборудования. " +
            "Другая компания выпустила самолёт на своём предприятии."
        )
        result = select_tags(title, summary, text=body)
        self.assertEqual(result.outcome, "tagged")
        self.assertIn("электротехническое направление", result.tags, result)
        self.assertNotIn("авиация", result.tags, result)

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
