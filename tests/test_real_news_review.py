"""Offline contract tests for independent human-reviewed validation/test data."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.real_news_review import (
    FIELDS, url_hash, load_previous_hashes,
    select_articles, prepare, finalize,
)


EXAMPLE_TITLE = "Новая производственная линия по выпуску промышленных насосов"
EXAMPLE_TEXT = (
    "На предприятии в Туле открыли участок производства промышленных "
    "насосов для водоочистных станций. Новая линия включает пять "
    "обрабатывающих центров и испытательный стенд. "
    "На первой стадии инженеры изготовили опытную серию насосов "
    "и выполнили комплекс заводских испытаний. "
    "Сотрудники зафиксировали стабильные параметры работы "
    "оборудования и завершили проверку качества."
)
NEG_TEXT = (
    "Совет директоров промышленного предприятия обсудил "
    "корпоративную стратегию и назначил нового директора. "
    "Компания сообщила о проведении конференции, встрече "
    "с акционерами и подготовке годового отчёта. "
    "Информация о новом выпуске или разработке изделий "
    "в пресс-релизе отсутствует. Встреча не была связана с запуском "
    "оборудования, технической разработкой или модернизацией завода."
)
QUOTE1 = (
    "На предприятии в Туле открыли участок производства промышленных "
    "насосов для водоочистных станций."
)
QUOTE2 = (
    "На первой стадии инженеры изготовили опытную серию насосов "
    "и выполнили комплекс заводских испытаний."
)
SUMMARY = (
    "На тульском предприятии начали выпуск насосов для водоочистных "
    "станций. Инженеры собрали опытную партию и проверили работу "
    "оборудования на испытательном стенде."
)


def news_file(path):
    path.write_text(json.dumps({"articles": [
        {"title": EXAMPLE_TITLE, "text": EXAMPLE_TEXT,
         "source": "Тестовый источник 1",
         "url": "https://example.org/new-pumps-2026",
         "published_at": "2026-11-09"},
        {"title": "Компания сменила генерального директора",
         "text": NEG_TEXT, "source": "Тестовый источник 2",
         "url": "https://example.org/appointment-2026",
         "published_at": "2026-11-09"},
        {"title": "РУСАЛ создает производство ценного редкоземельного металла скандия",
         "text": EXAMPLE_TEXT, "source": "Тестовый источник 3",
         "url": "https://iapn.ru/rusal-sozdaet-proizvodstvo-czennogo-redkozemelnogo-metalla-skandiya/",
         "published_at": "2026-11-09"},
    ]}, ensure_ascii=False), encoding="utf-8")


class IndependentReviewTests(unittest.TestCase):
    def test_prior_125_exclusions_present(self):
        prior = load_previous_hashes()
        self.assertEqual(len(prior), 125)
        self.assertIn(url_hash(
            "https://iapn.ru/rusal-sozdaet-proizvodstvo-czennogo-redkozemelnogo-metalla-skandiya/"
        ), prior)

    def test_selection_excludes_prior_news_and_splits_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "new.json"
            news_file(f)
            chosen, dropped = select_articles(
                [f], load_previous_hashes(), [], min_text=100
            )
            self.assertEqual(len(chosen), 2)
            self.assertEqual(dropped["previous_eval_or_duplicate_url"], 1)
            self.assertEqual({a["split"] for a in chosen} <= {
                "validation", "test"
            }, True)
            again, _ = select_articles([f], load_previous_hashes(), [], min_text=100)
            self.assertEqual(chosen, again)

    def test_reviewed_data_only_writes_validation_and_test(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "new.json"
            news_file(source)
            folder = root / "review"
            manifest = prepare([source], [], folder)
            self.assertEqual(manifest["candidates"], 2)
            rows = []
            with (folder / "review_sheet.csv").open(
                "r", newline="", encoding="utf-8-sig"
            ) as fd:
                rows = list(csv.DictReader(fd, delimiter=";"))
            for row in rows:
                row["review_status"] = "verified"
                if "производственная линия" in row["title"]:
                    row["relevant"] = "yes"
                    row["tags"] = "машиностроение"
                    row["evidence_1"] = QUOTE1
                    row["evidence_2"] = QUOTE2
                    row["summary"] = SUMMARY
                else:
                    row["relevant"] = "no"
            with (folder / "review_sheet.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as fd:
                writer = csv.DictWriter(fd, fieldnames=FIELDS, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)

            train = root / "news_sft" / "train.jsonl"
            train.parent.mkdir(parents=True)
            train.write_text("sentinel-train-remains-unchanged\n", encoding="utf-8")
            outcome = finalize(folder, train.parent)
            self.assertEqual(outcome["verified_articles"], 2)
            self.assertEqual(outcome["train_touched"], False)
            self.assertEqual(
                train.read_text(encoding="utf-8"),
                "sentinel-train-remains-unchanged\n",
            )
            validated = json.loads(
                (train.parent / "real_gold_verified.json").read_text(encoding="utf-8")
            )["examples"]
            self.assertEqual(len(validated), 2)
            self.assertTrue(all(x["origin"] == "real" for x in validated))
            self.assertEqual(
                outcome["validation_messages"] + outcome["test_messages"], 3
            )

    def test_differently_named_syndicated_report_is_removed(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "new.json"
            first = {
                "title": "Начат выпуск новых промышленных насосов для водоочистки",
                "text": EXAMPLE_TEXT * 3,
                "source": "Первое издание", "url": "https://example.org/a-independent",
                "published_at": "2026-11-09",
            }
            second = {
                "title": "На новом участке инженеры провели испытания техники",
                "text": EXAMPLE_TEXT * 3,
                "source": "Второе издание", "url": "https://example.org/b-independent",
                "published_at": "2026-11-09",
            }
            source.write_text(json.dumps({"articles": [first, second]},
                                         ensure_ascii=False), encoding="utf-8")
            chosen, dropped = select_articles([source], load_previous_hashes(), [])
            self.assertEqual(len(chosen), 1)
            self.assertEqual(dropped["same_story_body_in_new_batch"], 1)

    def test_prepare_never_overwrites_existing_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "new.json"
            news_file(source)
            folder = root / "review"
            prepare([source], [], folder)
            sheet = folder / "review_sheet.csv"
            index = folder / "source_index.json"
            saved_sheet = sheet.read_bytes()
            saved_index = index.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "уже есть данные разметки"):
                prepare([source], [], folder)
            self.assertEqual(sheet.read_bytes(), saved_sheet)
            self.assertEqual(index.read_bytes(), saved_index)

    def test_drafts_never_become_gold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "new.json"
            news_file(source)
            folder = root / "review"
            prepare([source], [], folder)
            with self.assertRaisesRegex(ValueError, "Нет вручную подтверждённых"):
                finalize(folder, root / "sft")


if __name__ == "__main__":
    unittest.main()
