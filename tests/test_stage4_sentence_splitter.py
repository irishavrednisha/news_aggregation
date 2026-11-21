"""Russian sentence boundary regressions discovered in real Stage-4 reports."""
from __future__ import annotations

import unittest

from processing.abstractive_validator import sentences, validate_summary
from processing.qwen_evidence import build_claims, source_passages
from processing.qwen_semantic_claimwise import review_cluster_claimwise
from processing.qwen_abstractive_summary import canonical_materials, source_digest


def docs(body: str) -> list[dict]:
    return canonical_materials([{
        "id": "source-1", "source": "Издание",
        "url": "https://example.org/industrial-news",
        "title": "Завод имени Зверева наладил выпуск оборудования",
        "text": body, "published_at": "2026-11-21", "is_primary": True,
    }])


class SentenceBoundaryTests(unittest.TestCase):
    def test_personal_initials_not_isolated_claims(self):
        text = ("Красногорский завод им. С. А. Зверева обновил производство. "
                "На предприятии запустили линию. Выпуск оборудования вырос.")
        result = sentences(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0],
                         "Красногорский завод им. С. А. Зверева обновил производство.")
        self.assertEqual(result[1], "На предприятии запустили линию.")

    def test_abbreviations_and_place_names_do_not_break_claims(self):
        text = ("В г. Казани завод выпустил 15 тыс. шт. изделий. "
                "ООО «Техмаш» увеличило мощности. "
                "Второе предприятие начало строительство.")
        self.assertEqual(len(sentences(text)), 3)
        self.assertTrue(sentences(text)[0].startswith("В г. Казани"))
        self.assertIn("15 тыс. шт.", sentences(text)[0])

    def test_regular_three_sentences_still_split(self):
        self.assertEqual(sentences("Завод открылся. На площадке начали выпуск. "
                                   "Предприятие расширит производство."),
                         ["Завод открылся.", "На площадке начали выпуск.",
                          "Предприятие расширит производство."])

    def test_acronym_and_decimals_preserve_structure(self):
        text = ("ПАО «ОДК» внедрило новый агрегат мощностью 3,5 МВт. "
                "ООО «Вектор» выпустило 1000 установок. "
                "Компания сообщила об увеличении выпуска.")
        self.assertEqual(len(sentences(text)), 3)

    def test_evidence_quote_does_not_end_in_initial(self):
        source = ("Красногорский завод им. С. А. Зверева наладил выпуск "
                  "электронного оборудования на новом участке. "
                  "Первая партия продукции передана заказчику. "
                  "На предприятии модернизировали сборочную линию.")
        summary = ("Красногорский завод им. С. А. Зверева наладил выпуск "
                   "электронного оборудования. Первую партию передали заказчику. "
                   "Предприятие обновило сборочный участок.")
        parts = sentences(summary)
        self.assertEqual(len(parts), 3)
        claims = build_claims(summary, docs(source))
        self.assertEqual(len(claims), 3)
        self.assertTrue(any("им. С. А. Зверева" in x["evidence_quotes"][0]
                            for x in claims if x["evidence_quotes"]))
        self.assertTrue(all(claim["claim"].strip() not in ("С.", "А.")
                            for claim in claims))
        self.assertFalse(any(x["quote"].strip() in ("С.", "А.")
                             for x in source_passages(docs(source))))
        self.assertEqual(validate_summary(summary, docs(source)).sentence_count, 3)

    def test_nonempty_source_and_no_model_calls_when_no_summary(self):
        x = docs("Завод выпустил оборудование для шахт.")
        row = {"cluster_id": 1, "source_materials": x,
               "source_sha256": source_digest(x), "qwen_summary": "",
               "main_event": ""}
        reviewed = review_cluster_claimwise(row)
        self.assertEqual(reviewed["model_calls"], 0)
        self.assertEqual(reviewed["issues"], ["no_summary"])


if __name__ == "__main__":
    unittest.main()
