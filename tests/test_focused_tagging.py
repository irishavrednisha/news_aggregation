"""Проверки тэгирования по главному сюжету без скачивания SBERT и БД.

Запуск: python -m unittest discover -s tests -p 'test_focused_tagging.py' -v
Используются реальные реализации функций из tagger.py через AST
и заглушки только для эмбеддингов и запросов к базе.
"""
import ast
from functools import lru_cache
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock


TAGGER = Path(__file__).resolve().parents[1] / "processing" / "tagger.py"


def load_functions():
    tree = ast.parse(TAGGER.read_text(encoding="utf-8"))
    names = {
        "get_cluster_text_for_tagging", "cut_text",
        "score_tags_detailed", "_count_prepared_keywords",
        "_matching_prepared_keywords", "_normalize_tag_keyword",
        "select_tags_by_score_gap",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "re": re,
        "lru_cache": lru_cache,
        "Cluster": object,
        "lemmatize_text": lambda s: str(s).lower().replace("ё", "е"),
        "TAG_KEYWORDS": {},
        "TAG_NEGATIVE_KEYWORDS": {},
    }
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=nodes, type_ignores=[])
    ), str(TAGGER), "exec"), namespace)
    return namespace


class FocusedTaggingTests(unittest.TestCase):
    def test_keyword_requires_word_boundary(self):
        ns = load_functions()
        count = ns["_count_prepared_keywords"]
        self.assertEqual(count("газета и газеты", ["газ"]), 0)
        self.assertEqual(count("добыча газа и производство газ", ["газ"]), 1)

    def test_summary_is_more_important_than_incidental_body(self):
        ns = load_functions()
        marine = "судостроение"
        engineering = "инжиниринг"
        ns["get_tag_embeddings"] = lambda: {
            marine: marine, engineering: engineering,
        }
        ns["build_embedding"] = lambda text: text

        def similarity(vector, tag):
            # Основной сюжет о судостроении; в теле вскользь
            # упоминается инженерное проектирование.
            focus = "Ледокол" in vector
            if tag == marine:
                return 0.84 if focus else 0.30
            return 0.32 if focus else 0.94

        ns["cosine_sim"] = similarity
        ns["TAG_KEYWORDS"] = {
            marine: ["ледокол"],
            engineering: ["инженерный отдел", "проектирование"],
        }
        focus = "Ледокол прошёл испытания. Завод построил судно."
        body = (
            "В дополнительном контексте упоминаются инженерный отдел "
            "и проектирование, но главный сюжет о новом корабле."
        )
        text = "Ледокол прошёл испытания.\nЗавод построил судно.\n" + body
        scored = ns["score_tags_detailed"](text, focus_text=focus)
        self.assertEqual(scored[0]["tag"], marine)
        self.assertGreater(scored[0]["focus_similarity"],
                           scored[0]["body_similarity"])
        self.assertEqual(len(scored), 2)
        self.assertIn("focus_matches", scored[0])
        self.assertIn("ледокол", scored[0]["focus_keywords"])
        selected = ns["select_tags_by_score_gap"](
            scored, min_score=0.35, max_tags=2, max_gap_from_best=0.05,
        )
        self.assertEqual(selected, [marine])

    def test_body_alone_has_small_keyword_boost(self):
        ns = load_functions()
        ns["get_tag_embeddings"] = lambda: {"инжиниринг": "инжиниринг"}
        ns["build_embedding"] = lambda text: text
        ns["cosine_sim"] = lambda *_: 0.40
        ns["TAG_KEYWORDS"] = {
            "инжиниринг": [
                "конструктор", "проектирование", "чертеж",
                "инженер", "производство", "проект",
            ]
        }
        text = (
            "Основная тема\nКлючевой результат\n"
            "конструктор проектирование чертеж инженер производство проект"
        )
        score = ns["score_tags_detailed"](
            text, focus_text="Основная тема. Ключевой результат",
        )[0]
        self.assertLessEqual(score["keyword_boost"], 0.018)
        self.assertEqual(score["focus_matches"], 0)

    def test_cluster_text_has_no_repeated_title_or_summary(self):
        ns = load_functions()
        news = SimpleNamespace(
            title="Завод запустил линию",
            text="На предприятии начат выпуск комплектующих.",
        )
        fields = MagicMock()
        fields.cluster_id = 1
        fields.is_primary.desc.return_value = 1
        fields.published_at.asc.return_value.nullslast.return_value = 1
        fields.collected_at.asc.return_value = 1
        ns["News"] = fields
        session = MagicMock()
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [news]
        cluster = SimpleNamespace(
            id=7, title="Выпущен новый ледокол",
            summary_text="На верфи закончено строительство судна.",
        )
        text = ns["get_cluster_text_for_tagging"](session, cluster)
        self.assertEqual(text.splitlines()[:2], [
            cluster.title, cluster.summary_text,
        ])
        self.assertEqual(text.count(cluster.title), 1)
        self.assertEqual(text.count(cluster.summary_text), 1)
        self.assertIn(news.text, text)


if __name__ == "__main__":
    unittest.main()
