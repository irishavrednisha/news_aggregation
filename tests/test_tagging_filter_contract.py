"""Тестируем фильтрацию внутри тегирования БЕЗ загрузки SBERT и PostgreSQL.

Загружаем только функции tag_cluster/is_not_publication_by_keywords из AST:
этот тест проверяет реальную реализацию с подменёнными зависимостями.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

TAGGER_PATH = Path(__file__).resolve().parent.parent / "processing" / "tagger.py"


def load_tagger_functions():
    tree = ast.parse(TAGGER_PATH.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"tag_cluster", "is_not_publication_by_keywords", "get_cluster_text_for_tagging"}
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {
        "Cluster": type("Cluster", (), {"id": 1}),
        "os": os,
    }
    exec(compile(ast.fix_missing_locations(module), str(TAGGER_PATH), "exec"),
         namespace)
    return namespace


class TaggerFilteringContractTests(unittest.TestCase):
    def prepare(self, title, summary):
        ns = load_tagger_functions()
        cluster = SimpleNamespace(
            id=1, title=title, summary_text=summary,
            tags=None, status="new",
        )
        session = Mock()
        session.query.return_value.filter.return_value.first.return_value = cluster
        ns.update({
            "is_not_related_to_russia": Mock(return_value=False),
            "get_cluster_text_for_tagging": Mock(
                return_value=title + ". На заводе установили станок. "
                             "На прошлой выставке показывали прототип. "
                             "Предприятие создало новую производственную линию."),
            "cluster_has_content": Mock(return_value=True),
            "score_tags_detailed": Mock(return_value=[
                {"tag": "машиностроение", "final_score": 0.75,
                 "embedding_score": 0.70,
                 "keyword_boost": 0.05, "keyword_penalty": 0},
            ]),
            "select_tags_by_score_gap": Mock(return_value=["машиностроение"]),
        })
        return ns, cluster, session

    def test_incidental_exhibition_must_not_block_industrial_cluster(self):
        ns, cluster, session = self.prepare(
            "Завод запустил новый станок после международной выставки",
            "На заводе началось производство инновационных станков."
        )
        selected = ns["tag_cluster"](session, 1, debug=False)
        self.assertEqual(selected, ["машиностроение"])
        self.assertEqual(cluster.status, "ready")
        self.assertEqual(cluster.tags, "машиностроение")
        ns["score_tags_detailed"].assert_not_called()
        ns["is_not_related_to_russia"].assert_not_called()
        ns["get_cluster_text_for_tagging"].assert_not_called()

    def test_ambiguous_title_stays_out_of_publication(self):
        ns, cluster, session = self.prepare(
            "Сталь не вывозит",
            "Компания рассказала об изменениях спроса."
        )
        tags = ns["tag_cluster"](session, 1, debug=False)
        self.assertEqual(tags, [])
        self.assertEqual(cluster.tags, None)
        self.assertEqual(cluster.status, "tagging_review")
        ns["score_tags_detailed"].assert_not_called()

    def test_direct_factory_event_does_not_load_sbert(self):
        ns, cluster, session = self.prepare(
            "Тюменский завод ЖБИ восстанавливает работу после паводка",
            "Производство железобетонных изделий восстанавливается."
        )
        ns["score_tags_detailed"].side_effect = RuntimeError("SBERT недоступен")
        tags = ns["tag_cluster"](session, 1, debug=False)
        self.assertEqual(tags, ["заводы"])
        self.assertEqual(cluster.status, "ready")
        ns["score_tags_detailed"].assert_not_called()

    def test_geography_is_not_a_required_tagger_condition(self):
        ns, cluster, session = self.prepare(
            "В Южной Корее построили новый корабль",
            "Построен корабль для исследовательских работ."
        )
        ns["is_not_related_to_russia"].side_effect = RuntimeError(
            "This geographic check must not run in grounded"
        )
        selected = ns["tag_cluster"](session, 1, debug=False)
        self.assertEqual(selected, ["судостроение"])
        self.assertEqual(cluster.status, "ready")
        ns["is_not_related_to_russia"].assert_not_called()

    def test_corporate_pr_title_filtered_in_tagger(self):
        ns, cluster, session = self.prepare(
            "Социальная деятельность компании отмечена престижными наградами",
            "Компания получила премии."
        )
        selected = ns["tag_cluster"](session, 1, debug=False)
        self.assertEqual(selected, [])
        self.assertEqual(cluster.status, "not_for_publication")
        ns["score_tags_detailed"].assert_not_called()

    def test_summary_drives_tags_without_accessing_source_body(self):
        ns, cluster, session = self.prepare(
            "На предприятии модернизировали производственную линию",
            "На предприятии модернизировали электрические трансформаторы."
        )
        ns["get_cluster_text_for_tagging"].side_effect = AssertionError(
            "grounded must not fetch the source article text"
        )
        ns["cluster_has_content"].side_effect = AssertionError(
            "grounded does not require article text to classify a ready summary"
        )
        tags = ns["tag_cluster"](session, 1, debug=False)
        self.assertEqual(tags[0], "электротехническое направление")
        self.assertEqual(cluster.status, "ready")
        ns["get_cluster_text_for_tagging"].assert_not_called()
        ns["cluster_has_content"].assert_not_called()

    def test_missing_summary_does_not_continue_to_tag(self):
        ns, cluster, session = self.prepare(
            "На заводе запустили линию", None
        )
        selected = ns["tag_cluster"](session, 1, debug=False)
        self.assertEqual(selected, [])
        self.assertEqual(cluster.status, "not_for_publication")
        ns["score_tags_detailed"].assert_not_called()
        ns["is_not_related_to_russia"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
