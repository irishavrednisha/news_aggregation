"""Production hybrid opt-in contract tested without SBERT and PostgreSQL.

Loads the actual tag_cluster function from AST; all DB/embedding operations
are mocked. The challenger is real, but is a pure Python module.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / "processing" / "tagger.py"


def load_tag_cluster():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "tag_cluster"]
    if len(nodes) != 1:
        raise AssertionError("Expected exactly one tag_cluster definition")
    ns = {
        "Cluster": type("Cluster", (), {"id": 1}),
        "os": os,
        "is_not_related_to_russia": Mock(return_value=False),
        "cluster_has_content": Mock(return_value=True),
        "get_cluster_text_for_tagging": Mock(return_value="Article text"),
        "is_not_publication_by_keywords": Mock(return_value=False),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(PATH), "exec"), ns)
    return ns


def mock_cluster(ns, title, summary, ranked, default_tags):
    cluster = SimpleNamespace(
        id=17, title=title, summary_text=summary,
        status="new", tags=None,
    )
    session = Mock()
    session.query.return_value.filter.return_value.first.return_value = cluster
    ns["score_tags_detailed"] = Mock(return_value=ranked)
    ns["select_tags_by_score_gap"] = Mock(return_value=default_tags)
    return cluster, session


class HybridTaggerIntegrationTests(unittest.TestCase):
    def test_hybrid_corrects_secondary_subject_without_changing_sbert_scores(self):
        ns = load_tag_cluster()
        title = "В Невельске возрождают производство по переработке морепродуктов"
        summary = "Компания управляет морскими терминалами и 51 судном."
        ranked = [
            {"tag": "судостроение", "final_score": 0.51},
            {"tag": "заводы", "final_score": 0.40},
            {"tag": "технологическое производство", "final_score": 0.276},
        ]
        cluster, session = mock_cluster(ns, title, summary, ranked, ["судостроение"])
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "hybrid"}):
            selected = ns["tag_cluster"](session, 17, min_score=0.35,
                                         max_tags=2, debug=False)
        self.assertEqual(selected, ["заводы", "технологическое производство"])
        self.assertEqual(cluster.tags, "заводы, технологическое производство")
        self.assertEqual(cluster.status, "ready")
        ns["score_tags_detailed"].assert_called_once()
        ns["select_tags_by_score_gap"].assert_called_once()
        session.commit.assert_called_once()

    def test_legacy_remains_unchanged(self):
        ns = load_tag_cluster()
        title = "В Невельске возрождают производство по переработке морепродуктов"
        summary = "Компания управляет морскими терминалами и 51 судном."
        ranked = [
            {"tag": "судостроение", "final_score": 0.51},
            {"tag": "заводы", "final_score": 0.40},
            {"tag": "технологическое производство", "final_score": 0.276},
        ]
        cluster, session = mock_cluster(ns, title, summary, ranked, ["судостроение"])
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "legacy"}):
            selected = ns["tag_cluster"](session, 17, min_score=0.35,
                                         max_tags=2, debug=False)
        self.assertEqual(selected, ["судостроение"])
        self.assertEqual(cluster.tags, "судостроение")
        self.assertEqual(cluster.status, "ready")

    def test_gemma_review_status_survives_legacy_tagging(self):
        ns = load_tag_cluster()
        cluster, session = mock_cluster(
            ns, "Завод выпустил новый станок", "Предприятие запустило линию.",
            [{"tag": "машиностроение", "final_score": 0.6}],
            ["машиностроение"],
        )
        cluster.status = "review_required"
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "legacy"}):
            selected = ns["tag_cluster"](session, 17, debug=False)
        self.assertEqual(selected, ["машиностроение"])
        self.assertEqual(cluster.status, "review_required")
        self.assertEqual(cluster.tags, "машиностроение")

    def test_gemma_review_status_survives_hybrid_tagging(self):
        ns = load_tag_cluster()
        cluster, session = mock_cluster(
            ns, "Завод выпустил новый станок", "Предприятие запустило линию.",
            [{"tag": "машиностроение", "final_score": 0.6}],
            ["машиностроение"],
        )
        cluster.status = "review_required"
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "hybrid"}):
            ns["tag_cluster"](session, 17, debug=False)
        self.assertEqual(cluster.status, "review_required")

    def test_hybrid_keeps_genuine_aircraft_manufacturing(self):
        ns = load_tag_cluster()
        ranked = [
            {"tag": "авиация", "final_score": 0.51},
            {"tag": "машиностроение", "final_score": 0.28},
        ]
        title = "Конструкторы испытали новое крыло беспилотного летательного аппарата"
        summary = "Разработчики подтвердили аэродинамику опытного БПЛА."
        cluster, session = mock_cluster(ns, title, summary, ranked, ["авиация"])
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "hybrid"}):
            selected = ns["tag_cluster"](session, 17, min_score=0.35,
                                         max_tags=2, debug=False)
        self.assertEqual(selected, ["авиация"])
        self.assertEqual(cluster.status, "ready")

    def test_hybrid_removes_aviation_from_field_use_of_drones(self):
        ns = load_tag_cluster()
        title = "Дроны следят за пожарами"
        summary = "Беспилотники мониторят леса и передают снимки спасателям."
        ranked = [
            {"tag": "авиация", "final_score": 0.48},
            {"tag": "технологии", "final_score": 0.27},
        ]
        cluster, session = mock_cluster(ns, title, summary, ranked, ["авиация"])
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "hybrid"}):
            selected = ns["tag_cluster"](session, 17, min_score=0.35,
                                         max_tags=2, debug=False)
        self.assertEqual(selected, [])
        self.assertEqual(cluster.tags, None)
        self.assertEqual(cluster.status, "not_for_publication")

    def test_nli_configuration_and_invalid_backend_stay_separate(self):
        ns = load_tag_cluster()
        cluster, session = mock_cluster(
            ns, "Завод выпустил новый станок", "Предприятие приступило к выпуску.",
            [{"tag": "машиностроение", "final_score": 0.6}],
            ["машиностроение"],
        )
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "invalid"}):
            with self.assertRaisesRegex(ValueError, "legacy, hybrid или nli"):
                ns["tag_cluster"](session, 17, debug=False)
        ns["score_tags_detailed"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
