"""Офлайн-проверка AI-tagging без скачивания весов и без БД."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from processing.ai_tagger import (
    AITagResult, EVENT_LABELS, INDUSTRIAL_EVENTS, TAG_LABELS,
    make_ai_input, predict_tags,
)
from test_tagging_filter_contract import load_tagger_functions


class FakeNLI:
    def __init__(self, event="industrial_production", event_score=0.78,
                 tag_scores=None):
        self.event = event
        self.event_score = event_score
        self.tag_scores = tag_scores or {}
        self.calls = []

    def __call__(self, text, candidate_labels, hypothesis_template, multi_label):
        self.calls.append((text, list(candidate_labels), multi_label))
        if set(candidate_labels) == set(EVENT_LABELS.values()):
            rem = (1 - self.event_score) / (len(EVENT_LABELS) - 1)
            scores = {
                label: (self.event_score if key == self.event else rem)
                for key, label in EVENT_LABELS.items()
            }
        else:
            rem = (1 - sum(self.tag_scores.values()))
            unprovided = len(candidate_labels) - len(self.tag_scores)
            if unprovided:
                rem /= unprovided
            scores = {
                label: self.tag_scores.get(label, rem)
                for label in candidate_labels
            }
        return {
            "labels": list(reversed(candidate_labels)),
            "scores": [scores[label] for label in reversed(candidate_labels)],
        }


TITLE = "В России открыли новый металлургический завод"
SUMMARY = ("Предприятие ввело сталеплавильную печь и организовало "
           "серийный выпуск отечественного металлопроката.")


class AITaggerTests(unittest.TestCase):
    def test_exactly_same_18_tag_names_as_existing_tagger(self):
        source = Path(__file__).resolve().parents[1] / "processing" / "tagger.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        tags = next(
            x for x in tree.body
            if isinstance(x, ast.Assign)
            and any(isinstance(y, ast.Name) and y.id == "TAGS" for y in x.targets)
        )
        existing = ast.literal_eval(tags.value)
        self.assertEqual(set(TAG_LABELS), set(existing))
        self.assertEqual(len(TAG_LABELS), 18)
        self.assertEqual(len(set(TAG_LABELS.values())), 18)

    def test_input_has_title_summary_once_and_short_lead(self):
        text = make_ai_input(
            TITLE, SUMMARY, "Начало статьи с подробностями модернизации"
        )
        self.assertEqual(text.count(TITLE), 1)
        self.assertEqual(text.count(SUMMARY), 1)
        self.assertIn("Начало статьи:", text)

    def test_two_competing_industrial_tags_selected(self):
        fake = FakeNLI(tag_scores={
            TAG_LABELS["металлургия"]: 0.50,
            TAG_LABELS["заводы"]: 0.32,
            TAG_LABELS["инновации"]: 0.05,
        })
        result = predict_tags(
            TITLE, SUMMARY, classifier=fake,
            max_tags=2, tag_threshold=0.22, relevance_threshold=0.60,
        )
        self.assertEqual(result.tags, ["металлургия", "заводы"])
        self.assertEqual(result.event_type, "industrial_production")
        self.assertEqual(result.reason, "tagged")
        self.assertEqual(len(fake.calls), 2)
        # ВАЖНО: для главного события и для тегов
        # используем сравнительное, а не независимое NLI.
        self.assertFalse(fake.calls[0][2])
        self.assertFalse(fake.calls[1][2])

    def test_staff_training_not_tagged_as_engineering(self):
        fake = FakeNLI(
            event="staff_training", event_score=0.80,
            tag_scores={TAG_LABELS["инжиниринг"]: 0.75},
        )
        result = predict_tags(
            "Руководители судостроительных проектов прошли обучение",
            "В РАНХиГС провели курс проектного управления.",
            classifier=fake,
        )
        self.assertEqual(result.tags, [])
        self.assertEqual(result.reason, "no_industrial_relevance")
        self.assertEqual(result.event_type, "staff_training")
        self.assertEqual(len(fake.calls), 1)

    def test_robotics_competition_not_automatically_machinery(self):
        fake = FakeNLI(event="competitions", event_score=0.76)
        result = predict_tags(
            "В Казани стартует финал «Битвы роботов»",
            "Участники инженерного соревнования покажут своих роботов.",
            classifier=fake,
        )
        self.assertEqual(result.tags, [])
        self.assertEqual(result.event_type, "competitions")
        self.assertEqual(len(fake.calls), 1)

    def test_nonindustrial_article_skips_tag_scoring(self):
        fake = FakeNLI(event="consumer_promotion", event_score=0.92)
        result = predict_tags(
            "В магазине скидка на кофе", "Посетители воспользовались акцией.",
            classifier=fake
        )
        self.assertEqual(result.tags, [])
        self.assertEqual(result.reason, "no_industrial_relevance")
        self.assertEqual(len(fake.calls), 1)

    def test_no_tag_above_threshold_no_publication(self):
        fake = FakeNLI(event_score=0.96)
        result = predict_tags(TITLE, SUMMARY, classifier=fake)
        self.assertEqual(result.tags, [])
        self.assertEqual(result.reason, "no_confident_industrial_tag")
        self.assertEqual(len(result.ranked_tags), 18)
        self.assertAlmostEqual(sum(x[1] for x in result.ranked_tags), 1.0)

    def test_missing_summary_does_not_load_model(self):
        fake = FakeNLI()
        result = predict_tags(TITLE, "", classifier=fake)
        self.assertEqual(result.reason, "missing_title_or_summary")
        self.assertEqual(fake.calls, [])

    def test_invalid_scores_are_not_accepted(self):
        fake = FakeNLI(event_score=float("nan"))
        with self.assertRaises(ValueError):
            predict_tags(TITLE, SUMMARY, classifier=fake)

    def test_out_of_range_threshold_rejected(self):
        with self.assertRaises(ValueError):
            predict_tags(TITLE, SUMMARY, classifier=FakeNLI(), tag_threshold=1.4)

    def test_ambiguous_event_without_clear_majority_is_not_published(self):
        # Возможная неоднозначная новость: отраслевые термины есть,
        # но ни один главный тип события не доминирует.
        fake = FakeNLI(event_score=0.10)
        result = predict_tags(TITLE, SUMMARY, classifier=fake)
        self.assertEqual(result.tags, [])
        self.assertEqual(result.reason, "no_industrial_relevance")
        self.assertEqual(len(fake.calls), 1)

    def test_model_integration_writes_tags_and_ready(self):
        ns = load_tagger_functions()
        cluster = SimpleNamespace(
            id=12, title=TITLE, summary_text=SUMMARY, tags=None, status="new"
        )
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = cluster
        # news is used only to obtain the first nonempty lead.
        primary = SimpleNamespace(text="Дополнительная информация о заводе.")
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = primary
        entity = MagicMock()
        ns.update({
            "News": entity,
            "is_not_related_to_russia": Mock(return_value=False),
            "cluster_has_content": Mock(return_value=True),
        })
        expected = AITagResult(
            ["металлургия"], "tagged", 0.89, [("металлургия", 0.91)]
        )
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "nli"}):
            with patch("processing.ai_tagger.predict_tags", return_value=expected) as ai:
                result = ns["tag_cluster"](session, 12, debug=False)
        self.assertEqual(result, ["металлургия"])
        self.assertEqual(cluster.tags, "металлургия")
        self.assertEqual(cluster.status, "ready")
        ai.assert_called_once()
        session.commit.assert_called_once()

    def test_model_failure_never_marks_ready(self):
        ns = load_tagger_functions()
        cluster = SimpleNamespace(
            id=12, title=TITLE, summary_text=SUMMARY,
            tags="заводы", status="ready"
        )
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = cluster
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        ns.update({
            "News": MagicMock(),
            "is_not_related_to_russia": Mock(return_value=False),
            "cluster_has_content": Mock(return_value=True),
        })
        with patch.dict(os.environ, {"NEWS_TAGGER_BACKEND": "nli"}):
            with patch("processing.ai_tagger.predict_tags", side_effect=RuntimeError("test unavailable")):
                result = ns["tag_cluster"](session, 12, debug=False)
        self.assertEqual(result, [])
        self.assertIsNone(cluster.tags)
        self.assertEqual(cluster.status, "new")


if __name__ == "__main__":
    unittest.main()
