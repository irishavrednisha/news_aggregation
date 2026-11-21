"""Offline tests: cross-source recovery without inventing evidence or dates."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.stage4_reconcile_multisource import (
    _fetch_new_article, propose_pairs, reconcile, reconcile_then_generate,
)
from scripts.stage4_independent_holdout import atomic_json


def article(idnum: int, source: str, headline: str, description: str) -> dict:
    return {
        "id": f"{source}-{idnum}", "source": source,
        "title": headline,
        "url": f"https://{source.lower()}.test/news/{idnum}",
        "text": (description + " ") * 12,
        "published_at": "2026-11-20T09:00:00",
        "text_method": "html", "is_primary": True,
    }


def originals():
    pairs = [
        ("Ковдорский ГОК запускает новый конвейерный комплекс",
         "Ковдорский ГОК меняет схему перевозки вскрыши на ЦПТ",
         "Горно-обогатительный комбинат ЕвроХима переходит на циклично-поточную "
         "технологию транспортировки вскрыши. Конвейер запустят в 2027 году."),
        ("Росатом готовит к запуску центр переработки ядерного топлива",
         "Росатом готовится начать переработку ядерного топлива в Железногорске",
         "В Железногорске специалисты Росатома завершили отладку оборудования "
         "для переработки отработавшего ядерного топлива."),
        ("В ОАЭ открылось производство нижегородской компании Хромос",
         "Нижегородский Хромос открыл производственную площадку в Дубае",
         "В Дубае компания Хромос открыла производственную площадку и "
         "представила промышленное хроматографическое оборудование."),
        ("Госдума приняла законопроект о контроле золотодобычи",
         "Госдума в первом чтении приняла законопроект о контроле добычи золота",
         "Госдума приняла в первом чтении законопроект о контроле за добычей "
         "золота по геологоразведочным лицензиям."),
    ]
    rows = []
    for i, (title_a, title_b, lead) in enumerate(pairs, 1):
        rows.append({"cluster_id": 9000+i, "source_materials": [
            article(i, "PublisherA", title_a, lead)]})
        rows.append({"cluster_id": 9100+i, "source_materials": [
            article(i, "PublisherB", title_b, lead)]})
    return rows


class RecoveryTests(unittest.TestCase):
    def test_strong_two_source_event_candidates_are_not_claimed_human_verified(self):
        pairs = propose_pairs(originals(), title_min=.1, lead_min=.20)
        self.assertGreaterEqual(len(pairs), 4)
        self.assertTrue(all(not p["event_identity_verified"] for p in pairs))
        self.assertTrue(all(p["requires_human_event_identity_review"] for p in pairs))

    def test_nearby_technical_topic_is_not_same_event(self):
        rows = [
            {"cluster_id": 1, "source_materials": [article(
                1,"PublisherA","Завод начал строительство нового конвейера",
                "Компания строит новую угольную фабрику.") ]},
            {"cluster_id": 2, "source_materials": [article(
                2,"PublisherB","Завод завершил строительство нового конвейера",
                "Предприятие закончило сборку железнодорожного состава.") ]},
        ]
        self.assertEqual(propose_pairs(rows), [])

    def test_external_supplement_cannot_invent_a_date(self):
        def fake(preview):
            return SimpleNamespace(
                source=preview.source,title=preview.title,url=preview.url,
                text="Надёжно собранный текст " * 30,
                published_at=None,text_source="html")
        result = _fetch_new_article({
            "source":"Additional publisher",
            "url":"https://publisher.test/event",
            "title":"На предприятии модернизировали оборудование",
        },fetch=fake)
        self.assertIsNone(result["published_at"])

    def test_verified_https_required_for_supplement(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            _fetch_new_article({"source":"Publisher", "url":"http://example.test/path"})

    def test_frozen_input_never_modified_and_bad_fifth_page_does_not_pass(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            report = {"stage":4,"clusters":[]}
            old = root/"historic.json"
            atomic_json(old, report)
            from hashlib import sha256
            manifest = {
                "stage":4, "kind":"stage4-independent-holdout-v1",
                "evaluation_date":"2026-11-20", "previous_files":[{
                    "path":"historic.json",
                    "sha256":sha256(old.read_bytes()).hexdigest(),
                }],
                "max_age_days":30, "required_minimum": 4,
                "minimum_sample_met":True, "sample_size":8,
                "selection_method":"fresh dated full HTML",
                "selection_stats":{"missing_publication_dates":0},
                "clusters":originals(),
            }
            src=root/"old"/"holdout_manifest.json"
            src.parent.mkdir()
            atomic_json(src,manifest)
            before=src.read_bytes()
            out=root/"new"
            specs=root/"supplement.json"
            atomic_json(specs,{"articles":[{
                "target_cluster_id":9001,"source":"Other",
                "url":"https://new.test/news/1",
                "title":"Новый пресс для завода",
            }]})
            def fake(preview):
                return SimpleNamespace(
                    source=preview.source,title=preview.title,url=preview.url,
                    text="Текст нового завода " * 40,
                    published_at=None,text_source="html",
                )
            result = reconcile(src,out,supplement_specs=specs,
                               root=root,fetch=fake)
            self.assertEqual(src.read_bytes(),before)
            self.assertEqual(result["supplemental_sources"],0)
            self.assertEqual(result["recovered_pairs"],4)
            self.assertFalse(result["gate"]["ready_for_model_generation"])
            self.assertIn("fewer_than_required_multi_document_events",
                          result["gate"]["errors"])
            self.assertEqual(result["result_sample_size"],4)

    def test_one_command_model_gate_blocks_generation(self):
        with patch("scripts.stage4_reconcile_multisource.reconcile",
                   return_value={"gate":{"ready_for_model_generation":False}}):
            with patch("scripts.stage4_fresh_qwen_eval.generate") as model:
                result = reconcile_then_generate(
                    Path("old.json"),Path("new"),generate_after=True)
                model.assert_not_called()
        self.assertFalse(result["generation"]["model_called"])

    def test_one_command_generates_only_after_successful_gate(self):
        with patch("scripts.stage4_reconcile_multisource.reconcile",
                   return_value={"gate":{"ready_for_model_generation":True}}):
            with patch("scripts.stage4_fresh_qwen_eval.generate",
                       return_value={"complete":True}) as model:
                result = reconcile_then_generate(
                    Path("old.json"),Path("new"),generate_after=True)
                model.assert_called_once()
        self.assertTrue(result["generation"]["complete"])

    def test_double_run_never_overwrites_frozen_manifest(self):
        with TemporaryDirectory() as td:
            src=Path(td)/"same"/"holdout_manifest.json"
            src.parent.mkdir()
            src.write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):
                reconcile(src,src.parent)


if __name__ == "__main__":
    unittest.main()
