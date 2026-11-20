"""Tests only: synthetic fixtures do not count as real holdout articles."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.stage4_fresh_qwen_eval import (
    _multi, _sha, check_manifest, score, write_blind_packets, main, generate, VERSION,
)
from scripts.stage4_independent_holdout import atomic_json
from processing.qwen_abstractive_summary import source_digest


def sample(index, *, multi=False):
    docs = [{
        "id": str(100+index), "source": "Site A",
        "url": f"https://site-a.test/news/{index}",
        "title": f"Завод производитель {index} открыл новую линию",
        "text": (f"На заводе {index} запустили новую технологическую линию. "
                 "Выпуск оборудования обеспечат после испытаний. " * 8),
        "published_at": "2026-11-21", "is_primary": True,
    }]
    if multi:
        docs.append({
            "id": str(300+index), "source": "Site B",
            "url": f"https://site-b.test/report/{index}",
            "title": f"На заводе {index} заработало оборудование",
            "text": (f"Предприятие {index} сообщило о запуске новой линии. "
                     "Испытания оборудования завершены. " * 8),
            "published_at": "2026-11-21", "is_primary": False,
        })
    return {"cluster_id": 900000+index, "source_materials": docs}


def manifest(*, count=45, multis=6, previous=True):
    rows = [sample(i, multi=i<=multis) for i in range(1, count+1)]
    return {
        "stage": 4, "kind": "stage4-independent-holdout-v1",
        "max_age_days": 30,
        "minimum_sample_met": count >= 40,
        "selection_stats": {"missing_publication_dates": 0},
        "previous_files": [{"path": "prior.json", "sha256": "abc"}] if previous else [],
        "clusters": rows,
    }


def make_report(m):
    rows = []
    for r in m["clusters"]:
        rows.append({
            "cluster_id": r["cluster_id"],
            "status": "summary_draft",
            "source_sha256": source_digest(r["source_materials"]),
            "qwen_summary": ("Предприятие запустило новую линию. "
                             "Оборудование успешно испытали. "
                             "Выпуск намечен на следующий месяц."),
        })
    return {"stage": 4, "model": "qwen3.5:9b-q4_K_M",
            "clusters": rows}


class FreshSelectionGateTests(unittest.TestCase):
    def test_rejects_under_40_and_missing_multisource(self):
        small = check_manifest(manifest(count=39), minimum=40, multi_minimum=5)
        self.assertFalse(small["ready_for_model_generation"])
        no_multi = check_manifest(manifest(multis=2), minimum=40, multi_minimum=5)
        self.assertFalse(no_multi["ready_for_model_generation"])
        self.assertIn("fewer_than_required_multi_document_events", no_multi["errors"])

    def test_accepts_structural_threshold_without_claiming_accuracy(self):
        result = check_manifest(manifest(), minimum=40, multi_minimum=5)
        self.assertTrue(result["ready_for_model_generation"])
        self.assertEqual(result["multi_document_events"], 6)
        self.assertFalse(result["human_multi_event_identity_verified"])
        self.assertFalse(result["independent_accuracy_measured"])

    def test_requires_prior_exclusions(self):
        result = check_manifest(manifest(previous=False),
                                minimum=40, multi_minimum=5)
        self.assertIn("missing_previous_exclusion_files", result["errors"])

    def test_duplicate_source_url_blocks_entire_sample(self):
        m = manifest()
        m["clusters"][10]["source_materials"][0]["url"] = (
            m["clusters"][9]["source_materials"][0]["url"])
        self.assertIn("duplicate_or_missing_source_evidence",
                      check_manifest(m, minimum=40, multi_minimum=5)["errors"])

    def test_multiple_sources_not_just_duplicate_urls(self):
        row = sample(1, multi=True)
        self.assertTrue(_multi(row))
        row["source_materials"][1]["source"] = "Site A"
        self.assertFalse(_multi(row))


class EndToEndCliSafetyTests(unittest.TestCase):
    def test_run_command_never_calls_model_below_thresholds(self):
        with patch("scripts.stage4_fresh_qwen_eval.collect",
                   return_value={
                       "ready_for_model_generation": False,
                       "errors": ["fewer_than_required_multi_document_events"],
                   }), patch("scripts.stage4_fresh_qwen_eval.generate") as model:
            exit_code = main(["run", "--exclude", "existing.json",
                              "--out-dir", "ignored_output"])
        self.assertEqual(exit_code, 2)
        model.assert_not_called()

    def test_run_command_generates_only_after_sample_gate(self):
        with patch("scripts.stage4_fresh_qwen_eval.collect",
                   return_value={"ready_for_model_generation": True,
                                 "errors": []}), patch(
                       "scripts.stage4_fresh_qwen_eval.generate",
                       return_value={"complete": True}) as model:
            exit_code = main(["run", "--exclude", "existing.json",
                              "--out-dir", "ignored_output"])
        self.assertEqual(exit_code, 0)
        model.assert_called_once()


class BlindHumanAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.m = manifest()
        self.report = make_report(self.m)
        atomic_json(self.root/"holdout_manifest.json", self.m)
        atomic_json(self.root/"qwen_holdout.json", self.report)
        atomic_json(self.root/"qwen_holdout_gate.json", {
            "minimum": 40, "multi_minimum": 5,
        })
        write_blind_packets(self.m, self.report, self.root)

    def test_resume_after_editor_started_keeps_frozen_sources_and_labels(self):
        manifest_path = self.root/"holdout_manifest.json"
        report_path = self.root/"qwen_holdout.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["input_sha256"] = _sha(manifest_path)
        report["complete_requested_run"] = True
        atomic_json(report_path, report)
        key_path = self.root/"private_qwen_review_key.json"
        key = json.loads(key_path.read_text(encoding="utf-8"))
        key["qwen_report_sha256"] = _sha(report_path)
        atomic_json(key_path, key)
        atomic_json(self.root/"qwen_holdout_gate.json", {
            "version": VERSION, "manifest_sha256": _sha(manifest_path),
            "minimum": 40, "multi_minimum": 5,
        })
        labels = self.root/"qwen_review_labels.csv"
        before = labels.read_bytes()
        with patch("scripts.compare_stage4_qwen.run") as runner:
            result = generate(out_dir=self.root, resume=True,
                              measure_gpu=False)
            runner.assert_not_called()
        self.assertTrue(result["complete"])
        self.assertTrue(result["reused_frozen_review_packet"])
        self.assertEqual(labels.read_bytes(), before)
        self.assertEqual(_sha(report_path), key["qwen_report_sha256"])

    def test_packet_is_blind_and_labels_preserved(self):
        packet = (self.root/"blind_qwen_review.jsonl").read_text(
            encoding="utf-8").splitlines()
        self.assertEqual(len(packet), 45)
        first = json.loads(packet[0])
        self.assertNotIn("status", first)
        self.assertNotIn("validation", first)
        self.assertNotIn("automatic_status", first)
        labels = self.root/"qwen_review_labels.csv"
        labels.write_text("manual sentinel", encoding="utf-8")
        write_blind_packets(self.m, self.report, self.root)
        self.assertEqual(labels.read_text(encoding="utf-8"), "manual sentinel")

    def test_missing_annotations_never_publish_results(self):
        result = score(out_dir=self.root)
        self.assertFalse(result["independent_evaluation_complete"])
        self.assertFalse(result["not_publication_approval"] is False)
        self.assertEqual(result["unlabeled_ids"].__len__(), 45)
        self.assertIsNone(result["draft_precision_95_wilson"])

    def fill(self, *, wrong=(), source_bad=(), event_bad=()):
        labels_path = self.root/"qwen_review_labels.csv"
        with labels_path.open(encoding="utf-8-sig", newline="") as fp:
            rows = list(csv.DictReader(fp, delimiter=";"))
        mapping = json.loads((self.root/"private_qwen_review_key.json").read_text(
            encoding="utf-8"))["mapping"]
        for line in rows:
            rid = line["review_id"]
            for field in ("faithful_facts", "natural_russian", "main_event_covered",
                          "meaningful_news_event", "source_quality"):
                line[field] = "pass"
            line["event_coherence"] = (
                "pass" if mapping[rid]["multi_document"] else "na")
            if rid in wrong:
                line["faithful_facts"] = "fail"
            if rid in source_bad:
                line["source_quality"] = "fail"
            if rid in event_bad:
                line["meaningful_news_event"] = "fail"
        with labels_path.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=rows[0].keys(), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

    def test_manual_threshold_and_false_accept_score(self):
        labels = self.root/"qwen_review_labels.csv"
        with labels.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=";"))
        faulty = rows[0]["review_id"]
        self.fill(wrong={faulty})
        result = score(out_dir=self.root)
        self.assertTrue(result["independent_evaluation_complete"])
        self.assertEqual(result["drafts_human_fail"], 1)
        self.assertIn(faulty, result["false_accepted_review_ids"])
        self.assertEqual(result["manually_verified_industrial_events"], 45)
        self.assertEqual(result["manually_verified_multi_document_events"], 6)

    def test_claim_cannot_finish_with_not_enough_positive_events(self):
        mapping = json.loads((self.root/"private_qwen_review_key.json").read_text(
            encoding="utf-8"))["mapping"]
        # Only 39 manually accepted industrial events: fail the >=40 gate.
        self.fill(event_bad=set(list(mapping)[39:]))
        result = score(out_dir=self.root)
        self.assertFalse(result["independent_evaluation_complete"])
        self.assertLess(result["manually_verified_industrial_events"], 40)

    def test_tampered_model_report_denied(self):
        altered = dict(self.report)
        altered["clusters"] = list(self.report["clusters"])
        altered["clusters"][0] = dict(altered["clusters"][0], status="summary_review")
        atomic_json(self.root/"qwen_holdout.json", altered)
        with self.assertRaisesRegex(ValueError, "differs from review key"):
            score(out_dir=self.root)


if __name__ == "__main__":
    unittest.main()
