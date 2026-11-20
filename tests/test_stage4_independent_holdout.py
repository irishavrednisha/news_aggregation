"""Independent holdout: all tests offline; no Ollama, PostgreSQL or scraping."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.stage4_independent_holdout import (
    blind_review_files, build_manifest, clean_url, fingerprint,
    new_material, prior_signatures, select_holdout, title_duplicate, run_holdout,
    recover_rostec_article_dates,
)
from scripts.score_stage4_holdout import score, confidence_interval

TODAY = date(2026, 11, 20)
BODY = ("Завод испытал новое оборудование и подготовил производственную линию. "
        "В сообщении описываются технические параметры и изменения процесса. " * 5)


def article(i=1, source="Новые заводы", published="2026-11-19", url=None,
            title=None):
    return {
        "id": f"article:{i}", "title": title or
        f"Промышленный завод {i} запустил новую линию обработки деталей",
        "text": BODY + f" Выпуск деталей на предприятии {i} увеличился.",
        "url": url or f"https://factory.ru/news/{i}", "source": source,
        "text_method": "html", "published_at": published, "is_primary": True,
    }


class HoldoutTests(unittest.TestCase):
    def test_url_keeps_event_identity_query_and_strips_trackers(self):
        url = "http://www.nedradv.ru/nedradv/ru/page_news?utm_a=x&obj=12&fbclid=a"
        self.assertEqual(clean_url(url),
                         "https://nedradv.ru/nedradv/ru/page_news?obj=12")
        self.assertNotEqual(clean_url(url), clean_url(url.replace("obj=12", "obj=13")))

    def test_old_url_title_and_body_must_not_reenter(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"old.json"
            old = article(1)
            path.write_text(json.dumps({"stage": 4, "clusters": [
                {"cluster_id": 2, "source_materials": [old]}]},
                ensure_ascii=False), encoding="utf-8")
            known = prior_signatures([path])
            self.assertEqual(new_material(old, known, TODAY, 30), "seen_url")
            self.assertEqual(new_material(article(4, title=old["title"]),
                                          known, TODAY, 30),
                             "seen_or_near_duplicate_event")
            changed = article(6, title="Пензенский завод полностью обновил станки")
            changed["text"] = old["text"]
            self.assertEqual(new_material(changed, known, TODAY, 30), "seen_text")
            self.assertTrue(known["source_files"][0]["sha256"])

    def test_exclusion_source_is_mandatory(self):
        with self.assertRaises(ValueError):
            prior_signatures([])

    def test_missing_dates_and_rss_previews_excluded(self):
        known = {"urls": set(), "texts": set(), "titles": []}
        self.assertEqual(new_material(article(1, published=""), known, TODAY, 30),
                         "no_verified_date")
        self.assertEqual(new_material(article(1, published=""), known, TODAY, 30,
                                      allow_undated=True), "")
        rss = article(2)
        rss["text_method"] = "rss"
        self.assertEqual(new_material(rss, known, TODAY, 30), "rss_preview_not_full_html")
        self.assertEqual(new_material(article(3, published="2026-09-01"),
                                      known, TODAY, 30), "not_recent")
        self.assertEqual(new_material(article(4, published="2026-11-23"),
                                      known, TODAY, 30), "not_recent")
        self.assertEqual(new_material(article(5), known, TODAY, 30), "")

    def test_title_duplicate_requires_matching_number_when_both_provided(self):
        self.assertTrue(title_duplicate(
            "Завод начал производство новых деталей в Перми",
            "В Перми завод начал производство новых деталей"))
        self.assertFalse(title_duplicate(
            "Завод 10 запустил линию сварки", "Завод 11 запустил линию сварки"))
        self.assertFalse(title_duplicate(
            "В Якутии построили новый горнодобывающий объект",
            "В Москве запустили лабораторию по микроэлектронике"))

    def test_missing_dates_are_counted_only_after_old_exposure(self):
        old = article(1, source="Источник А", published="")
        prior = {"urls": {clean_url(old["url"])},
                 "texts": {fingerprint(old)}, "titles": [old["title"]],
                 "source_files": []}
        undated_new = article(2, source="Источник Б", published="")
        fresh_new = article(3, source="Источник В", published="2026-11-19")
        self.assertEqual(new_material(old, prior, TODAY, 30), "seen_url")
        self.assertEqual(new_material(undated_new, prior, TODAY, 30),
                         "no_verified_date")
        selected, stats = select_holdout(
            [[old], [undated_new], [fresh_new]], prior,
            TODAY, target=5, max_days=30)
        self.assertEqual(len(selected), 1)
        self.assertEqual(stats["rejections"]["seen_url"], 1)
        self.assertEqual(stats["rejections"]["no_verified_date"], 1)
        self.assertEqual(stats["candidate_input_groups"], 3)
        self.assertEqual(stats["candidate_source_counts"]["Источник Б"], 1)
        self.assertEqual(
            stats["rejections_by_source"]["Источник Б"]["no_verified_date"], 1)
        self.assertEqual(
            stats["undated_unseen_examples"][0]["url"], undated_new["url"])
        self.assertNotIn("text", stats["undated_unseen_examples"][0])

    def test_rostec_date_recovery_requires_official_domain_and_same_title(self):
        from types import SimpleNamespace
        undated = article(21, source="Ростех", published="",
                          url="https://rostec.ru/media/news/rostec-new-production/",
                          title="Ростех запустил новую линию производства деталей")
        other = article(22, source="Ростех", published="",
                        url="https://fake-rostec.ru/media/news/unsafe/",
                        title="Ростех начал испытания на заводе в Казани")
        originals = [[undated], [other]]
        calls = []
        def fake_fetch(preview):
            calls.append(preview.url)
            return SimpleNamespace(title=preview.title,
                                   published_at=datetime(2026, 11, 18))
        prior = {"urls": set(), "titles": [], "texts": set(), "source_files": []}
        updated, metrics = recover_rostec_article_dates(
            originals, prior, TODAY, 30, fetch_article=fake_fetch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(metrics["recovered"], 1)
        self.assertEqual(updated[0][0]["published_at"], "2026-11-18T00:00:00")
        self.assertEqual(updated[0][0]["date_provenance"],
                         "official_rostec_article_page")
        self.assertEqual(updated[1][0].get("published_at"), "")
        self.assertEqual(undated["published_at"], "")  # Original is untouched.
        self.assertEqual(other["published_at"], "")

    def test_rostec_date_recovery_rejects_unverified_related_date(self):
        from types import SimpleNamespace
        item = article(23, source="Ростех", published="",
                       url="https://www.rostec.ru/media/news/other/")
        prior = {"urls": set(), "titles": [], "texts": set(), "source_files": []}
        wrong = lambda preview: SimpleNamespace(
            title="Совершенно другой заголовок соседней статьи",
            published_at=datetime(2026, 11, 19))
        updated, metrics = recover_rostec_article_dates(
            [[item]], prior, TODAY, 30, fetch_article=wrong)
        self.assertEqual(metrics["recovered"], 0)
        self.assertEqual(len(metrics["unverified"]), 1)
        self.assertEqual(updated[0][0]["published_at"], "")

        future = lambda preview: SimpleNamespace(
            title=preview.title, published_at=datetime(2026, 11, 23))
        updated, metrics = recover_rostec_article_dates(
            [[item]], prior, TODAY, 30, fetch_article=future)
        self.assertEqual(metrics["recovered"], 0)
        self.assertEqual(updated[0][0]["published_at"], "")

    def test_rostec_date_recovery_never_refetches_seen_articles(self):
        from types import SimpleNamespace
        item = article(24, source="Ростех", published="",
                       url="https://rostec.ru/media/news/seen/")
        prior = {"urls": {clean_url(item["url"])}, "titles": [],
                 "texts": set(), "source_files": []}
        def forbidden(preview):
            raise AssertionError("No fetch for old sources")
        updated, metrics = recover_rostec_article_dates(
            [[item]], prior, TODAY, 30, fetch_article=forbidden)
        self.assertEqual(metrics["attempted"], 0)
        self.assertEqual(updated[0][0]["published_at"], "")

    def test_source_cap_and_multiple_overlaps(self):
        old = article(1)
        prior = {"urls": {clean_url(old["url"])}, "texts": {fingerprint(old)},
                 "titles": [old["title"]], "source_files": []}
        groups = [[article(1)], [article(2)], [article(3)], [article(4)],
                  [article(5, source="Другой источник")]]
        picked, stats = select_holdout(
            groups, prior, TODAY, target=5, max_days=30, max_per_source=2)
        self.assertEqual(len(picked), 3)
        self.assertEqual(stats["rejections"]["seen_url"], 1)
        self.assertEqual(stats["rejections"]["source_cap"], 1)
        self.assertTrue(all(x["cluster_id"] > 900000 for x in picked))

    def test_real_database_multisource_group_is_preserved_not_reinvented(self):
        prior = {"urls": set(), "titles": [], "texts": set(), "source_files": []}
        a, b = article(1, "Сайт A"), article(3, "Сайт B")
        a["id"], b["id"] = 101, 103
        picked, stats = select_holdout([[a, b], [article(2)]],
                                       prior, TODAY, target=10)
        self.assertEqual(len(picked[0]["source_materials"]), 2)
        self.assertEqual(stats["multi_source_groups"], 1)
        self.assertEqual(picked[0]["holdout_provenance"], "database_event_cluster")

    def test_manifest_does_not_claim_completeness_or_accuracy(self):
        known = {"source_files": [{"sha256": "abc", "path": "old.json"}]}
        report = build_manifest([], {"source_counts": {}}, target=100,
                                min_sample=40, mode="both", previous=known,
                                now=TODAY, errors=[], max_days=30)
        self.assertFalse(report["minimum_sample_met"])
        self.assertTrue(report["read_only"])
        self.assertFalse(report["publisher_called"])
        self.assertEqual(report["stage"], 4)

    def test_blind_file_cannot_expose_automatic_verdicts(self):
        with TemporaryDirectory() as tmp:
            rows = [{"cluster_id": 900001, "status": "summary_draft",
                     "gemma_summary": "Завод увеличил выпуск.",
                     "source_materials": [article(3)],
                     "validation": {"issues": [], "passed": True}},
                    {"cluster_id": 900002, "status": "summary_review",
                     "gemma_summary": "", "source_materials": [article(4)],
                     "validation": {"issues": ["unsupported"]}}]
            directory = Path(tmp)
            blind_review_files({"clusters": rows, "evaluation_date": "2026-11-20"},
                               directory)
            packet = (directory/"blind_review.jsonl").read_text(encoding="utf-8")
            for forbidden in ("summary_draft", "summary_review", "unsupported"):
                self.assertNotIn(forbidden, packet)
            self.assertIn("Завод увеличил выпуск.", packet)
            secret = json.loads((directory/"private_review_key.json").read_text("utf-8"))
            self.assertEqual({x["automatic_status"] for x in secret.values()},
                             {"summary_draft", "summary_review"})

    def test_human_labels_can_refute_auto_accepted_draft(self):
        report = {"clusters": [{"cluster_id": 900001, "status": "summary_draft",
                                "gemma_summary": "Выпуск запущен."},
                               {"cluster_id": 900002, "status": "summary_review",
                                "gemma_summary": ""}]}
        key = {"B001": {"cluster_id": 900001, "automatic_status": "summary_draft"},
               "B002": {"cluster_id": 900002, "automatic_status": "summary_review"}}
        labels = [
            {"review_id": "B001", "faithful_facts": "fail", "natural_russian": "pass",
             "main_event_covered": "pass", "meaningful_news_event": "pass",
             "source_quality": "pass"},
            {"review_id": "B002", "faithful_facts": "na", "natural_russian": "na",
             "main_event_covered": "na", "meaningful_news_event": "na",
             "source_quality": "na"},
        ]
        result = score(report=report, mapping=key, label_rows=labels)
        self.assertEqual(result["false_accepted_blind_ids"], ["B001"])
        self.assertEqual(result["drafts_passing_all_human_checks"], 0)
        self.assertTrue(result["all_accepted_drafts_reviewed"])
        self.assertTrue(result["not_publication_approval"])
        labels[0]["faithful_facts"] = ""
        incomplete = score(report=report, mapping=key, label_rows=labels)
        self.assertEqual(incomplete["human_unlabeled_summaries"], 1)
        self.assertFalse(incomplete["all_accepted_drafts_reviewed"])

    def test_malformed_manual_annotations_rejected(self):
        key = {"B001": {"cluster_id": 1, "automatic_status": "summary_review"}}
        result = {"clusters": [{"cluster_id": 1, "status": "summary_review",
                                "gemma_summary": ""}]}
        with self.assertRaises(ValueError):
            score(report=result, mapping=key,
                  label_rows=[{"review_id": "B001", "faithful_facts": "pass"}])
        with self.assertRaises(ValueError):
            score(report=result, mapping=key,
                  label_rows=[{"review_id": "B002", "faithful_facts": "na"}])

    def test_short_holdout_must_not_call_local_gemma(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            old = directory/"old.json"
            old.write_text(json.dumps({"stage": 4, "clusters": [
                {"cluster_id": 1, "source_materials": [article(1)]}
            ]}, ensure_ascii=False), encoding="utf-8")
            diagnostics = directory/"diagnostics.json"
            diagnostics.write_text(json.dumps({"results": [{
                "name": "Fresh", "articles": [article(2, "Fresh",
                    published=datetime.now(timezone.utc).date().isoformat())]
            }]}, ensure_ascii=False), encoding="utf-8")
            with patch("scripts.run_stage4_summaries.run") as generate:
                manifest = run_holdout(
                    previous_reports=[old], out_dir=directory/"out", target=5,
                    minimum=4, max_days=30, source_limit=1, source_cap=5,
                    mode="diagnostics", diagnostics=diagnostics)
            self.assertFalse(manifest["minimum_sample_met"])
            self.assertEqual(manifest["sample_size"], 1)
            generate.assert_not_called()

    def test_resume_cannot_overwrite_independent_editor_labels(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            old = directory/"old.json"
            old.write_text(json.dumps({"stage": 4, "clusters": [
                {"cluster_id": 1, "source_materials": [article(1)]}
            ]}, ensure_ascii=False), encoding="utf-8")
            diagnostics = directory/"diagnostics.json"
            diagnostics.write_text(json.dumps({"results": [{
                "name": "Fresh", "articles": [article(2, "Fresh",
                    published=datetime.now(timezone.utc).date().isoformat())]
            }]}, ensure_ascii=False), encoding="utf-8")
            output = directory/"out"

            def dummy_run(*, input_json, **kwargs):
                frozen = json.loads(input_json.read_text(encoding="utf-8"))
                entries = []
                for row in frozen["clusters"]:
                    item = dict(row)
                    item.update({"status": "summary_draft",
                                 "gemma_summary": "На предприятии запустили новую линию."})
                    entries.append(item)
                return {"stage": 4, "pipeline_version": "stage4-source-date-roles-v5",
                        "model": "gemma3:4b",
                        "evaluation_date": frozen["evaluation_date"],
                        "clusters": entries}

            with patch("scripts.run_stage4_summaries.run", side_effect=dummy_run) as call:
                run_holdout(previous_reports=[old], out_dir=output, target=1,
                            minimum=1, max_days=30, source_limit=1,
                            source_cap=5, mode="diagnostics", diagnostics=diagnostics)
                label_path = output/"review_labels.csv"
                text_before = label_path.read_text(encoding="utf-8-sig")
                label_path.write_text(text_before.replace("B001;;;;;;",
                    "B001;pass;pass;pass;pass;pass;Подтверждено"),
                    encoding="utf-8-sig")
                saved_labels = label_path.read_bytes()
                run_holdout(previous_reports=[old], out_dir=output, target=1,
                            minimum=1, max_days=30, source_limit=1, source_cap=5,
                            mode="diagnostics", diagnostics=diagnostics, resume=True)
                self.assertEqual(label_path.read_bytes(), saved_labels)
                self.assertEqual(call.call_count, 2)
                source_mode = json.loads((output/"gemma_holdout.json").read_text(
                    encoding="utf-8"))["source_input_mode"]
                self.assertEqual(source_mode, "frozen_new_independent_holdout_json")

    def test_wilson_bounds_not_perfect_when_small_sample(self):
        conf = confidence_interval(9, 10)
        self.assertGreater(conf["upper_95_wilson"], 0.90)
        self.assertLess(conf["lower_95_wilson"], 0.90)
        self.assertIsNone(confidence_interval(0, 0))


if __name__ == "__main__":
    unittest.main()
