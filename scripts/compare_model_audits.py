"""Read-only Qwen/SBERT comparison, producing an editor's JSON and CSV queue.

Usage:
python -m scripts.compare_model_audits collector_test_results/sbert_batch40.json \
 collector_test_results/hybrid_batch40.json collector_test_results/qwen_batch40.json \
 --batch collector_test_results/model_batch_v2.json \
 --output-json collector_test_results/model_comparison40.json \
 --output-csv collector_test_results/model_review40.csv

This is exploratory analysis, not independent gold, a trained model, or approval
for publication. Every positive recommendation requires editorial review.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REVIEW_FIELDS = (
    "priority", "reasons", "source", "title", "url",
    "primary_filter_accepted", "primary_filter_reason", "unverified_tls",
    "sbert_tags", "hybrid_tags", "qwen_relevant", "qwen_tags",
    "qwen_error", "qwen_warnings",
    "previous_qwen_relevant", "previous_qwen_tags", "previous_hybrid_tags",
    "human_relevant", "human_tags", "human_review_status", "human_notes",
)


def indexed(payload: dict, label: str) -> dict[str, dict]:
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"{label}: нужен список results")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("url"), str) or not row["url"]:
            raise ValueError(f"{label}: строка без URL")
        if row["url"] in result:
            raise ValueError(f"{label}: повтор URL {row['url']}")
        result[row["url"]] = row
    return result


def validate_tags(tags, label: str) -> list[str]:
    if (not isinstance(tags, list)
            or any(not isinstance(t, str) or not t.strip() for t in tags)):
        raise ValueError(f"{label}: ожидается список тегов")
    if len(tags) != len(set(tags)):
        raise ValueError(f"{label}: повторные теги")
    return tags


def compare(
    sbert: dict, hybrid: dict, qwen: dict, batch: dict | None = None, *,
    previous_qwen: dict | None = None,
    previous_hybrid: dict | None = None,
) -> dict:
    s = indexed(sbert, "SBERT")
    h = indexed(hybrid, "hybrid")
    q = indexed(qwen, "Qwen")
    if not set(s) == set(h) == set(q):
        raise ValueError("Наборы URL моделей не совпадают")
    previous_q = indexed(previous_qwen, "previous Qwen") if previous_qwen is not None else None
    previous_h = indexed(previous_hybrid, "previous hybrid") if previous_hybrid is not None else None
    for label, previous in (("Qwen", previous_q), ("hybrid", previous_h)):
        if previous is not None and set(previous) != set(s):
            raise ValueError(f"Предыдущий отчёт {label} содержит другой набор URL")
    if previous_qwen is not None and previous_qwen.get("tags_only") is not True:
        raise ValueError("Предыдущий Qwen-отчёт тоже должен быть --tags-only")
    if previous_hybrid is not None and previous_hybrid.get("tagger_mode") != "hybrid":
        raise ValueError("Предыдущий отчёт hybrid должен иметь tagger_mode=hybrid")
    if sbert.get("tagger_mode") != "legacy" or hybrid.get("tagger_mode") != "hybrid":
        raise ValueError("Нужны отчёты SBERT legacy и hybrid")
    if qwen.get("tags_only") is not True:
        raise ValueError("Нужен Qwen-отчёт --tags-only")
    if (sbert.get("threshold") != hybrid.get("threshold")
            or sbert.get("max_tags") != hybrid.get("max_tags")):
        raise ValueError("SBERT и hybrid запущены с разными параметрами")

    raw = {}
    if batch is not None:
        articles = batch.get("articles")
        if not isinstance(articles, list):
            raise ValueError("В исходной выборке нет articles")
        for article in articles:
            url = article.get("url") if isinstance(article, dict) else None
            if not url or url in raw:
                raise ValueError("Пустой или повторяющийся URL в выборке")
            raw[url] = article
        if set(raw) != set(s):
            raise ValueError("URL исходной выборки отличаются от отчётов")

    rows = []
    counts = Counter()
    for url in sorted(s):
        sr, hr, qr = s[url], h[url], q[url]
        st = validate_tags(sr.get("tags"), "SBERT")
        ht = validate_tags(hr.get("tags"), "hybrid")
        qt = validate_tags(qr.get("tags"), "Qwen")
        relevant = qr.get("relevant")
        if relevant is not None and type(relevant) is not bool:
            raise ValueError("Qwen relevant должен быть true/false/null")
        if relevant is not True and qt:
            raise ValueError("Qwen назначил теги при отрицательном/пустом решении")
        if len({r.get("title") for r in (sr, hr, qr) if r.get("title")}) > 1:
            raise ValueError("Разные заголовки по URL " + url)
        original = raw.get(url, {})
        reason = original.get("primary_filter_reason")
        filter_pass = original.get("primary_filter_accepted")
        reasons = []
        if relevant is None:
            reasons.append("qwen_abstained_or_failed")
        elif relevant:
            if filter_pass is False:
                reasons.append("qwen_vs_primary_filter")
                if reason == "foreign_without_russian_connection":
                    reasons.append("foreign_scope_conflict")
            if not st:
                reasons.append("qwen_positive_sbert_empty")
            elif set(st) != set(qt):
                reasons.append("different_industry_tags")
            else:
                reasons.append("models_agree_positive_requires_editorial_check")
        elif st:
            reasons.append("sbert_positive_qwen_negative")
        else:
            reasons.append("models_agree_negative_not_gold")
        if st != ht:
            reasons.append("hybrid_changes_sbert")
        if original.get("unverified_tls") is True:
            reasons.append("source_tls_unverified")
        if qr.get("error"):
            reasons.append("qwen_error")

        previous_relevant = None
        previous_tags = None
        previous_hybrid_tags = None
        if previous_q is not None:
            previous_record = previous_q[url]
            previous_relevant = previous_record.get("relevant")
            if previous_relevant is not None and type(previous_relevant) is not bool:
                raise ValueError("Предыдущий Qwen relevant должен быть true/false/null")
            previous_tags = validate_tags(previous_record.get("tags"), "previous Qwen")
            if previous_relevant != relevant:
                reasons.append("qwen_relevance_changed_between_runs")
                counts["qwen_relevance_changed_between_runs"] += 1
            if set(previous_tags) != set(qt):
                reasons.append("qwen_tag_set_changed_between_runs")
                counts["qwen_tag_set_changed_between_runs"] += 1
            if previous_record.get("error") and qr.get("error"):
                reasons.append("qwen_error_repeated")
                counts["qwen_error_repeated"] += 1
        if previous_h is not None:
            previous_hybrid_tags = validate_tags(previous_h[url].get("tags"), "previous hybrid")
            if set(previous_hybrid_tags) != set(ht):
                reasons.append("hybrid_changed_between_runs")
                counts["hybrid_changed_between_runs"] += 1

        high_flags = (
            "qwen_abstained_or_failed", "foreign_scope_conflict",
            "qwen_vs_primary_filter", "qwen_positive_sbert_empty",
            "sbert_positive_qwen_negative",
            "qwen_relevance_changed_between_runs",
            "qwen_error_repeated",
        )
        if any(flag in reasons for flag in high_flags):
            priority = "high"
        elif reasons[0] == "models_agree_negative_not_gold" and len(reasons) == 1:
            priority = "low"
        else:
            priority = "medium"

        counts["qwen_positive" if relevant is True else "qwen_negative"
               if relevant is False else "qwen_abstained"] += 1
        counts["sbert_positive" if st else "sbert_empty"] += 1
        counts["hybrid_positive" if ht else "hybrid_empty"] += 1
        counts[priority + "_priority"] += 1
        if st != ht:
            counts["hybrid_changed"] += 1
        if set(st) != set(qt):
            counts["sbert_qwen_different_tag_sets"] += 1
        if (bool(st) and relevant is False) or (not st and relevant is True):
            counts["binary_model_disagreement"] += 1
        if relevant is None:
            counts["qwen_unresolved"] += 1

        rows.append({
            "priority": priority, "reasons": reasons,
            "source": qr.get("source") or sr.get("source"),
            "title": qr.get("title") or sr.get("title"), "url": url,
            "primary_filter_accepted": filter_pass, "primary_filter_reason": reason,
            "unverified_tls": original.get("unverified_tls"),
            "sbert_tags": st, "hybrid_tags": ht, "qwen_relevant": relevant,
            "qwen_tags": qt, "qwen_error": qr.get("error"),
            "qwen_warnings": qr.get("warnings") or [],
            "previous_qwen_relevant": previous_relevant if previous_q is not None else "",
            "previous_qwen_tags": previous_tags if previous_q is not None else [],
            "previous_hybrid_tags": previous_hybrid_tags if previous_h is not None else [],
            "human_relevant": "", "human_tags": "",
            "human_review_status": "draft", "human_notes": "",
        })
    order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda row: (
        order[row["priority"]], row["source"] or "", row["title"] or "",
    ))
    counts["articles"] = len(rows)
    counts["agreement_negative"] = sum(
        "models_agree_negative_not_gold" in row["reasons"] for row in rows
    )
    counts["agreement_positive_exact_tags"] = sum(
        "models_agree_positive_requires_editorial_check" in row["reasons"]
        for row in rows
    )
    return {
        "read_only": True, "publishing_performed": False,
        "training_performed": False, "gold_labels_provided": False,
        "human_verified_count": 0,
        "purpose": "exploratory_editorial_comparison_not_final_metric",
        "summary": dict(sorted(counts.items())),
        "results": rows,
        "notes": [
            "Согласие моделей не является эталоном и не разрешает публикацию.",
            "Любая незавершённая проверка Qwen — abstention, а не отрицательная новость.",
            "Смена решения Qwen между прогонами ставит новость в высокий приоритет.",
            "Сравнение по URL, не по позиции в файле.",
            "Эти предсказания получены до новых исправлений; нужен повторный прогон.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS, delimiter=";")
        writer.writeheader()
        for row in report["results"]:
            data = {}
            for field in REVIEW_FIELDS:
                value = row.get(field)
                if isinstance(value, list):
                    value = ", ".join(value)
                elif value is None:
                    value = ""
                data[field] = value
            writer.writerow(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen vs SBERT: безопасный аудит")
    parser.add_argument("sbert", type=Path)
    parser.add_argument("hybrid", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--previous-qwen", type=Path,
                        help="Предыдущий Qwen-отчёт с теми же URL")
    parser.add_argument("--previous-hybrid", type=Path,
                        help="Предыдущий hybrid-отчёт с теми же URL")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    inputs = [args.sbert, args.hybrid, args.qwen]
    if args.batch:
        inputs.append(args.batch)
    if args.previous_qwen:
        inputs.append(args.previous_qwen)
    if args.previous_hybrid:
        inputs.append(args.previous_hybrid)
    outputs = [args.output_json, args.output_csv]
    if len({p.resolve() for p in inputs + outputs}) != len(inputs + outputs):
        parser.error("Выходной файл не должен совпадать со входным")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    current = payloads[:3]
    batch = json.loads(args.batch.read_text(encoding="utf-8")) if args.batch else None
    old_qwen = (json.loads(args.previous_qwen.read_text(encoding="utf-8"))
                if args.previous_qwen else None)
    old_hybrid = (json.loads(args.previous_hybrid.read_text(encoding="utf-8"))
                  if args.previous_hybrid else None)
    result = compare(*current, batch, previous_qwen=old_qwen,
                     previous_hybrid=old_hybrid)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(result, args.output_csv)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("Это черновой аудит, не подтверждённая оценка точности.")


if __name__ == "__main__":
    main()
