"""Read-only audit of grounded industry tags on existing collector JSON.

python -m scripts.check_grounded_tagging collector_test_results/news_fixed_20261109_031438.json --limit 230

Uses exactly the same title + weighted-ready-summary classifier as production.
The raw article body is never read by the classifier.
This is an isolated tagging diagnostic, not an end-to-end test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from processing.grounded_tagger import select_tags


def load_articles(payload: dict) -> list[dict]:
    if isinstance(payload.get("articles"), list):
        rows = payload["articles"]
    elif isinstance(payload.get("results"), list):
        rows = payload["results"]
    else:
        raise ValueError("Нужен JSON с массивом articles или results")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Все статьи должны быть объектами JSON")
    return rows


def audit(payload: dict, *, limit: int, max_tags: int = 2) -> dict:
    if limit < 1 or max_tags < 1:
        raise ValueError("limit и max_tags должны быть положительными")
    rows = load_articles(payload)[:limit]
    results = []
    seen_urls = set()
    for index, article in enumerate(rows, 1):
        url = str(article.get("url") or "").strip()
        title = str(article.get("title") or "").strip()
        if url and url in seen_urls:
            raise ValueError("Повтор URL в выборке: " + url)
        if url:
            seen_urls.add(url)
        # Missing summary means *headline-only* audit, not a substitute
        # summary constructed from the article.
        summary = str(
            article.get("summary") or
            article.get("summary_text") or
            article.get("summary_used") or ""
        ).strip()
        if title:
            decision = select_tags(
                title, summary=summary, max_tags=max_tags
            )
            outcome, tags, reason = (
                decision.outcome, list(decision.tags), decision.reason
            )
            evidence = list(decision.evidence)
        else:
            outcome, tags, reason, evidence = "review", [], "missing_title", []
        results.append({
            "index": index, "source": article.get("source"),
            "title": title, "url": url,
            "summary_present": bool(summary), "outcome": outcome,
            "tags": tags, "reason": reason, "evidence": evidence,
            "human_tags": "", "human_notes": "",
        })
    return {
        "read_only": True,
        "database_modified": False,
        "publishing_performed": False,
        "mode": "grounded_title_weighted_summary_no_body",
        "note": "Сводка имеет приоритет над заголовком; полный текст не "
                "используется. Если в JSON нет поля summary/summary_text/"
                "summary_used, классификация работает только по заголовку. "
                "Это не проверка качества тегов или публикации в MAX.",
        "counts": {
            "articles": len(results),
            "summaries_present": sum(row["summary_present"] for row in results),
            "tagged": sum(r["outcome"] == "tagged" for r in results),
            "rejected": sum(r["outcome"] == "reject" for r in results),
            "review": sum(r["outcome"] == "review" for r in results),
            "tag_frequency": dict(sorted(Counter(
                tag for row in results for tag in row["tags"]
            ).items())),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит grounded-теггера на реальных новостях")
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=230)
    parser.add_argument("--max-tags", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("collector_test_results/grounded_news_audit.json"),
    )
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("Нельзя перезаписывать исходный файл")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = audit(payload, limit=args.limit, max_tags=args.max_tags)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    if csv_path.resolve() == args.input.resolve():
        parser.error("CSV совпадает с входным файлом")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("index", "source", "title", "url", "outcome",
                        "tags", "reason", "evidence", "summary_present", "human_tags", "human_notes"),
            delimiter=";",
        )
        writer.writeheader()
        for row in result["results"]:
            writer.writerow({
                **row, "tags": ", ".join(row["tags"]),
                "evidence": "; ".join(row["evidence"]),
            })
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    print("JSON:", args.output)
    print("CSV:", csv_path)
    if result["counts"]["summaries_present"] == 0:
        print("ВНИМАНИЕ: в выборке нет сводок; проверка только по заголовкам.")
    print("Проверка по заголовку и готовой сводке, без полного текста, БД и MAX.")


if __name__ == "__main__":
    main()
