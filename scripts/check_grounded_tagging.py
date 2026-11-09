"""Read-only industrial tagging audit with existing TextRank -> local Gemma.

Run from project root:
  .\.venv\Scripts\python.exe -m scripts.check_grounded_tagging \
    collector_test_results/news_fixed_20261109_031438.json \
    --limit 5 --model gemma3:4b

Summarization:
  1. processing.summarizer.build_textrank_summary selects source sentences.
  2. processing.summary_pipeline.rewrite_summary_draft uses the existing
     Gemma whole-paragraph minimal editor with fact-change checks.
  3. On a rejected edit or Ollama error, preserve the TextRank extract.
Tagging:
  Only the original headline and final summary are passed to select_tags().
  The classifier never sees the raw article body.

No database writes, publication, or automatic approval of summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Callable

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


def summary_from(article: dict) -> str:
    """Prefer the provided finished summary, never treat body as a summary."""
    return str(
        article.get("summary")
        or article.get("summary_text")
        or article.get("summary_used")
        or ""
    ).strip()


def enrich_summaries(
    payload: dict, *, limit: int, model: str = "gemma3:4b",
    backend: str = "gemma", sentence_count: int = 3,
    max_extractive_chars: int = 700,
    summarizer: Callable | None = None,
    rewrite_draft: Callable | None = None,
    audit_alignment: Callable | None = None,
    progress: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Create summaries for articles that lack them. No in-place mutations.

    Dependency injection keeps unit tests free of Ollama, sklearn and a DB.
    Production uses exactly the previously implemented helpers.
    """
    if not isinstance(payload, dict) or limit < 1:
        raise ValueError("Нужен JSON-объект и положительный limit")
    if not 1 <= sentence_count <= 8 or not 200 <= max_extractive_chars <= 2000:
        raise ValueError("Неверные sentence_count или max_extractive_chars")
    if backend not in ("gemma", "textrank"):
        raise ValueError("summary backend должен быть gemma или textrank")
    # Validate even if the row already contains a summary.
    if backend == "gemma" and (
        not model.startswith(("gemma3:", "gemma2:"))
        or "cloud" in model.casefold()
    ):
        raise ValueError("Только локальные gemma3:/gemma2: через Ollama")

    rows = load_articles(payload)
    updated: list[dict] = []
    total = min(len(rows), limit)
    for number, item in enumerate(rows[:limit], 1):
        row = dict(item)
        existing = summary_from(row)
        if existing:
            row["summary"] = existing
            row.setdefault("extractive_summary", "")
            row.setdefault("summary_method", "provided")
            row.setdefault("summary_warnings", [])
            row.setdefault("source_quality_warnings", [])
            row.setdefault("summary_rewritten", False)
            row.setdefault("summary_review_required", True)
        elif not str(row.get("text") or "").strip():
            row.update({
                "summary": "", "extractive_summary": "",
                "summary_method": "missing_source_text",
                "summary_warnings": ["missing_text_for_summary"],
                "source_quality_warnings": [],
                "summary_rewritten": False, "summary_review_required": True,
            })
        else:
            if summarizer is None:
                from processing.summarizer import build_textrank_summary
                summarizer = build_textrank_summary
            if audit_alignment is None:
                from processing.summary_quality import audit_source_alignment
                audit_alignment = audit_source_alignment
            if rewrite_draft is None:
                from processing.summary_pipeline import rewrite_summary_draft
                rewrite_draft = rewrite_summary_draft

            title = str(row.get("title") or "")
            text = str(row["text"])
            # TextRank is an extractive step, not a generative LLM.
            extractive = summarizer(
                text, title=title, sentence_count=sentence_count,
                max_chars=max_extractive_chars,
            )
            extractive = str(extractive or "").strip()
            source_warnings = list(audit_alignment(title, extractive))
            if not extractive:
                row.update({
                    "summary": "", "extractive_summary": "",
                    "summary_method": "empty_extractive",
                    "summary_warnings": ["empty_extractive"],
                    "source_quality_warnings": source_warnings,
                    "summary_rewritten": False, "summary_review_required": True,
                })
            else:
                # Whole mode is the agreed local-Gemma editor. It receives
                # ONLY selected TextRank sentences, never the article text.
                try:
                    draft = rewrite_draft(
                        extractive, backend=backend, model=model,
                        rewrite_mode="whole",
                    )
                    finished = str(draft.text or "").strip()
                    warnings = list(draft.warnings)
                    rewritten = bool(draft.rewritten)
                    if not finished:
                        finished = extractive
                        rewritten = False
                        warnings.append("empty_draft_fallback")
                    if backend == "textrank":
                        method = "textrank_only"
                    elif rewritten:
                        method = "textrank_gemma_whole"
                    elif warnings:
                        method = "textrank_fallback"
                    else:
                        method = "textrank_unchanged"
                except Exception as exc:
                    # Do not let a failed Gemma request silently turn into
                    # a misleading "AI-generated" summary.
                    finished, rewritten = extractive, False
                    warnings = [
                        "rewrite_error:" + type(exc).__name__ + ":"
                        + str(exc)[:180]
                    ]
                    method = "textrank_fallback"

                row.update({
                    "summary": finished,
                    "extractive_summary": extractive,
                    "summary_method": method,
                    "summary_warnings": warnings,
                    "source_quality_warnings": source_warnings,
                    "summary_rewritten": rewritten,
                    # Any generated draft must be manually checked, even
                    # when mechanical checks accept Gemma's edit.
                    "summary_review_required": True,
                })
        updated.append(row)
        if progress is not None:
            progress(number, total, row)

    copy = dict(payload)
    # Auditing a prior report is possible, but cannot create a summary
    # unless its rows contain the original source text.
    if "articles" in payload:
        copy["articles"] = updated
    else:
        copy["results"] = updated
    return copy


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
        summary = summary_from(article)
        # Tagger input is deliberately only headline + ready summary.
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
            "summary": summary,
            "extractive_summary": str(
                article.get("extractive_summary") or ""
            ),
            "summary_method": article.get("summary_method")
            or ("provided" if summary else "not_generated"),
            "summary_warnings": list(article.get("summary_warnings") or []),
            "source_quality_warnings": list(
                article.get("source_quality_warnings") or []
            ),
            "summary_rewritten": bool(
                article.get("summary_rewritten", False)
            ),
            "summary_review_required": bool(
                article.get("summary_review_required", True)
            ),
            "summary_present": bool(summary),
            "outcome": outcome,
            "tags": tags, "reason": reason, "evidence": evidence,
            "human_tags": "", "human_notes": "",
        })
    methods = Counter(row["summary_method"] for row in results)
    return {
        "read_only": True,
        "database_modified": False,
        "publishing_performed": False,
        "summaries_require_human_review": True,
        "mode": "grounded_title_weighted_summary_textrank_gemma_whole",
        "note": (
            "Сводка: TextRank -> локальная Gemma3 whole, после проверок. "
            "При ошибке — исходный TextRank. Для тегов используются "
            "только заголовок и готовая сводка; исходный текст не читается "
            "теггером. Это не accuracy, precision, recall или F1."
        ),
        "counts": {
            "articles": len(results),
            "summaries_present": sum(r["summary_present"] for r in results),
            "summaries_rewritten": sum(r["summary_rewritten"] for r in results),
            "summary_method_counts": dict(sorted(methods.items())),
            "tagged": sum(r["outcome"] == "tagged" for r in results),
            "rejected": sum(r["outcome"] == "reject" for r in results),
            "review": sum(r["outcome"] == "review" for r in results),
            "tag_frequency": dict(sorted(Counter(
                tag for row in results for tag in row["tags"]
            ).items())),
        },
        "results": results,
    }


def save_report(result: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "index", "source", "title", "url", "summary",
                "extractive_summary", "summary_method",
                "summary_warnings", "source_quality_warnings",
                "summary_review_required", "summary_present",
                "outcome", "tags", "reason", "evidence",
                "human_tags", "human_notes",
            ),
            delimiter=";",
        )
        writer.writeheader()
        for row in result["results"]:
            writer.writerow({
                **row,
                "tags": ", ".join(row["tags"]),
                "summary_warnings": "; ".join(row["summary_warnings"]),
                "source_quality_warnings": "; ".join(
                    row["source_quality_warnings"]
                ),
                "evidence": "; ".join(row["evidence"]),
            })
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=230)
    parser.add_argument("--max-tags", type=int, default=2)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument(
        "--summary-backend", choices=("gemma", "textrank"),
        default="gemma",
        help="gemma=TextRank + local Gemma whole; textrank=no model",
    )
    parser.add_argument("--sentence-count", type=int, default=3)
    parser.add_argument("--max-extractive-chars", type=int, default=700)
    parser.add_argument(
        "--no-generate-summaries", action="store_true",
        help="Use only supplied ready summaries; do not run TextRank or Gemma",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "collector_test_results/grounded_news_audit_with_summaries.json"
        ),
    )
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("Нельзя перезаписывать исходный файл")
    if args.output.with_suffix(".csv").resolve() == args.input.resolve():
        parser.error("CSV совпадает с входным файлом")
    if args.limit < 1 or args.max_tags < 1:
        parser.error("limit и max-tags должны быть положительными")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not args.no_generate_summaries:
        def report_progress(current: int, total: int, article: dict) -> None:
            method = article.get("summary_method", "provided")
            print(
                f"[{current}/{total}] {str(article.get('title') or '')[:55]}: "
                f"сводка={method}; {len(article.get('summary') or '')} симв.",
                flush=True,
            )

        payload = enrich_summaries(
            payload, limit=args.limit, model=args.model,
            backend=args.summary_backend,
            sentence_count=args.sentence_count,
            max_extractive_chars=args.max_extractive_chars,
            progress=report_progress,
        )

    result = audit(payload, limit=args.limit, max_tags=args.max_tags)
    csv_path = save_report(result, args.output)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    if result["counts"]["summaries_present"] < result["counts"]["articles"]:
        print("ВНИМАНИЕ: некоторые сводки отсутствуют; для них "
              "теггер получил только заголовок.")
    print("JSON:", args.output)
    print("CSV:", csv_path)
    print("Результат — черновик, без изменения БД или публикации в MAX.")


if __name__ == "__main__":
    main()
