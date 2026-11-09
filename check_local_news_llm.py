"""Read-only local Qwen news-classification and abstractive-summary audit.

Examples:
    python check_local_news_llm.py tagging_fresh_25.json --limit 10
    python check_local_news_llm.py tagging_fresh_25.json --model qwen3.5:9b

No database session, MAX publishing, network/cloud API or model training.
The only HTTP request goes to a running Ollama on 127.0.0.1:11434.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from processing.local_news_llm import analyze_news


def load_gemma_report(path: Path) -> dict[str, dict]:
    """Read the local Gemma report by URL, not by result list position."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict)
            or not isinstance(report.get("results"), list)
            or report.get("all_require_review") is not True
            or report.get("publishing_performed") is not False):
        raise ValueError("Нужен read-only отчёт Gemma с ручной проверкой")
    by_url = {}
    for item in report["results"]:
        if not isinstance(item, dict) or not item.get("url"):
            raise ValueError("В отчёте Gemma отсутствует URL")
        url = str(item["url"])
        if url in by_url:
            raise ValueError("В отчёте Gemma повторяется URL: " + url)
        by_url[url] = item
    return by_url


def eligible_gemma_draft(row: dict | None, title: str) -> tuple[str | None, str]:
    """Reject unsafe or unmatched drafts; never send model_candidate to Qwen."""
    if row is None:
        return None, "missing_gemma_draft"
    if row.get("title") != title:
        return None, "gemma_title_mismatch"
    if (row.get("review_required") is not True
            or row.get("publication_performed") is not False):
        return None, "gemma_review_contract_missing"
    if row.get("editorial_outcome") not in {"edited", "unchanged"}:
        return None, "gemma_edit_rejected"
    if row.get("warnings"):
        return None, "gemma_validation_warnings"
    if row.get("source_quality_warnings"):
        return None, "source_quality_review_needed"
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "gemma_summary_empty"
    return text.strip(), ""


def choose_gemma_focus(
    row: dict | None, title: str, *, fallback_extractive: bool = False,
) -> tuple[str | None, str, str]:
    """Return (focus text, focus kind, skip reason); fail closed.

    A rejected model candidate must never be used as source text. With an
    explicit opt-in, the unchanged TextRank fallback can still focus Qwen,
    provided the fallback is present and the source is not flagged.
    """
    edited, reason = eligible_gemma_draft(row, title)
    if not reason:
        return edited, "gemma_" + row["editorial_outcome"], ""
    if (fallback_extractive
            and reason == "gemma_edit_rejected"
            and row is not None
            and row.get("source_quality_warnings") == []
            and row.get("editorial_outcome") == "rejected"
            and isinstance(row.get("warnings"), list)
            and bool(row["warnings"])
            and isinstance(row.get("extractive"), str)
            and isinstance(row.get("text"), str)
            and row["extractive"].strip()
            and row["text"].strip() == row["extractive"].strip()):
        return row["extractive"].strip(), "textrank_fallback", ""
    return None, "none", reason


def main() -> None:
    p = argparse.ArgumentParser(description="Автономный локальный LLM: теги + новое саммари")
    p.add_argument("news_json", type=Path, help="JSON со списком articles")
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--tags-only", action="store_true",
        help="Только релевантность и теги с цитатами-основаниями; сводку не генерировать",
    )
    p.add_argument(
        "--gemma-report", type=Path,
        help="Эксперимент: тэги по заголовку и проверенной сводке Gemma, "
             "цитаты только из оригинала. Требует --tags-only",
    )
    p.add_argument(
        "--fallback-extractive", action="store_true",
        help="При отклонённой редактуре Gemma использовать неизменённый "
             "TextRank, только если первоисточник не помечен как сомнительный. "
             "Требует --gemma-report и --tags-only",
    )
    p.add_argument("--output", type=Path,
                   default=Path("collector_test_results/local_llm_news_audit.json"))
    args = p.parse_args()
    payload = json.loads(args.news_json.read_text(encoding="utf-8"))
    articles = payload.get("articles")
    if not isinstance(articles, list):
        p.error("Нужен объект JSON с массивом articles")
    if args.gemma_report and not args.tags_only:
        p.error("--gemma-report требует --tags-only")
    if args.fallback_extractive and (not args.gemma_report or not args.tags_only):
        p.error("--fallback-extractive требует --gemma-report и --tags-only")
    gemma_by_url = load_gemma_report(args.gemma_report) if args.gemma_report else None
    if not 1 <= args.limit <= 1000:
        p.error("--limit должен быть от 1 до 1000")
    results = []
    for i, item in enumerate(articles[:args.limit], start=1):
        title, body = item.get("title") or "", item.get("text") or ""
        row = {
            "source": item.get("source"),
            "url": item.get("url"),
            "title": title,
            "article_is_synthetic": bool(item.get("synthetic")),
        }
        focus_summary = None
        if gemma_by_url is not None:
            gemma_row = gemma_by_url.get(str(item.get("url") or ""))
            focus_summary, focus_kind, skip_reason = choose_gemma_focus(
                gemma_row, title, fallback_extractive=args.fallback_extractive,
            )
            row["focus_kind"] = focus_kind
            if focus_kind == "textrank_fallback":
                row["gemma_rejection_warnings"] = gemma_row.get("warnings", [])
            row["gemma_outcome"] = (
                gemma_row.get("editorial_outcome") if gemma_row else None
            )
            row["gemma_source_quality_warnings"] = (
                gemma_row.get("source_quality_warnings", []) if gemma_row else []
            )
            if skip_reason:
                row.update({
                    "relevant": None,
                    "tags": [],
                    "summary": "",
                    "evidence": [],
                    "review_required": True,
                    "warnings": ["skipped:" + skip_reason],
                    "error": "Gemma draft unavailable or unsuitable",
                    "model": args.model,
                })
                print(f"[{i}] Пропуск: {title[:65]} — {skip_reason}")
                results.append(row)
                continue
            row["summary_for_tagging"] = focus_summary
        try:
            decision = analyze_news(
                title, body, model=args.model, tags_only=args.tags_only,
                focus_summary=focus_summary,
            )
            row.update(decision.to_dict())
            print(f"[{i}/{min(len(articles),args.limit)}] {title[:65]} "
                  f"-> {', '.join(decision.tags) if decision.tags else 'без тегов'} "
                  + ("" if args.tags_only else f"; сводка: {decision.summary[:90]}"))
        except Exception as exc:
            row.update({
                "relevant": None,
                "tags": [],
                "summary": "",
                "evidence": [],
                "review_required": True,
                "warnings": ["inference_or_validation_failed"],
                "error": f"{type(exc).__name__}: {exc}",
                "model": args.model,
            })
            print(f"[{i}] Ошибка/отложено на проверку: {type(exc).__name__}: {exc}")
        results.append(row)
    report = {
        "read_only": True,
        "publishing_performed": False,
        "training_performed": False,
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.news_json),
        "count": len(results),
        "tags_only": args.tags_only,
        "gemma_report": str(args.gemma_report) if args.gemma_report else None,
        "tagging_basis": (
            "title_gemma_or_textrank_article" if args.fallback_extractive
            else "title_gemma_article" if args.gemma_report
            else "original_article"
        ),
        "fallback_extractive_enabled": args.fallback_extractive,
        "fallback_extractive_count": sum(
            row.get("focus_kind") == "textrank_fallback" for row in results
        ),
        "skipped_gemma": sum(
            any(w.startswith("skipped:") for w in (row.get("warnings") or []))
            for row in results
        ),
        "review_required": sum(bool(r.get("review_required")) for r in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Отчёт: {args.output}")


if __name__ == "__main__":
    main()
