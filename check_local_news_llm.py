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


def main() -> None:
    p = argparse.ArgumentParser(description="Автономный локальный LLM: теги + новое саммари")
    p.add_argument("news_json", type=Path, help="JSON со списком articles")
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--tags-only", action="store_true",
        help="Только релевантность и теги с цитатами-основаниями; сводку не генерировать",
    )
    p.add_argument("--output", type=Path,
                   default=Path("collector_test_results/local_llm_news_audit.json"))
    args = p.parse_args()
    payload = json.loads(args.news_json.read_text(encoding="utf-8"))
    articles = payload.get("articles")
    if not isinstance(articles, list):
        p.error("Нужен объект JSON с массивом articles")
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
        try:
            decision = analyze_news(
                title, body, model=args.model, tags_only=args.tags_only
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
