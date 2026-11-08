"""Read-only: TextRank -> local Gemma 3 4B rewriter (NO training or publishing).

Run on Windows from your project environment:
  python check_extractive_rewrite.py tagging_fresh_25.json --limit 3

Ollama is accessed only on localhost. Results are saved as a new JSON file.
If the rewrite changes names/numbers or becomes much shorter, the original
TextRank extraction is used; a human still needs to verify factual meaning.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from processing.extractive_rewriter import rewrite_extractive, rewrite_extractive_whole
from processing.summary_quality import audit_source_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("news_json", type=Path, help="JSON file containing articles")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument(
        "--rewrite-mode", choices=("whole", "aligned"), default="whole",
        help="whole: единая связная сводка; aligned: каждое предложение отдельно",
    )
    parser.add_argument("--sentence-count", type=int, default=3)
    parser.add_argument("--max-extractive-chars", type=int, default=700)
    parser.add_argument(
        "--output", type=Path,
        default=Path("collector_test_results/gemma3_rewrite_preview.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 500 or not 1 <= args.sentence_count <= 8:
        parser.error("Неверный --limit или --sentence-count")
    if not 200 <= args.max_extractive_chars <= 2000:
        parser.error("Длина TextRank должна быть между 200 и 2000 символами")
    if args.output.exists():
        parser.error("Файл результата уже существует; укажи новое имя через --output")
    payload = json.loads(args.news_json.read_text(encoding="utf-8"))
    articles = payload.get("articles") if isinstance(payload, dict) else payload
    if not isinstance(articles, list):
        parser.error("Ожидался JSON-объект с полем articles")
    # Heavy original summarizer dependencies are loaded only for actual use.
    from processing.summarizer import build_textrank_summary

    results = []
    for i, item in enumerate(articles[:args.limit], 1):
        title = str(item.get("title") or "")
        source = str(item.get("text") or "")
        extractive = build_textrank_summary(
            source, title=title, sentence_count=args.sentence_count,
            max_chars=args.max_extractive_chars,
        ) if source else ""
        rewrite = rewrite_extractive_whole if args.rewrite_mode == "whole" else rewrite_extractive
        result = rewrite(extractive, model=args.model)
        row = {
            "source_quality_warnings": audit_source_alignment(title, extractive),
            "title": title, "url": item.get("url"),
            "source": item.get("source"),
            "tags": item.get("tags"),  # never generate or modify tags here
            **result.to_dict(),
        }
        results.append(row)
        print(
            f"[{i}/{min(len(articles), args.limit)}] {title[:58]}: "
            f"{'переписано' if result.rewritten else 'оставлен TextRank'}; "
            f"предупреждений={len(result.warnings)}",
            flush=True,
        )
    report = {
        "mode": "extractive_then_paraphrase_only",
        "model": args.model,
        "rewrite_mode": args.rewrite_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "publishing_performed": False,
        "all_require_review": True,
        "warning": (
            "Mechanical checks cannot guarantee meaning preservation. "
            "Only a read-only draft, not approved for unattended publishing."
        ),
        "count": len(results),
        "rewritten_count": sum(r["rewritten"] for r in results),
        "fallback_count": sum(not r["rewritten"] for r in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Отчёт: {args.output}")
    print(f"Переписано: {report['rewritten_count']}; TextRank без изменений: {report['fallback_count']}")


if __name__ == "__main__":
    main()
