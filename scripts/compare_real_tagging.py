"""Compare read-only tagging audits on NEW independently verified real news.

Input gold comes from: python -m scripts.real_news_review finalize ...
Never use old tuning articles, drafts, or synthetic news as final quality.
Unresolved predictions are ABSTENT, not a negative classification.
No PostgreSQL, Ollama, MAX, or retraining. Only reads JSON and writes report.

python -m scripts.compare_real_tagging \
    training/news_sft/real_gold_verified.json \
    --audit sbert=collector_test_results/sbert_new.json \
    --audit qwen=collector_test_results/qwen_new.json \
    --split validation --output collector_test_results/verified_validation.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

from processing.local_news_llm import TAGS

_CANONICAL_TAGS = set(TAGS)


def read_verified_gold(path: Path, split: str = "validation") -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("examples"), list):
        raise ValueError("Ожидался real_gold_verified.json с полем examples")
    result = {}
    for row in payload["examples"]:
        if not isinstance(row, dict):
            raise ValueError("Некорректная строка эталона")
        if row.get("origin") != "real" or row.get("review_status") != "verified":
            raise ValueError("Эталон содержит не подтверждённую реальную новость")
        if row.get("split") not in ("validation", "test"):
            raise ValueError("Каждой новости нужен фиксированный validation/test split")
        url = str(row.get("url") or "")
        if not url or url in result:
            raise ValueError("Пустой или повторяющийся URL эталона")
        relevant = row.get("relevant")
        tags = row.get("tags")
        if type(relevant) is not bool or not isinstance(tags, list):
            raise ValueError("Неверные типы ручной разметки")
        if (any(not isinstance(tag, str) or tag not in _CANONICAL_TAGS for tag in tags)
                or len(tags) != len(set(tags)) or (not relevant and tags)
                or (relevant and not tags)):
            raise ValueError("Несогласованная ручная разметка тегов и релевантности")
        result[url] = row
    if split != "all":
        result = {url: row for url, row in result.items() if row["split"] == split}
    if not result:
        raise ValueError("В указанном split нет независимо подтверждённых новостей")
    return result


def read_audit(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("read_only") is not True:
        raise ValueError("Ожидался read-only JSON-отчёт")
    if payload.get("publishing_performed") is True:
        raise ValueError("Нельзя оценивать отчёт с публикацией")
    if not isinstance(payload.get("results"), list):
        raise ValueError("В отчёте должен быть список results")
    result = {}
    for row in payload["results"]:
        if not isinstance(row, dict):
            raise ValueError("Некорректная строка прогноза")
        url = str(row.get("url") or "")
        if not url or url in result:
            raise ValueError("Пустой или повторяющийся URL прогноза")
        result[url] = row
    return result


def _resolved(row: dict | None) -> bool:
    """An absent/failed prediction is never silently scored as tags=[]."""
    if row is None or row.get("error"):
        return False
    relevant = row.get("relevant", "legacy_without_relevance_field")
    if relevant is None:
        return False
    if relevant != "legacy_without_relevance_field" and type(relevant) is not bool:
        return False
    tags = row.get("tags")
    if not isinstance(tags, list):
        return False
    if any(not isinstance(t, str) or t not in _CANONICAL_TAGS for t in tags):
        return False
    if len(tags) != len(set(tags)):
        return False
    if relevant is True and not tags:
        return False
    if relevant is False and tags:
        return False
    return True


def score(gold: dict[str, dict], audit: dict[str, dict], urls: set[str]) -> dict:
    tp = fp = fn = exact = relevant_correct = relevant_total = 0
    mismatches = []
    for url in sorted(urls):
        expected = set(gold[url]["tags"])
        row = audit[url]
        predicted = set(row["tags"])
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        exact += expected == predicted
        if type(row.get("relevant")) is bool:
            relevant_total += 1
            relevant_correct += (row["relevant"] == gold[url]["relevant"])
        if expected != predicted:
            mismatches.append({
                "url": url, "title": gold[url]["title"],
                "expected": sorted(expected), "predicted": sorted(predicted),
                "focus_kind": row.get("focus_kind"),
            })
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "scored_articles": len(urls),
        "micro_precision": round(p, 4) if urls else None,
        "micro_recall": round(r, 4) if urls else None,
        "micro_f1": round(2 * p * r / (p + r), 4) if (p + r) else (
            0.0 if urls else None
        ),
        "exact_match": round(exact / len(urls), 4) if urls else None,
        "relevance_accuracy": (
            round(relevant_correct / relevant_total, 4) if relevant_total else None
        ),
        "relevance_scored": relevant_total,
        "tp": tp, "fp": fp, "fn": fn,
        "mismatches": mismatches,
    }


def compare(gold: dict[str, dict], audits: dict[str, dict[str, dict]]) -> dict:
    gold_urls = set(gold)
    all_resolved = {
        name: {url for url in gold_urls if _resolved(pred.get(url))}
        for name, pred in audits.items()
    }
    common = set.intersection(*all_resolved.values()) if all_resolved else set()
    results = {}
    for name, prediction in audits.items():
        resolved = all_resolved[name]
        abstentions = Counter()
        for url in gold_urls - resolved:
            row = prediction.get(url)
            if row is None:
                reason = "not_in_audit"
            elif row.get("warnings"):
                reason = str(row["warnings"][0])
            else:
                reason = "unresolved_result"
            abstentions[reason] += 1
        results[name] = {
            "coverage": round(len(resolved) / len(gold_urls), 4),
            "resolved": len(resolved),
            "unresolved": len(gold_urls) - len(resolved),
            "unresolved_reasons": dict(sorted(abstentions.items())),
            "metrics_on_resolved": score(gold, prediction, resolved),
            "metrics_on_shared_resolved": score(gold, prediction, common),
        }
    return {
        "gold_origin": "independently_verified_real_news",
        "verified_gold_articles": len(gold_urls),
        "shared_resolved_articles": len(common),
        "shared_comparison_warning": (
            "Метрики на общих разрешённых статьях сравнимы, но "
            "покрытие нужно учитывать отдельно. Пропуски не равны отказу."
        ),
        "audits": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("--audit", action="append", required=True,
                        help="Имя=путь-к-read-only-отчёту (укажите >=2)")
    parser.add_argument("--split", choices=("validation", "test", "all"),
                        default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.audit) < 2:
        parser.error("Для сравнения нужно хотя бы два --audit")
    audits = {}
    for item in args.audit:
        if "=" not in item:
            parser.error("--audit указывается в формате имя=файл.json")
        name, path = item.split("=", 1)
        if not name or name in audits:
            parser.error("Пустое или повторное имя --audit")
        audits[name] = read_audit(Path(path))
    result = compare(read_verified_gold(args.gold, args.split), audits)
    result["split"] = args.split
    result["read_only"] = True
    result["publishing_performed"] = False
    if args.output.exists():
        parser.error("Файл результата уже существует; укажите новый --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"Сравнение готово: {args.output}")
    for name, row in result["audits"].items():
        score_row = row["metrics_on_shared_resolved"]
        print(f"{name}: coverage={row['coverage']:.1%} "
              f"shared={score_row['scored_articles']}, "
              f"shared_micro_F1={score_row['micro_f1']}")


if __name__ == "__main__":
    main()
