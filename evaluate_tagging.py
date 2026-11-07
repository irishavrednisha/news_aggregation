"""Offline evaluation and calibration of tagging against reviewed labels.

Examples:
  python evaluate_tagging.py collector_test_results/tagging_audit_100.json tagging_reference_100.csv --include-drafts --optimize
  python evaluate_tagging.py collector_test_results/tagging_audit_100.json tagging_reference_100.json --optimize

Only reference_status='verified' counts by default. --include-drafts is
exploratory, NOT a confirmed measurement of tagging quality.
This script does not import PyTorch/SQLAlchemy, touch databases or publish.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_references(path: Path, include_drafts: bool = False) -> dict[str, dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            raw = list(csv.DictReader(stream))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("entries", [])
    result = {}
    for row in raw:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        status = (row.get("review_status") or "").strip().lower()
        if status != "verified" and not (include_drafts and status == "draft"):
            continue
        tags = row.get("expected_tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(";") if t.strip()]
        if not isinstance(tags, list):
            raise ValueError(f"Некорректные expected_tags для {url}")
        if len(set(tags)) != len(tags):
            raise ValueError(f"Повторён тег для {url}")
        result[url] = {**row, "expected_tags": tags}
    return result


def read_predictions(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["url"]: row for row in payload.get("results", [])
        if isinstance(row, dict) and row.get("url")
    }


def labeled_rows(predictions: dict, references: dict) -> list[tuple[dict, dict]]:
    return [
        (predictions[url], reference)
        for url, reference in references.items() if url in predictions
    ]


def metrics(rows: list[tuple[dict, dict]], predictions=None) -> dict:
    """Multi-label micro P/R/F1, exact match and per-tag results."""
    tp = fp = fn = exact = negatives = wrong_negative = 0
    by_tag = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    disagreements = []
    for record, reference in rows:
        gold = set(reference["expected_tags"])
        predicted = set(
            predictions(record) if predictions is not None
            else (record.get("tags") or [])
        )
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        exact += gold == predicted
        if not gold:
            negatives += 1
            wrong_negative += bool(predicted)
        for tag in gold | predicted:
            cell = by_tag[tag]
            cell["support"] += tag in gold
            cell["tp"] += tag in gold and tag in predicted
            cell["fp"] += tag in predicted and tag not in gold
            cell["fn"] += tag in gold and tag not in predicted
        if gold != predicted:
            disagreements.append({
                "source": reference.get("source"),
                "title": reference.get("title"),
                "url": reference.get("url"),
                "expected": sorted(gold),
                "predicted": sorted(predicted),
            })
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    details = {}
    for tag, cell in sorted(by_tag.items()):
        pt = cell["tp"] / (cell["tp"] + cell["fp"]) if cell["tp"] + cell["fp"] else 0.0
        rt = cell["tp"] / (cell["tp"] + cell["fn"]) if cell["tp"] + cell["fn"] else 0.0
        details[tag] = {
            **cell, "precision": round(pt, 4), "recall": round(rt, 4),
            "f1": round(2 * pt * rt / (pt + rt), 4) if pt + rt else 0.0,
        }
    supported = [v["f1"] for v in details.values() if v["support"]]
    return {
        "articles": len(rows),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "micro_f1": round(f1, 4),
        "macro_f1_supported_tags": round(sum(supported) / len(supported), 4) if supported else 0.0,
        "exact_match": round(exact / len(rows), 4) if rows else 0.0,
        "untagged_gold_articles": negatives,
        "false_tag_rate_on_untagged": round(wrong_negative / negatives, 4) if negatives else None,
        "per_tag": details,
        "errors": disagreements,
    }


def threshold_tags(record: dict, weight: float, minimum: float,
                   gap: float, max_tags: int) -> list[str]:
    ranked = record.get("ranked") or []
    scored = []
    for entry in ranked:
        if not all(
            field in entry
            for field in ("focus_similarity", "body_similarity",
                          "keyword_boost", "keyword_penalty")
        ):
            raise ValueError("В отчёте отсутствуют компоненты оценок SBERT")
        score = (
            weight * float(entry["focus_similarity"])
            + (1.0 - weight) * float(entry["body_similarity"])
            + float(entry["keyword_boost"])
            - float(entry["keyword_penalty"])
        )
        scored.append((entry["tag"], score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    if not scored:
        return []
    best = scored[0][1]
    return [
        tag for tag, score in scored
        if score >= minimum and (best - score) <= gap + 1e-9
    ][:max_tags]


def bucket(url: str) -> str:
    # Stable split by original URL, not the order of rows.
    number = int(hashlib.sha256(url.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if number < 70 else "holdout"


def optimize(rows: list[tuple[dict, dict]], max_tags: int = 2) -> dict:
    if not rows or not all(
        row.get("ranked") and all(
            field in row["ranked"][0] for field in
            ("focus_similarity", "body_similarity", "keyword_boost", "keyword_penalty")
        ) for row, _ in rows
    ):
        return {"status": "incompatible_audit"}
    train = [(r, g) for r, g in rows if bucket(g["url"]) == "train"]
    holdout = [(r, g) for r, g in rows if bucket(g["url"]) == "holdout"]
    if len(train) < 20 or len(holdout) < 8:
        return {"status": "not_enough_labeled_articles",
                "train": len(train), "holdout": len(holdout)}
    candidates = []
    for w in (0.55, 0.65, 0.78, 0.85, 0.92):
        for minimum in (0.25, 0.275, 0.30, 0.325, 0.35, 0.375,
                        0.40, 0.425, 0.45, 0.475, 0.50):
            for gap in (0.025, 0.05, 0.075, 0.10):
                predictor = lambda row, w=w, minimum=minimum, gap=gap: threshold_tags(
                    row, w, minimum, gap, max_tags
                )
                outcome = metrics(train, predictor)
                candidates.append((
                    outcome["micro_f1"],
                    outcome["precision"],
                    -abs(w - 0.78),
                    -abs(minimum - 0.35),
                    -abs(gap - 0.05),
                    w, minimum, gap,
                ))
    best = max(candidates)
    w, minimum, gap = best[-3:]
    selected = lambda row: threshold_tags(row, w, minimum, gap, max_tags)
    return {
        "status": "exploratory_grid_search",
        "warning": "Weights/thresholds chosen on train; holdout only for validation. Review draft labels before production use.",
        "parameters": {"focus_weight": w, "min_score": minimum,
                       "max_gap_from_best": gap, "max_tags": max_tags},
        "train": {"n": len(train), **{k:v for k,v in metrics(train,selected).items() if k not in ("per_tag","errors")}},
        "holdout": {"n": len(holdout), **{k:v for k,v in metrics(holdout,selected).items() if k not in ("per_tag","errors")}},
        "baseline_holdout": {k:v for k,v in metrics(holdout).items() if k not in ("per_tag","errors")},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Офлайн-оценка качества 18-тегового классификатора")
    p.add_argument("audit", type=Path, help="JSON из check_focused_tagging.py")
    p.add_argument("reference", type=Path, help="JSON/CSV с expected_tags")
    p.add_argument("--include-drafts", action="store_true",
                   help="ПРЕДВАРИТЕЛЬНЫЕ метрики по непроверенным меткам")
    p.add_argument("--optimize", action="store_true",
                   help="Перебор весов/порогов на train с проверкой на holdout")
    p.add_argument("--max-tags", type=int, default=2)
    p.add_argument("--baseline", type=Path, help="Старый NLI/SBERT отчёт для сравнения по общим URL")
    p.add_argument("--output", type=Path,
                   default=Path("collector_test_results/tagging_evaluation.json"))
    args = p.parse_args()
    if args.max_tags < 1:
        p.error("--max-tags должен быть >= 1")

    references = read_references(args.reference, args.include_drafts)
    preds = read_predictions(args.audit)
    rows = labeled_rows(preds, references)
    if not rows:
        raise SystemExit(
            "Нет проверенных совпадающих меток. Подтвердите review_status=verified "
            "в CSV/JSON или используйте --include-drafts для предварительной оценки."
        )

    report = {
        "label_status": "DRAFT - NOT VERIFIED" if args.include_drafts else "verified",
        "audit": str(args.audit),
        "reference": str(args.reference),
        "reference_count": len(references),
        "matched_count": len(rows),
        "unmatched_count": len(references) - len(rows),
        "default": metrics(rows),
    }
    if args.optimize:
        report["optimization"] = optimize(rows, args.max_tags)
    if args.baseline:
        old = read_predictions(args.baseline)
        common = {url: ref for url, ref in references.items()
                  if url in preds and url in old}
        report["comparison"] = {
            "same_urls": len(common),
            "focused_sbert": metrics(labeled_rows(preds, common)),
            "previous_model": metrics(labeled_rows(old, common)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    m = report["default"]
    print(f"Статус разметки: {report['label_status']}")
    print(f"Проверено: {m['articles']} / {len(references)}; TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    print(f"Micro Precision={m['precision']:.3f} Recall={m['recall']:.3f} F1={m['micro_f1']:.3f}")
    print(f"Exact match={m['exact_match']:.3f}; лишние теги на негативных={m['false_tag_rate_on_untagged']}")
    if args.optimize:
        print("Оптимизация:", report["optimization"].get("parameters",
              report["optimization"].get("status")))
    if args.baseline:
        c = report["comparison"]
        print(f"На тех же {c['same_urls']} статьях: новый F1={c['focused_sbert']['micro_f1']:.3f}, "
              f"старый F1={c['previous_model']['micro_f1']:.3f}")
    print(f"JSON-отчёт: {args.output}")


if __name__ == "__main__":
    main()
