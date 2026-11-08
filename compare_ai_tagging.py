"""Сравнение двух AI/NLI-моделей на ОДНОМ наборе промышленных новостей.

Запуск:
  python compare_ai_tagging.py ai_filtered_textrank.json ai_mdeberta_textrank.json

Скрипт не имеет доступа к моделям, БД или MAX; работает только с
сохранёнными отчётами. Количество тегов не является метрикой точности:
для выбора модели нужна ручная эталонная разметка.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _index(data: dict) -> dict[str, dict]:
    rows = data.get("results")
    if not isinstance(rows, list):
        raise ValueError("Отчёт должен содержать results: []")
    result = {}
    for row in rows:
        key = row.get("url") or (row.get("source", "") + "::" + row.get("title", ""))
        if not key or key in result:
            raise ValueError("Неуникальная или отсутствующая ссылка новости")
        result[key] = row
    return result


def compare_reports(baseline: dict, challenger: dict) -> dict:
    """Validate identical source texts and compare predictions by URL."""
    for field in ("primary_filter", "as_of", "summary_mode"):
        if baseline.get(field) != challenger.get(field):
            raise ValueError(f"Несовпадение параметра {field}: "
                             "сначала проверьте одинаковую выборку")
    old = _index(baseline)
    new = _index(challenger)
    if set(old) != set(new):
        missing = len(set(old) - set(new))
        extra = len(set(new) - set(old))
        raise ValueError(
            f"Наборы новостей различаются: отсутствует {missing}, добавлено {extra}"
        )

    changes = []
    for key, before in old.items():
        after = new[key]
        if (before.get("title"), before.get("source")) != (
                after.get("title"), after.get("source")):
            raise ValueError(f"Название/источник различаются для {key}")
        # Avoid silently comparing a title+lead run with a TextRank run.
        if before.get("summary_used") != after.get("summary_used"):
            raise ValueError(
                f"Разный текст summary для новости {before.get('title')}; "
                "нельзя сравнивать модели с разными входами"
            )
        a = before.get("tags") or []
        b = after.get("tags") or []
        changes.append({
            "source": before.get("source"),
            "title": before.get("title"),
            "url": key,
            "old_tags": a,
            "new_tags": b,
            "old_event": before.get("event_type"),
            "new_event": after.get("event_type"),
            "old_reason": before.get("reason"),
            "new_reason": after.get("reason"),
            "old_top": (before.get("ranked_tags") or [{}])[0].get("tag"),
            "new_top": (after.get("ranked_tags") or [{}])[0].get("tag"),
            "changed": set(a) != set(b),
        })

    return {
        "baseline_model": baseline.get("model", "unknown"),
        "challenger_model": challenger.get("model", "unknown"),
        "count": len(changes),
        "baseline_tagged": sum(bool(row["old_tags"]) for row in changes),
        "challenger_tagged": sum(bool(row["new_tags"]) for row in changes),
        "same_tag_sets": sum(not row["changed"] for row in changes),
        "changed_tag_sets": sum(row["changed"] for row in changes),
        "only_baseline_tagged": sum(
            bool(row["old_tags"]) and not row["new_tags"] for row in changes
        ),
        "only_challenger_tagged": sum(
            not row["old_tags"] and bool(row["new_tags"]) for row in changes
        ),
        "both_tagged_different": sum(
            bool(row["old_tags"]) and bool(row["new_tags"]) and row["changed"]
            for row in changes
        ),
        "baseline_tag_frequencies": dict(Counter(
            tag for row in changes for tag in row["old_tags"]
        )),
        "challenger_tag_frequencies": dict(Counter(
            tag for row in changes for tag in row["new_tags"]
        )),
        "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнить два отчёта AI-теггера на одинаковых новостях"
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("challenger", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    before = json.loads(args.baseline.read_text(encoding="utf-8"))
    after = json.loads(args.challenger.read_text(encoding="utf-8"))
    comparison = compare_reports(before, after)
    print(f"Базовая модель: {comparison['baseline_model']}")
    print(f"Проверяемая модель: {comparison['challenger_model']}")
    print(f"Новостей: {comparison['count']}")
    print(f"С тегами: {comparison['baseline_tagged']} → "
          f"{comparison['challenger_tagged']}")
    print(f"Совпали наборы тегов: {comparison['same_tag_sets']}; "
          f"изменились: {comparison['changed_tag_sets']}")
    print(f"Только новая назначила теги: {comparison['only_challenger_tagged']}; "
          f"только старая: {comparison['only_baseline_tagged']}")
    print("\nИЗМЕНЕНИЯ:")
    for row in comparison["changes"]:
        if not row["changed"]:
            continue
        print(f"- {row['source']} — {row['title']}")
        print(f"  раньше: {', '.join(row['old_tags']) or 'нет'}"
              f"; теперь: {', '.join(row['new_tags']) or 'нет'}")
    print("\nВНИМАНИЕ: без эталонных правильных тегов это сравнение "
          "не доказывает, какая модель точнее.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Отчёт сохранён: {args.output}")


if __name__ == "__main__":
    main()
