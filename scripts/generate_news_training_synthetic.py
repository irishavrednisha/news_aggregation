"""Generate FICTIONAL news SFT candidates via LOCAL Ollama.

Example (offline, may need many inferences):
  python -m scripts.generate_news_training_synthetic --per-tag 10 --negatives 40

Every item is marked origin='synthetic', split='train',
review_status='auto_checked' only after mechanical quote/number checks.
The generator NEVER produces human-verified labels. Use
--allow-auto-checked-synthetic explicitly when building an SFT dataset.
Check a random sample manually; automatic checks cannot prove semantics.
No internet, news APIs, DB or MAX are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from processing.local_news_llm import (
    TAGS, ollama_chat, source_payload, validate_evidence, validate_summary,
)


GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "text": {"type": "string"},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "text", "summary", "evidence"],
    "additionalProperties": False,
}
VARIATIONS = (
    "техническая разработка или промышленный прототип",
    "модернизация технологической линии или производственной площадки",
    "испытание агрегата, инженерной системы либо материала",
    "начало изготовления нового продукта",
    "описание сложной межотраслевой новости с отвлекающим контекстом",
    "неочевидная техническая терминология, но однозначное основное событие",
)
NEGATIVE_THEMES = (
    "корпоративный праздник и рассказ о предприятии",
    "назначение менеджера или дивиденды промышленной компании",
    "расширение логистических перевозок без изготовления техники",
    "спорт, образование или благотворительность в отраслевой организации",
    "строительство жилья, празднование юбилея или розничная акция",
    "упоминание промышленных изделий в юридическом или финансовом событии",
)


def generation_messages(tag: str | None, index: int) -> list[dict]:
    topic = (VARIATIONS if tag else NEGATIVE_THEMES)[index % 6]
    if tag:
        target = (
            "Главная тематика: " + tag + ". "
            "Это должна быть новая конкретная промышленная новость. "
            "Сделай главную тематику однозначной, но не повторяй её "
            "название как ярлык или заголовок категории."
        )
    else:
        target = (
            "Это НЕ промышленная производственная новость. "
            "Можно упоминать заводы и отрасли как второстепенный фон, "
            "но никакой новой техники/производства/разработки в главном "
            "событии нет."
        )
    return [
        {"role": "system", "content": (
            "Ты создаёшь ВЫМЫШЛЕННЫЕ учебные новостные материалы "
            "на русском языке. Все компании и события вымышлены: "
            "избегай настоящих компаний, лиц и точных узнаваемых происшествий. "
            "Нужна разнообразная, правдоподобная стилистика реальной новости. "
            "Создай заголовок, текст на 450–1500 знаков, "
            "отдельную абстрактивную сводку своими словами "
            "на 2–3 предложения. Обычно 250–400 символов с пробелами; "
            "если фактов мало, достаточно 150–250, при большом числе "
            "значимых подробностей допустимо до 550, но НЕ БОЛЬШЕ. "
            "Не растягивай текст ради числа символов. "
            "Добавь 2–4 ДОСЛОВНЫЕ цитаты из созданного текста. "
            "Дословные цитаты должны подтверждать ВСЕ конкретные "
            "утверждения, числа и названия из сводки. "
            "Для непрофильной новости summary и evidence должны быть пустыми. "
            "Верни только JSON по схеме. Никакие факты не должны "
            "выдаваться за действительные события."
        )},
        {"role": "user", "content": (
            f"Номер варианта: {index}. {target}\n"
            f"Сюжетная вариация: {topic}. "
            "Не копируй предыдущие формулировки."
        )},
    ]


def validate_generated(raw: dict, expected_tag: str | None) -> dict:
    title = raw.get("title") or ""
    body = raw.get("text") or ""
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("bad generated title/text")
    if len(title.strip()) < 15 or not 200 <= len(body.strip()) <= 2800:
        raise ValueError("too short or long synthetic example")
    source = source_payload(title, body)
    evidence = validate_evidence(source, raw.get("evidence"))
    summary = raw.get("summary")
    if not isinstance(summary, str):
        raise ValueError("invalid summary")
    if expected_tag:
        if not evidence or not summary.strip():
            raise ValueError("positive news requires source quotes and paraphrase")
        errors = validate_summary(summary, evidence)
        if errors:
            raise ValueError("unverified summary facts: " + str(errors))
    elif evidence or summary.strip():
        raise ValueError("irrelevant training example must have no summary/evidence")
    return {
        "title": title.strip(),
        "text": body.strip(),
        "summary": summary.strip(),
        "evidence": evidence,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Сгенерировать вымышленные кандидаты для SFT")
    p.add_argument("--per-tag", type=int, default=10,
                   help="Число синтетических новостей на одну из 18 категорий")
    p.add_argument("--negatives", type=int, default=40)
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--output", type=Path,
                   default=Path("training/synthetic_candidates.json"))
    args = p.parse_args()
    if not 0 <= args.per_tag <= 500 or not 0 <= args.negatives <= 10000:
        p.error("Некорректное число новостей")
    if not args.per_tag and not args.negatives:
        p.error("Нужно хотя бы одно событие")

    items = []
    if args.output.is_file():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        items = saved.get("examples", [])
        if not isinstance(items, list):
            raise ValueError("Файл продолжения повреждён")
        print(f"Продолжение: найдено {len(items)} предыдущих записей")

    jobs = [(tag, i) for tag in TAGS for i in range(args.per_tag)]
    jobs += [(None, i) for i in range(args.negatives)]
    completed = {item.get("story_id") for item in items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    for tag, i in jobs:
        key = f"synthetic-{tag or 'negative'}-{i:05d}"
        if key in completed:
            continue
        try:
            raw = ollama_chat(generation_messages(tag, i), GEN_SCHEMA, model=args.model)
            checked = validate_generated(raw, tag)
            items.append({
                "story_id": key,
                "origin": "synthetic",
                "split": "train",
                "review_status": "auto_checked",
                "relevant": bool(tag),
                "tags": [tag] if tag else [],
                **checked,
            })
            # Each success checkpoints the user-owned draft file to allow resume.
            args.output.write_text(
                json.dumps({"synthetic": True, "examples": items},
                           ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"{len(items)}: {key}, {checked['title'][:55]}")
        except (ValueError, RuntimeError) as exc:
            failed += 1
            print(f"ПРОПУЩЕНО {key}: {type(exc).__name__}: {exc}")
    print(f"Готово кандидатов: {len(items)}; ошибок генерации: {failed}")
    print("ТОЛЬКО АВТОПРОВЕРКА цитат/чисел! Смысловые ошибки возможны.")
    print("В SFT эти записи попадут лишь по явному --allow-auto-checked-synthetic.")


if __name__ == "__main__":
    main()
