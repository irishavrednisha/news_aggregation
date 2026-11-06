"""Локальное AI-тегирование: zero-shot NLI для 18 промышленных тематик.

Новая модель опциональна, старый SBERT-теггер остаётся доступен.
Два независимых этапа:
1. Есть ли в новости промышленный сюжет?
2. Какие из 18 известных тегов действительно подтверждаются текстом?

Никаких платных API; transformers скачивает веса на ПК при первом запуске.
Оценки NLI НЕ являются откалиброванной вероятностью истинности.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
import os
import re
from typing import Callable


DEFAULT_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

# Имена — те же 18, что в processing.tagger.TAGS и таблице clusters.tags.
# Отрасли описаны коротко, без длинных мешков с ключевыми словами.
TAG_LABELS: dict[str, str] = {
    "горнодобывающая промышленность": "промышленная добыча и обогащение полезных ископаемых, рудники и шахты",
    "химическая промышленность": "химическое производство, удобрения, полимеры и промышленные реагенты",
    "оборонная промышленность": "разработка и производство оборонной техники, оружия и военных систем",
    "железнодорожное машиностроение": "производство и модернизация поездов, локомотивов, вагонов и железнодорожной техники",
    "автомобильная промышленность": "производство автомобилей, автобусов, электробусов и автомобильных компонентов",
    "заводы": "строительство и запуск промышленных заводов, фабрик и производственных цехов",
    "металлургия": "металлургическое производство, плавка металлов, металлургические комбинаты и сплавы",
    "машиностроение": "разработка и производство промышленного оборудования, станков, двигателей и машин",
    "ТЭК": "электроэнергетика, генерация энергии, электростанции, теплоснабжение и энергетическая инфраструктура",
    "технологическое производство": "производственные линии, автоматизация, переработка сырья и выпуск промышленной продукции",
    "авиация": "производство самолётов, вертолётов, авиационных двигателей и авиационной техники",
    "судостроение": "строительство, ремонт и производство кораблей, подводных лодок и судового оборудования",
    "электротехническое направление": "промышленное электрооборудование, электродвигатели, трансформаторы и кабельная техника",
    "атомная отрасль": "атомная промышленность, ядерные реакторы, атомные электростанции и ядерное топливо",
    "нефтегазовая промышленность": "добыча и переработка нефти и газа, нефтегазовые месторождения, трубопроводы и СПГ",
    "технологии": "разработка и внедрение промышленных цифровых технологий, искусственного интеллекта и автоматизации",
    "инновации": "новые исследовательские разработки и инновационные технологии для реальной промышленности",
    "инжиниринг": "проектирование, испытания и разработка инженерных решений для промышленных объектов",
}

INDUSTRIAL_LABEL = (
    "реальная промышленная деятельность: производство, оборудование, "
    "добыча ресурсов, технологические разработки, энергетическая "
    "инфраструктура или регулирование отраслей промышленности"
)
NONINDUSTRIAL_LABEL = (
    "непромышленная деятельность: потребительские покупки, бытовые "
    "акции, развлечения, фестивали, поздравления, биржевые котировки "
    "или мероприятия без новых промышленных разработок"
)
HYPOTHESIS_TEMPLATE = "Основная тема новости — {}."


@dataclass(frozen=True)
class AITagResult:
    tags: list[str]
    reason: str
    relevance_score: float
    ranked_tags: list[tuple[str, float]]


def _clean(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit].strip()


def make_ai_input(title: str, summary: str, lead: str = "") -> str:
    """Начало + TextRank-summary без повторения текста пять раз.

    Заголовок и summary идут раньше лида: при ограничении 512 токенов
    это уменьшает потерю главного события из-за усечения конца текста.
    """
    pieces = []
    title = _clean(title, 260)
    summary = _clean(summary, 780)
    lead = _clean(lead, 330)
    if title:
        pieces.append("Заголовок: " + title)
    if summary:
        pieces.append("Содержание: " + summary)
    if lead and lead.casefold() not in summary.casefold():
        pieces.append("Начало статьи: " + lead)
    return "\n".join(pieces)


@lru_cache(maxsize=2)
def load_classifier(model_name: str = DEFAULT_MODEL):
    """Загрузка модели только при явном вызове AI-режима."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Для AI-теггера установи в .venv: "
            'python -m pip install "transformers>=4.40,<5" torch sentencepiece'
        ) from exc
    return pipeline("zero-shot-classification", model=model_name, device=-1)


def _score_map(output: dict, expected: list[str]) -> dict[str, float]:
    labels, scores = output.get("labels"), output.get("scores")
    if not isinstance(labels, list) or not isinstance(scores, list):
        raise ValueError("AI-модель не вернула labels/scores")
    if len(labels) != len(scores) or set(labels) != set(expected):
        raise ValueError("AI-модель вернула неожиданный набор меток")
    result = {}
    for label, value in zip(labels, scores):
        score = float(value)
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("AI-модель вернула некорректный score")
        result[label] = score
    return result


def _threshold(env_name: str, explicit: float | None, default: float) -> float:
    value = float(explicit) if explicit is not None else float(os.getenv(env_name, str(default)))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{env_name} должен быть в диапазоне от 0 до 1")
    return value


def predict_tags(
    title: str,
    summary: str,
    lead: str = "",
    *,
    max_tags: int = 2,
    tag_threshold: float | None = None,
    relevance_threshold: float | None = None,
    classifier: Callable | None = None,
) -> AITagResult:
    """Возвращает только теги, которые AI подтвердил выше порога.

    classifier — подставной объект для офлайн-тестов. Тег, не набравший
    порог, НЕ ставится, даже если он лучший из всех 18.
    """
    text = make_ai_input(title, summary, lead)
    if not _clean(title, 260) or not _clean(summary, 780):
        return AITagResult([], "missing_title_or_summary", 0.0, [])
    if max_tags < 1:
        raise ValueError("max_tags должен быть >= 1")
    tag_min = _threshold("NEWS_AI_TAG_MIN_SCORE", tag_threshold, 0.65)
    relevance_min = _threshold("NEWS_AI_RELEVANCE_MIN_SCORE", relevance_threshold, 0.60)

    if classifier is None:
        model_name = os.getenv("NEWS_AI_TAG_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        classifier = load_classifier(model_name)

    candidates = [INDUSTRIAL_LABEL, NONINDUSTRIAL_LABEL]
    relevance = _score_map(classifier(
        text, candidate_labels=candidates,
        hypothesis_template=HYPOTHESIS_TEMPLATE, multi_label=False,
    ), candidates)
    industry_score = relevance[INDUSTRIAL_LABEL]
    if industry_score < relevance_min or industry_score <= relevance[NONINDUSTRIAL_LABEL]:
        return AITagResult([], "no_industrial_relevance", industry_score, [])

    labels = list(TAG_LABELS.values())
    output = _score_map(classifier(
        text, candidate_labels=labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE, multi_label=True,
    ), labels)
    ranked = sorted(
        ((tag, output[label]) for tag, label in TAG_LABELS.items()),
        key=lambda item: item[1], reverse=True,
    )
    eligible = [item for item in ranked if item[1] >= tag_min]
    if not eligible:
        return AITagResult([], "no_confident_industrial_tag", industry_score, ranked)

    # Защита от «прилипания» несвязанных классов: максимум 2–3 тега
    # с оценкой близкой к лучшему, а не все 18 выше порога.
    best = eligible[0][1]
    selected = [
        tag for tag, score in eligible if best - score <= 0.15
    ][:max_tags]
    return AITagResult(selected, "tagged", industry_score, ranked)
