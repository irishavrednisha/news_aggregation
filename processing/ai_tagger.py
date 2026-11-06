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
    "горнодобывающая промышленность": "добыче полезных ископаемых, обогащении руды, шахтах и горных предприятиях",
    "химическая промышленность": "производстве удобрений, полимеров и химических веществ на химических заводах",
    "оборонная промышленность": "разработке и производстве военной техники, вооружения и оборонных систем",
    "железнодорожное машиностроение": "производстве и модернизации поездов, локомотивов и вагонов",
    "автомобильная промышленность": "производстве и модернизации автомобилей, автобусов и автокомпонентов",
    "заводы": "строительстве, открытии и модернизации промышленных заводов и фабрик",
    "металлургия": "выплавке металлов, металлургических комбинатах и выпуске металлопроката",
    "машиностроение": "производстве станков, промышленных машин, двигателей и механизмов",
    "ТЭК": "производстве электроэнергии, электростанциях, теплоснабжении и энергетических сетях",
    "технологическое производство": "новых производственных линиях, автоматизации выпуска продукции и модернизации производственных процессов",
    "авиация": "разработке и производстве самолётов, вертолётов и авиационных двигателей",
    "судостроение": "строительстве и ремонте кораблей, судов и судового оборудования",
    "электротехническое направление": "производстве трансформаторов, кабелей, электродвигателей и электрооборудования",
    "атомная отрасль": "строительстве АЭС, ядерных реакторах, атомной промышленности и ядерном топливе",
    "нефтегазовая промышленность": "добыче и переработке нефти и газа, месторождениях, трубопроводах и СПГ",
    "технологии": "разработке и внедрении цифровых технологий и искусственного интеллекта в промышленности",
    "инновации": "новых научных разработках и инновационных решениях для промышленности",
    "инжиниринг": "проектировании и испытаниях нового промышленного оборудования и инженерных систем",
}

# Сначала ИИ выбирает ГЛАВНОЕ СОБЫТИЕ. Так мы не называем
# инженерным проектом обычное обучение руководителей верфи.
# Формулировки — дополнения к шаблону «В статье говорится о {}».
EVENT_LABELS: dict[str, str] = {
    "industrial_production": "запуске, расширении или модернизации промышленного производства",
    "industrial_equipment": "разработке, испытаниях или внедрении промышленного оборудования",
    "industrial_research": "новой технологической разработке для промышленности",
    "industrial_mining": "разработке месторождений, добыче ресурсов и промышленной безопасности рудников",
    "industrial_energy": "строительстве, работе или модернизации электростанций и энергосистем",
    "industrial_operations": "производственных результатах, поставках промышленной продукции и регулировании отрасли",
    "industrial_transport": "производстве и поставках транспортной техники и оборудования",
    "staff_training": "обучении сотрудников, управленческих курсах и студенческой практике",
    "competitions": "конкурсах, фестивалях, праздниках и соревнованиях",
    "consumer_promotion": "потребительских скидках, рекламных акциях и розничных продажах",
    "financial_markets": "биржевых котировках, стоимости акций и финансовых инвестициях",
    "corporate_pr": "корпоративных наградах, благотворительности и поздравлениях",
    "nonindustrial_policy": "административных программах поддержки сельского хозяйства без изменений производства",
    "consular_law": "консульских предупреждениях и юридической ответственности граждан",
}
INDUSTRIAL_EVENTS = frozenset(k for k in EVENT_LABELS if k.startswith("industrial_"))
HYPOTHESIS_TEMPLATE = "В статье говорится о {}."


@dataclass(frozen=True)
class AITagResult:
    tags: list[str]
    reason: str
    relevance_score: float
    ranked_tags: list[tuple[str, float]]
    event_type: str | None = None
    ranked_events: list[tuple[str, float]] | None = None


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
    """Zero-shot v2: тип главного события, затем конкурирующие отрасли.

    Аргументы tag_threshold/relevance_threshold относятся к долям
    распределений multi_label=False — они НЕ совместимы со старыми
    независимыми NLI-оценками и НЕ являются вероятностями истинности.
    """
    if not _clean(title, 260) or not _clean(summary, 780):
        return AITagResult([], "missing_title_or_summary", 0.0, [])
    if max_tags < 1:
        raise ValueError("max_tags должен быть >= 1")
    tag_min = _threshold("NEWS_AI_TAG_MIN_SCORE", tag_threshold, 0.12)
    relevance_min = _threshold("NEWS_AI_RELEVANCE_MIN_SCORE", relevance_threshold, 0.60)

    if classifier is None:
        model_name = os.getenv("NEWS_AI_TAG_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        classifier = load_classifier(model_name)

    # На первом этапе главную роль играет событие заголовка; полный
    # лид способен «перетянуть» классификацию с обучения на металлургию
    # только потому, что речь об отраслевом предприятии.
    focus = make_ai_input(title, summary[:320])
    candidates = list(EVENT_LABELS.values())
    event_scores = _score_map(classifier(
        focus, candidate_labels=candidates,
        hypothesis_template=HYPOTHESIS_TEMPLATE, multi_label=False,
    ), candidates)
    events = sorted(
        ((key, event_scores[label]) for key, label in EVENT_LABELS.items()),
        key=lambda item: item[1], reverse=True,
    )
    event_key, _best_event_score = events[0]
    industry_score = sum(event_scores[EVENT_LABELS[key]] for key in INDUSTRIAL_EVENTS)
    # Не отбрасываем отраслевой сюжет из-за произвольного абсолютного
    # порога 0.18 на *один* тип события: оценка распределяется между
    # несколькими близкими промышленными событиями. Именно поэтому
    # история про двигатели ОДК (лидер 0.1781, сумма 0.7432) не доходила
    # до определения отрасли. Итоговая проверка порога тегов остаётся.
    if (event_key not in INDUSTRIAL_EVENTS
            or industry_score < relevance_min):
        return AITagResult(
            [], "no_industrial_relevance", industry_score,
            [], event_key, events,
        )

    # Устаревший multi_label=True давал >0.9 одновременно для
    # металлургии, ТЭК, авиации и инжиниринга в одной новости.
    # Теперь 18 отраслей КОНКУРИРУЮТ: суммы оценок ≈1, и теги
    # выбираются по сравнительной значимости для главного сюжета.
    text = make_ai_input(title, summary, lead)
    labels = list(TAG_LABELS.values())
    output = _score_map(classifier(
        text, candidate_labels=labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE, multi_label=False,
    ), labels)
    ranked = sorted(
        ((tag, output[label]) for tag, label in TAG_LABELS.items()),
        key=lambda item: item[1], reverse=True,
    )
    best = ranked[0][1]
    if best < tag_min:
        return AITagResult(
            [], "no_confident_industrial_tag", industry_score,
            ranked, event_key, events,
        )

    # Второй тег допустим, только если его модель оценила почти
    # наравне с первым. Понижение порога нельзя компенсировать
    # простым наличием слова отрасли во второстепенном абзаце.
    selected = [
        tag for tag, score in ranked
        if score >= tag_min and score >= best * 0.60
    ][:max_tags]
    return AITagResult(
        selected, "tagged", industry_score,
        ranked, event_key, events,
    )
