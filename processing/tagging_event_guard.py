"""Conservative, explainable protection from non-industrial headline events.

Industry names in a headline are context, not evidence that the event is
about manufacturing. This guard identifies a limited set of UNAMBIGUOUS
non-industrial *headline events*. It does not classify the article's topic,
replace Sentence-BERT, or filter collected news.

This module has no ML/DB dependencies and is shared by production tagging,
read-only audits, and offline evaluation.
"""
from __future__ import annotations

import re


_INDUSTRIAL_DELIVERABLE = re.compile(
    r"\b(?:разработал\w*|создал\w*|изготовил\w*|выпустил\w*|"
    r"запустил\w*|ввел\w*|построил\w*|модернизировал\w*|"
    r"начал\w*\s+производств\w*|начал\w*\s+выпуск\w*|"
    r"представил\w*\s+(?:нов\w*\s+)?"
    r"(?:двигател\w*|оборудовани\w*|станок|установк\w*))\b"
)

_EVENT_PATTERNS = (
    (
        "public_or_staff_event",
        re.compile(
            r"\b(?:обучени\w*|вебинар\w*|конкурс\w*|фестивал\w*|"
            r"поздравлени\w*|семинар\w*|выставк\w*|конференци\w*|"
            r"форум\w*|приглаша\w*|обсудят|обсудили)\b"
        ),
    ),
    (
        "corporate_finance_or_retail_promotion",
        re.compile(
            r"\b(?:дивиденд\w*|акционер\w*|акци\w*|котировк\w*|"
            r"рейтин\w*|ипотек\w*|банкротств\w*|страхов\w*\s+взнос\w*)\b"
        ),
    ),
    (
        "transport_operations",
        re.compile(
            r"\b(?:грузоперевозк\w*|погрузк\w*|отгрузк\w*|"
            r"пассажиропоток\w*)\b"
        ),
    ),
    (
        "consumer_or_agriculture",
        re.compile(r"\b(?:скидк\w*|промоакци\w*|распродаж\w*|посев\w*)\b"),
    ),
    (
        "airport_operations",
        re.compile(r"\bприостановил\w*\s+работ\w*\s+аэропорт\w*\b"),
    ),
)


def nonindustrial_event_reason(title: str) -> str | None:
    """Return a diagnostic reason only for clearly non-industrial headlines.

    Prefer precision to coverage: ambiguous headlines pass to the semantic
    model. An industrial result announced at an exhibition/conference is not
    rejected solely due to the event venue.

    The guard deliberately examines the HEADLINE, not keywords mentioned
    anywhere in the article's body.
    """
    text = re.sub(r"\s+", " ", title or "").casefold().replace("ё", "е")
    if not text:
        return None

    announces_deliverable = bool(_INDUSTRIAL_DELIVERABLE.search(text))
    for reason, pattern in _EVENT_PATTERNS:
        if not pattern.search(text):
            continue
        if announces_deliverable and reason in {
            "public_or_staff_event", "transport_operations"
        }:
            continue
        return reason
    return None
