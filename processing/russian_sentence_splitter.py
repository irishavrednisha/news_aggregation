"""Sentence boundaries for Russian industrial news, keeping name initials intact.

This is only a conservative segmenter, not a syntactic parser. Prefer a
potentially long sentence to manufacturing 1-letter claims from "им. С. А.".
"""
from __future__ import annotations

import re

_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"„])")
_SHORT_FORMS = frozenset((
    "им", "г", "гор", "ул", "пр", "пер", "д", "корп", "стр", "п",
    "пос", "с", "обл", "респ", "акад", "проф", "т", "е", "н",
    "тыс", "млн", "млрд", "руб", "коп", "шт", "кв", "см", "мм",
    "см", "кг", "ст", "рис", "табл", "ред", "см", "др",
))
# Ambiguous single-letter initial (C. A. Зверева); do not split even where
# the initial is followed by a new uppercase word or another initial.
_INITIAL = re.compile(r"(?<!\w)[А-ЯЁA-Z]\.\s*$")


def russian_sentences(text: str) -> list[str]:
    """Split on plausible sentence terminators; preserve exact substrings.

    Keeps titles and initial chains (им. С. А. Зверева), addresses (г. Казань),
    abbreviations (млн. руб.), decimals and acronyms inside their sentences.
    Does not alter text, case, whitespace inside a sentence, or punctuation.
    """
    value = (text or "").strip()
    if not value:
        return []
    result: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(value):
        before = value[start:m.start()]
        if not before.strip():
            continue
        if before.endswith("."):
            if _INITIAL.search(before):
                continue
            words = re.findall(r"[А-ЯЁа-яёA-Za-z]+", before)
            if words and words[-1].casefold().replace("ё", "е") in _SHORT_FORMS:
                continue
            # e.g. "т. д. Анализ..." should still end a sentence after
            # "т. д.", but never manufacture an isolated "д." claim.
            # The conservative default is to keep abbreviation pairs intact.
        part = value[start:m.start()].strip()
        if part:
            result.append(part)
        start = m.end()
    remainder = value[start:].strip()
    if remainder:
        result.append(remainder)
    return result
