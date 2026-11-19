"""Conservative stage-4 source integrity and Russian draft quality checks.

These checks reject clear contradictions / malformed text; they are not a
grammatical or semantic proof and must not approve publication on their own.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone

TONNAGE_RE = re.compile(
    r"(?<!\w)(\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?)\s*"
    r"(?:(тыс(?:яч\w*)?\.?|миллион\w*|млн\.?)\s*)?"
    r"(тонн\w*|т\b)", re.I)
GARBLED_LANGUAGE = (
    ("broken_agreement_years", re.compile(
        r"\b(?:соглашени\w*|договор\w*|контракт\w*)\s+на\s+годы\b", re.I)),
    ("incorrect_technical_gender", re.compile(
        r"\bсудно\b.{0,45}\bбудет\s+(?:достроен|оснащен|подготовлен)\b", re.I | re.S)),
    ("malformed_noun_agreement", re.compile(
        r"\b(?:основной активом|основным актив|социальный партнерство)\b", re.I)),
    ("nonsensical_collocation", re.compile(
        r"\bотмел\w*\s+значимост\w*\b", re.I)),
    ("missing_sentence_separator", re.compile(
        r'\."(?=[А-ЯЁA-Z])')),
    ("dangling_passive_gerund", re.compile(
        r"\b(?:цех|оборудование|завод|производство)\b.{0,90}"
        r"\b(?:был|была|было)\s+\w+\s*,\s*вложив\b", re.I | re.S)),
    ("misspelled_metro", re.compile(r"\bметрополитнов\b", re.I)),
    ("tax_debt_attributed_to_private_company", re.compile(
        r"\b(?:задолженн\w*|долг\w*)\s+по\s+налогам\s+перед\s+"
        r"(?:компани\w*|ООО|АО|ПАО|«)", re.I)),
)


def quality_issues(summary: str) -> list[str]:
    """Spot high-confidence editorial errors and request another rewrite."""
    return ["language_" + name for name, regex in GARBLED_LANGUAGE
            if regex.search(summary or "")]


def _tons(text: str, max_chars: int | None = None) -> list[tuple[float, str]]:
    if max_chars is not None:
        text = text[:max_chars]
    result = []
    for match in TONNAGE_RE.finditer(text):
        number = float(match.group(1).replace("\u00a0", "").replace(" ", "")
                       .replace(",", "."))
        multiplier = (match.group(2) or "").lower()
        if multiplier.startswith(("тыс", "тысяч")):
            number *= 1000
        elif multiplier.startswith(("млн", "миллион")):
            number *= 1_000_000
        result.append((number, match.group()))
    return result



MONTH_NUMBERS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
EVENT_DATE_RE = re.compile(
    r"\b(?:(?:\d{1,2})\s*(?:[-–—]|по)\s*)?"
    r"(\d{1,2})\s+(" + "|".join(MONTH_NUMBERS)
    + r")(?:\s+(20\d{2})(?:\s*г(?:ода|\.)?)?)?\b", re.I)
UPCOMING_RE = re.compile(
    r"\b(?:пройд\w*|состоится|состоятся|представят|будут\s+представлен\w*|"
    r"приглашаем|осталось\s+(?:всего\s+)?\d+\s+месяц\w*)\b", re.I)
PAST_EVENT_RE = re.compile(r"\b(?:прошл[аои]|состоял\w*|завершил\w*)\b", re.I)


def stale_upcoming_event_issues(materials: list[dict],
                                *, as_of: date | None = None) -> list[str]:
    """Identify obsolete event invitations, not historical news reports."""
    as_of = as_of or datetime.now(timezone.utc).date()
    for material in materials[:3]:
        lead = str(material.get("title") or "") + ". " + str(material.get("text") or "")[:1200]
        for match in EVENT_DATE_RE.finditer(lead[:850]):
            before = lead[max(0, match.start() - 280):match.start()]
            context = before + lead[match.start():match.end() + 95]
            if not UPCOMING_RE.search(context):
                continue
            year = match.group(3)
            if not year:
                # E.g. "в 2026 году пройдет ... 28 – 29 апреля".
                previous = re.search(r"\b(20\d{2})\s+году\b", before, re.I)
                if not previous:
                    continue
                year = previous.group(1)
            try:
                event_day = date(int(year), MONTH_NUMBERS[match.group(2).lower()],
                                 int(match.group(1)))
            except ValueError:
                continue
            if event_day < as_of:
                return ["source_stale_upcoming_event_date"]
    return []


def source_integrity_issues(materials: list[dict],
                            *, as_of: date | None = None) -> list[str]:
    """Flag extreme magnitude contradictions between article headline and lead.

    A strong discrepancy is review-only because headline and body could refer
    to different projects. Do not repair or silently pick one number.
    Only compare TONNAGES, not dates, generic percentages or unrelated units.
    """
    stale = stale_upcoming_event_issues(materials, as_of=as_of)
    if stale:
        return stale
    for material in materials[:3]:
        title = str(material.get("title") or "")
        body = str(material.get("text") or "")
        title_qty = _tons(title)
        if not title_qty:
            continue
        title_value = title_qty[0][0]
        if title_value <= 0:
            continue
        # Source scrapers often repeat the headline at the beginning of text.
        # Compare different expressions found in the first substantive lead.
        body_qty = _tons(body, max_chars=1200)
        for value, expression in body_qty:
            if value <= 0:
                continue
            ratio = max(value, title_value) / min(value, title_value)
            # 100x threshold avoids unrelated figures and near-equal rounding.
            # A title about 7.3 thousand tonnes and body about 7288.4
            # thousand tonnes is an unambiguous source consistency risk.
            if ratio > 100:
                return ["source_internal_tonnage_magnitude_conflict"]
    return []
