"""Conservative stage-4 source integrity and Russian draft quality checks.

These checks reject clear contradictions / malformed text; they are not a
grammatical or semantic proof and must not approve publication on their own.
"""
from __future__ import annotations

import re

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


def source_integrity_issues(materials: list[dict]) -> list[str]:
    """Flag extreme magnitude contradictions between article headline and lead.

    A strong discrepancy is review-only because headline and body could refer
    to different projects. Do not repair or silently pick one number.
    Only compare TONNAGES, not dates, generic percentages or unrelated units.
    """
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
            # 10x threshold avoids near-equal rounding or kg/tonne revisions.
            # A title about 7.3 thousand tonnes and body about 7288.4
            # thousand tonnes is an unambiguous source consistency risk.
            if ratio > 10 and (value > 0.1 * title_value):
                return ["source_internal_tonnage_magnitude_conflict"]
    return []
