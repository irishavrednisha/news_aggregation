"""Read-only editorial flags about TextRank input quality.

These flags are NOT proof of a source error. They only highlight
attributions that deserve manual inspection before using an LLM rewrite.
"""
from __future__ import annotations

import re


QUOTED_RE = re.compile(r"«([^»]+)»")


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold().replace("ё", "е")).strip()


def audit_source_alignment(title: str, extractive: str) -> list[str]:
    """Flag a press-service lead whose name does not match title entities.

    The press service could legitimately be a different source, so this is
    a human-review flag, never an automated factual verdict.
    """
    title = title or ""
    extractive = extractive or ""
    if not title.strip() or not extractive.strip():
        return []
    title_quotes = [_key(x) for x in QUOTED_RE.findall(title)]
    if not title_quotes:
        return []
    prefix = re.match(r"^\s*пресс-служба\s+«([^»]+)»\s*:", extractive, re.I)
    if not prefix:
        return []
    credited = _key(prefix.group(1))
    if credited in title_quotes:
        return []
    if any(entity in _key(extractive) for entity in title_quotes):
        return []
    return ["press_service_not_aligned_with_headline:review_source"]


__all__ = ["audit_source_alignment"]
