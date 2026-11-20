"""Deterministic, exact source-span retrieval for Qwen Stage 4.

Source quotes are NEVER written or translated by a language model. The
retriever is a *candidate evidence index*, not an entailment classifier:
a claim with valid quotes may still be false and always needs editorial review.
"""
from __future__ import annotations

import re

from processing.abstractive_validator import ACRONYM_RE, NUMBER_RE, normalize, sentences

_WORD = re.compile(r"[а-яёa-z]{5,}", re.I)
_NAME = re.compile(r"«([^»]{3,100})»")
STOP = {"которые", "который", "которого", "компания", "компании",
        "предприятия", "предприятие", "нового", "новой", "продукции",
        "россии", "российской", "российские", "также", "будет",
        "после", "сообщила", "сообщает", "сообщили", "производства",
        "выпуск", "планируется", "благодаря", "результате", "данного",
        "позволит", "провели", "отмечается"}


def _words(s: str) -> set[str]:
    return {normalize(word)[:6] for word in _WORD.findall(s)
            if normalize(word) not in STOP}


def _numbers(s: str) -> set[str]:
    return {re.sub(r"\s+", "", match.group()).replace(",", ".").rstrip("%")
            for match in NUMBER_RE.finditer(s)}


def _names(s: str) -> set[str]:
    return {normalize(m.group(1)) for m in _NAME.finditer(s)}


def _chunks(s: str, limit: int = 440) -> list[str]:
    """Return literal slices of text without ever retyping a source quote."""
    if len(s) <= limit:
        return [s] if len(s.strip()) >= 18 else []
    result = []
    start = 0
    while start < len(s):
        end = min(start + limit, len(s))
        if end < len(s):
            split = s.rfind(" ", start + 200, end)
            if split > start:
                end = split
        piece = s[start:end].strip()
        if len(piece) >= 18:
            result.append(piece)
        if end >= len(s):
            break
        start = end + (1 if s[end:end+1] == " " else 0)
    return result


def source_passages(materials: list[dict]) -> list[dict]:
    candidates, used = [], set()
    for source in sorted(materials, key=lambda item: (
            not bool(item.get("is_primary")), str(item.get("id")))):
        sid = str(source.get("id"))
        title, body = str(source.get("title") or ""), str(source.get("text") or "")
        # A title and a body sentence are separate authentic spans.
        strings = [title] + sentences(body)
        # Defensive fallback for scrapes without sentence spaces.
        if len(strings) <= 2 and len(body) > 440:
            strings = [title] + [body[i:i+440] for i in range(0, len(body), 440)]
        for index, string in enumerate(strings):
            for quote in _chunks(string):
                if quote not in title and quote not in body:
                    continue
                key = (sid, quote)
                if key in used:
                    continue
                used.add(key)
                candidates.append({
                    "source_id": sid, "quote": quote,
                    "words": _words(quote), "numbers": _numbers(quote),
                    "names": _names(quote),
                    "acronyms": set(ACRONYM_RE.findall(quote)),
                    "position": index,
                })
    return candidates


def build_claims(summary: str, materials: list[dict], max_quotes: int = 4
                 ) -> list[dict]:
    """One claim per generated sentence; evidence is ONLY literal source data.

    If no relevant evidence exists, make an empty citation. The validator
    treats this as review, never as an approved assertion.
    """
    passages = source_passages(materials)
    result = []
    for sentence in sentences(summary):
        keys, numbers = _words(sentence), _numbers(sentence)
        names, acronyms = _names(sentence), set(ACRONYM_RE.findall(sentence))
        choices, covered_words, covered_numbers = [], set(), set()
        covered_names, covered_acronyms = set(), set()
        excluded = set()
        for _ in range(max_quotes):
            best, best_score = None, 0
            for i, item in enumerate(passages):
                if i in excluded:
                    continue
                new_words = len((keys - covered_words) & item["words"])
                new_numbers = len((numbers - covered_numbers) & item["numbers"])
                new_names = len((names - covered_names) & item["names"])
                new_acronyms = len((acronyms - covered_acronyms) & item["acronyms"])
                # Numbers and names matter more than general word proximity.
                score = (new_words + 8 * new_numbers + 6 * new_names
                         + 4 * new_acronyms)
                if score > best_score:
                    best, best_score = i, score
            if best is None or best_score < 2:
                break
            excluded.add(best)
            chosen = passages[best]
            choices.append(chosen)
            covered_words |= chosen["words"]
            covered_numbers |= chosen["numbers"]
            covered_names |= chosen["names"]
            covered_acronyms |= chosen["acronyms"]
            if (numbers <= covered_numbers and names <= covered_names
                    and acronyms <= covered_acronyms
                    and len(keys & covered_words) >= min(4, len(keys))):
                break
        result.append({
            "claim": sentence,
            "source_ids": [item["source_id"] for item in choices],
            "evidence_quotes": [item["quote"] for item in choices],
        })
    return result
