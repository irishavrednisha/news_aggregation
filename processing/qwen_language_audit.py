"""Independent Russian editorial surface checks (no LLM, no rewriting).

These are high-confidence warnings and review blockers, not a Russian grammar
checker: passing is NOT a human judgment of fluency or stylistic suitability.
"""
from __future__ import annotations

from collections import Counter
import re
from processing.abstractive_validator import normalize, sentences

_INANIMATE_ACTOR = re.compile(
    r"^\s*(?:драгметалл\w*|металл\w*|руд\w*|кабел\w*|"
    r"оборудовани\w*|топлив\w*)"
    r"(?:\s+[а-яёА-ЯЁ-]+){0,4}\s+"
    r"(?:выразил\w*\s+(?:интерес|заинтересованност)\w*|"
    r"заявил\w*|подчеркнул\w*)\b", re.I)
_BAD_AGREEMENT = re.compile(
    r"\b(?:основной\s+активом|основным\s+актив|"
    r"социальный\s+партнерство|два\s+предприятий)\b", re.I)
_REPEATED_PREPOSITION = re.compile(
    r"\b(?:для\s+для|на\s+на|в\s+в|по\s+по)\b", re.I)
_WORD = re.compile(r"[а-яёa-z]+", re.I)


def audit_russian_quality(summary: str) -> dict:
    issues: list[str] = []
    flags: list[str] = []
    parts = sentences(summary)
    if _BAD_AGREEMENT.search(summary):
        issues.append("language_clear_agreement_error")
    if _REPEATED_PREPOSITION.search(summary):
        issues.append("language_duplicated_preposition")
    for sentence in parts:
        if _INANIMATE_ACTOR.search(sentence):
            issues.append("language_inanimate_reporting_actor")
            break
    words_per_sentence = [len(_WORD.findall(s)) for s in parts]
    if words_per_sentence and max(words_per_sentence) > 45:
        flags.append("language_sentence_too_long_review")
    leads = [normalize(" ".join(_WORD.findall(s)[:2]))
             for s in parts if len(_WORD.findall(s)) >= 2]
    if any(count >= 3 for count in Counter(leads).values()):
        flags.append("language_repetitive_sentence_openings")
    return {
        "issues": list(dict.fromkeys(issues)),
        "flags": list(dict.fromkeys(flags)),
        "sentence_word_counts": words_per_sentence,
        "grammar_verified": False,
        "naturalness_verified": False,
        "editorial_language_review_required": True,
    }
