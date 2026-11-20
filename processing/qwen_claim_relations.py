"""Claim-to-citation relation diagnostics for Qwen Stage 4.

This is deliberately *not* an entailment model. A valid literal source quote
never becomes a guarantee of factual support. Every check that cannot
determine the actor, action or event is explicitly marked inconclusive.
"""
from __future__ import annotations

import re
from processing.abstractive_validator import normalize, sentences

_INTEREST = re.compile(
    r"\bвыразил\w*\s+заинтересованност\w*\b", re.I)
_FIRST_WORD = re.compile(r"[А-ЯЁа-яё]{4,}")
_IGNORE_OPENERS = {"ранее", "сегодня", "вчера", "затем", "также", "позднее"}
_NEW_PRODUCT = re.compile(
    r"\b(?:представил\w*|презентовал\w*|анонсировал\w*)\b.{0,90}"
    r"\bнов\w*\b|\bнов\w*\b.{0,65}"
    r"\b(?:представил\w*|презентовал\w*)\b", re.I)
_EVERGREEN_PRODUCT = re.compile(
    r"\b(?:выпуска\w*|производит|предлагает|изготавлива\w*)\b", re.I)
_REAL_PRESENTATION = re.compile(
    r"\b(?:представил\w*|презентовал\w*|анонсировал\w*|"
    r"впервые\s+показал\w*|презентаци\w*|премьер\w*)\b", re.I)
_EXPLICIT_COMPLETION = re.compile(
    r"\b(?:внедрил\w*|завершил\w*\s+внедрени\w*|"
    r"развернул\w*|установил\w*\s+систем\w*)\b", re.I)
_IMPLEMENTED = re.compile(
    r"\b(?:компани\w*|организаци\w*|предприяти\w*|заказчик\w*)\s+"
    r"(?:\S+\s+){0,3}\b(?:внедрил\w*|ввел\w*\s+в\s+эксплуатаци\w*)\b", re.I)


def _actor_head(text: str, verb: re.Pattern) -> str | None:
    match = verb.search(text)
    if not match:
        return None
    prefix = text[:match.start()].strip(" ,:—–-")
    prefix = re.split(r"[.;!?]", prefix)[-1]
    words = _FIRST_WORD.findall(prefix)
    while words and normalize(words[0]) in _IGNORE_OPENERS:
        words.pop(0)
    # Do not guess the grammatical head of a long complex subject.
    if not 1 <= len(words) <= 5:
        return None
    return normalize(words[0])[:6]


def _claim_relationship_issues(claim: str, quotes: list[str],
                               materials: list[dict]) -> list[str]:
    errors: list[str] = []
    evidence = " ".join(quotes)
    if _INTEREST.search(claim):
        actor = _actor_head(claim, _INTEREST)
        source_heads = {
            head for q in quotes if _INTEREST.search(q)
            if (head := _actor_head(q, _INTEREST))
        }
        if actor and source_heads and actor not in source_heads:
            errors.append("actor_mismatch:expressed_interest")

    # A routine catalogue description is not evidence of a *new product
    # announcement*. Check full source too: a separate passage may describe
    # the actual unveiling even when the retriever did not select it.
    if _NEW_PRODUCT.search(claim):
        source = " ".join(str(m.get("title") or "") + ". " +
                          str(m.get("text") or "") for m in materials)
        if (_EVERGREEN_PRODUCT.search(evidence)
                and not _REAL_PRESENTATION.search(source)):
            errors.append("event_not_supported:new_product_presentation")

    # A company-wide implementation is different from "the product can be
    # used" or an illustrative calculation, but unknown = inconclusive.
    if _IMPLEMENTED.search(claim):
        source = " ".join(str(m.get("text") or "") for m in materials)
        if not _EXPLICIT_COMPLETION.search(source) and re.search(
            r"\b(?:можно|поможет|помогает|пример\w*|"
            r"расч[её]тн\w*|предлагает)\b", normalize(source)
        ):
            errors.append("event_not_supported:completed_implementation")
    return errors


def audit_claim_links(claims: list[dict], materials: list[dict]) -> dict:
    assessments = []
    all_issues = []
    originals = {
        str(m.get("id")): str(m.get("title") or "") + "\n" +
        str(m.get("text") or "") for m in materials
    }
    for index, c in enumerate(claims, 1):
        sentence = c.get("claim") if isinstance(c, dict) else None
        ids = c.get("source_ids") if isinstance(c, dict) else None
        quotes = c.get("evidence_quotes") if isinstance(c, dict) else None
        accepted = []
        if isinstance(sentence, str) and isinstance(ids, list) and isinstance(quotes, list):
            for sid, quote in zip(ids, quotes):
                if (isinstance(sid, str) and isinstance(quote, str)
                        and sid in originals and quote in originals[sid]):
                    accepted.append(quote)
        if not sentence or not accepted:
            local = ["evidence_missing_or_invalid"]
        else:
            local = _claim_relationship_issues(sentence, accepted, materials)
        names = [f"claim_{index}_{problem}" for problem in local]
        all_issues.extend(names)
        assessments.append({
            "claim_index": index, "claim": sentence,
            "verified_literal_quote_count": len(accepted),
            "relation_status": ("review_required" if local else
                                "not_refuted_in_supported_patterns"),
            "issues": names,
            "note": ("Exact quote provenance is checked; no semantic entailment "
                     "or grammatical correctness is certified."),
        })
    return {"issues": list(dict.fromkeys(all_issues)),
            "assessments": assessments, "factually_verified": False}
