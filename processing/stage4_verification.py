"""Auditable indexed-evidence verifier for locally generated news summaries.

Evidence comes from literal snippets of source text, not from quotations
retyped by the local LLM. The local LLM's supported label is advisory only.
"""
from __future__ import annotations

import re
from processing.abstractive_validator import normalize, sentences, NUMBER_RE

TOKEN = re.compile(r"[а-яёa-z]{5,}", re.I)


def _number_key(value: str) -> str:
    return normalize(value).replace(" ", "").replace("\u00a0", "").rstrip("%")
STOP = {"который", "которые", "также", "будет", "после", "сейчас",
        "этого", "новые", "новых", "стало", "может", "частью", "годов",
        "компания", "сообщили", "сообщила", "сообщает", "предприятия",
        "россии", "российская"}


def evidence_bank(materials: list[dict], facts: list[dict],
                  summary: str = "", max_chars: int = 4900,
                  max_items: int = 34) -> list[dict]:
    """Build numbered, immutable, directly sourced evidence passages."""
    seen, result, used = set(), [], 0
    source_by_id = {str(m.get("id")): m for m in materials}
    def add(raw: str, source_id: str):
        nonlocal used
        raw = (raw or "").strip()
        key = normalize(raw)
        if (len(raw) < 14 or len(result) >= max_items or not key or key in seen
                or used + len(raw) > max_chars
                or key not in normalize(str(source_by_id.get(source_id, {}).get("title") or "")
                                        + "\n" + str(source_by_id.get(source_id, {}).get("text") or ""))):
            return
        seen.add(key)
        used += len(raw)
        result.append({"id": len(result) + 1, "source_id": source_id, "text": raw})

    ordered = sorted(materials, key=lambda m: (not bool(m.get("is_primary")),
                                                  str(m.get("id"))))
    for material in ordered[:5]:
        add(str(material.get("title") or "")[:240], str(material.get("id")))
    for fact in facts:
        sid = str(fact.get("source_id") or "")
        add(str(fact.get("quote") or ""), sid)
    # For numeric claims, retrieve matching *literal* source passages before
    # filling the bank with generic early paragraphs. This avoids spurious
    # "number_not_in_evidence" for numbers later in long source articles.
    for claim in sentences(summary):
        keys = {_number_key(x.group()) for x in NUMBER_RE.finditer(claim)}
        if not keys:
            continue
        tokens = {normalize(t)[:5] for t in TOKEN.findall(claim)}
        matches = []
        for material in ordered[:5]:
            sid = str(material.get("id"))
            for idx, part in enumerate(sentences(str(material.get("text") or ""))):
                if not 20 <= len(part) <= 290:
                    continue
                source_keys = {_number_key(x.group())
                               for x in NUMBER_RE.finditer(part)}
                common = keys & source_keys
                if common:
                    overlap = len(tokens & {normalize(t)[:5]
                                            for t in TOKEN.findall(part)})
                    matches.append((-len(common), -overlap, idx, sid, part))
        for _, _, _, sid, part in sorted(matches)[:2]:
            add(part, sid)
    for material in ordered[:5]:
        sid = str(material.get("id"))
        # Only previously selected context is available to the 4B local model.
        text = str(material.get("text") or "")[:2100]
        for part in sentences(text):
            part = part.strip()
            if 30 <= len(part) <= 290:
                add(part, sid)
            if len(result) >= max_items or used >= max_chars - 200:
                break
    return result


def check_indexed_verdict(answer: object, summary: str, bank: list[dict]) -> list[str]:
    """Validate stable sentence/evidence IDs and anchored numerical claims.

    This verifies *identity* of evidence and numerical anchoring, not logical
    entailment. A human reviewer is still mandatory for every draft.
    """
    parts = sentences(summary)
    if not isinstance(answer, dict) or not isinstance(answer.get("claims"), list):
        return ["semantic_verdict_missing"]
    claims = answer["claims"]
    if len(claims) < len(parts):
        return ["semantic_claim_count_mismatch"]
    by_id = {x["id"]: x for x in bank}
    encountered = set()
    problems: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            problems.append("semantic_claim_bad_structure")
            continue
        ident = claim.get("sentence_id")
        if type(ident) is int and ident > len(parts):
            # Small models sometimes append extra speculative sentence IDs.
            # Never let those extra rows invalidate correctly aligned claims;
            # we still require every actual sentence to be verified below.
            continue
        if type(ident) is not int or ident < 1 or ident in encountered:
            problems.append("semantic_claim_bad_sentence_id")
            continue
        encountered.add(ident)
        verdict = claim.get("verdict")
        if verdict != "supported":
            problems.append(f"semantic_claim_{ident}_" + str(verdict or "unknown"))
        ids = claim.get("evidence_ids")
        if (not isinstance(ids, list) or not ids
                or any(type(x) is not int or x not in by_id for x in ids)):
            problems.append(f"semantic_claim_{ident}_unknown_evidence_id")
            continue
        selected = "\n".join(by_id[x]["text"] for x in dict.fromkeys(ids))
        # If a numerical claim is asserted, at least one cited passage must
        # contain the same value. Explicit unit checks are done elsewhere.
        nums_claim = {normalize(x.group()).replace(" ", "").rstrip("%")
                      for x in NUMBER_RE.finditer(parts[ident - 1])}
        nums_evidence = {normalize(x.group()).replace(" ", "").rstrip("%")
                         for x in NUMBER_RE.finditer(selected)}
        if not nums_claim.issubset(nums_evidence):
            problems.append(f"semantic_claim_{ident}_number_not_in_evidence")
        claim_tokens = {normalize(t)[:5] for t in TOKEN.findall(parts[ident - 1])
                        if normalize(t) not in STOP}
        evidence_tokens = {normalize(t)[:5] for t in TOKEN.findall(selected)}
        if len(claim_tokens & evidence_tokens) < min(2, len(claim_tokens)):
            problems.append(f"semantic_claim_{ident}_weak_evidence_overlap")
    if len(encountered) != len(parts):
        problems.append("semantic_claim_missing_sentence_ids")
    return list(dict.fromkeys(problems))
