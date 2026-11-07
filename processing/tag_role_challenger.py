"""Experimental title-role reranker. Never called by production tagging.

Runs against stored read-only check_focused_tagging.py JSON scores.
All heuristics below are provisional; review tags on fresh articles before
considering adoption. No ML, database or publication dependencies.
"""
from __future__ import annotations

import re
from processing.tagging_event_guard import nonindustrial_event_reason

SHIP_CUE = re.compile(r"\b(?:судостроен\w*|кораблестроен\w*|верф\w*|судоремонт\w*|судозавод\w*)\b")
SHIP_VESSEL = r"(?:корабл\w*|судн\w*|эсмин\w*|фрегат\w*|корвет\w*|ледокол\w*|танкер\w*|траулер\w*|газовоз\w*|тральщик\w*|подводн\w*\s+лодк\w*)"
SHIP_EVENT = r"(?:постро\w*|строительств\w*|разработ\w*|проектир\w*|концепт\w*|производ\w*|выпуск\w*|спуск\w*|заклад\w*|ремонт\w*|модернизац\w*|модернизир\w*)"
SHIP_COMBINED = re.compile(rf"(?:{SHIP_EVENT}.{{0,65}}{SHIP_VESSEL}|{SHIP_VESSEL}.{{0,65}}{SHIP_EVENT})")
AIRCRAFT_OBJECT = re.compile(r"(?:самолет\w*|вертолет\w*|авиационн\w*|авиастроен\w*|авиадвигател\w*|ил-\d+|су-\d+|пд-\d+|мс-\d+|ssj|sj-100|беспилотник\w*|бпла)")
AIRCRAFT_PRODUCTION = re.compile(r"(?:разработ\w*|изготов\w*|выпуст\w*|производ\w*|постав\w*|создал\w*|испыт\w*|сертифицир\w*|модерниз\w*)")
DRONE_MANUFACTURING = re.compile(r"(?:производ\w*|изготов\w*|разработ\w*).{0,28}(?:беспилотник\w*|бпла)")
PROCESS_TITLE_PATTERNS = (
    re.compile(r"(?:модернизац\w*\s+производств\w*|модернизир\w*.{0,80}производств\w*)"),
    re.compile(r"(?:наращива\w*|расширя\w*|запуст\w*|налажива\w*|возобнов\w*|возрожда\w*|откр\w*).{0,55}(?:выпуск\w*|производств\w*|переработк\w*|производственн\w*\s+лини\w*)"),
    re.compile(r"(?:повысил\w*|оптимизировал\w*|улучшил\w*|автоматизир\w*).{0,70}(?:надежност\w*|эффективност\w*|производств\w*|технологическ\w*\s+процесс\w*)"),
    re.compile(r"(?:производств\w*|производственн\w*\s+лини\w*).{0,55}(?:запуст\w*|модернизир\w*|наращива\w*)"),
)


def normalized(*parts: str) -> str:
    return " ".join(str(p or "") for p in parts).casefold().replace("ё", "е")


def shipbuilding_evidence(title: str, summary: str) -> bool:
    text = normalized(title, summary)
    return bool(SHIP_CUE.search(text) or SHIP_COMBINED.search(text))


def aviation_production_evidence(title: str, summary: str) -> bool:
    text = normalized(title, summary)
    if not AIRCRAFT_OBJECT.search(text):
        return False
    if (re.search(r"ракет\w*|космодром\w*", text)
            and not re.search(r"самолет\w*|авиационн\w*|вертолет\w*", text)):
        return False
    if "беспилотник" in text or "бпла" in text:
        return bool(DRONE_MANUFACTURING.search(text))
    return bool(AIRCRAFT_PRODUCTION.search(text))


def production_process_evidence(title: str) -> bool:
    text = normalized(title)
    return any(pattern.search(text) for pattern in PROCESS_TITLE_PATTERNS)


def candidate_tags(record: dict, *, min_score: float = 0.35,
                   gap: float = 0.05, max_tags: int = 2) -> dict:
    """Experimental reranking without recomputing any model embeddings."""
    title = str(record.get("title") or "")
    summary = str(record.get("summary_used") or "")
    event_reason = nonindustrial_event_reason(title)
    if event_reason:
        return {"tags": [], "event_reason": event_reason,
                "rejected_categories": [], "production_process": False}

    ship_ok = shipbuilding_evidence(title, summary)
    air_ok = aviation_production_evidence(title, summary)
    scores = sorted(
        [entry for entry in (record.get("ranked") or [])
         if isinstance(entry, dict) and entry.get("tag") is not None
         and entry.get("final_score") is not None],
        key=lambda item: float(item["final_score"]), reverse=True,
    )
    eligible, rejected = [], []
    for entry in scores:
        tag = entry["tag"]
        if tag == "судостроение" and not ship_ok:
            rejected.append(tag)
            continue
        if tag == "авиация" and not air_ok:
            rejected.append(tag)
            continue
        eligible.append(entry)
    best = float(eligible[0]["final_score"]) if eligible else None
    tags = [
        entry["tag"] for entry in eligible
        if float(entry["final_score"]) >= min_score
        and best - float(entry["final_score"]) <= gap + 1e-9
    ][:max_tags]

    production = production_process_evidence(title)
    process_row = next((entry for entry in scores
                        if entry["tag"] == "технологическое производство"), None)
    if (production and process_row is not None
            and float(process_row["final_score"]) >= 0.26):
        tags = [tag for tag in tags if tag != "технологическое производство"]
        tags = tags[:max_tags-1] + ["технологическое производство"]

    return {"tags": tags, "event_reason": None,
            "rejected_categories": rejected,
            "production_process": production}
