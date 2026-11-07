"""Experimental title-role reranker. Never called by production tagging.

Runs against stored read-only check_focused_tagging.py JSON scores.
All heuristics below are provisional; review tags on fresh articles before
considering adoption. No ML, database or publication dependencies.
"""
from __future__ import annotations

import re
from processing.tagging_event_guard import nonindustrial_event_reason

SHIP_CUE = re.compile(r"\b(?:судостроен\w*|кораблестроен\w*|верф\w*|судоремонт\w*|судозавод\w*)\b")
SHIP_VESSEL = r"(?:корабл\w*|судн\w*|эсмин\w*|фрегат\w*|корвет\w*|ледокол\w*|танкер\w*|траулер\w*|газовоз\w*|катер\w*|тральщик\w*|подводн\w*\s+лодк\w*)"
SHIP_EVENT = r"(?:постро\w*|строительств\w*|разработ\w*|спроект\w*|проектир\w*|концепт\w*|производ\w*|выпуск\w*|спуск\w*|заклад\w*|ремонт\w*|модернизац\w*|модернизир\w*)"
SHIP_COMBINED = re.compile(rf"(?:{SHIP_EVENT}.{{0,65}}{SHIP_VESSEL}|{SHIP_VESSEL}.{{0,65}}{SHIP_EVENT})")
# A drone can be called either "беспилотник", "БПЛА" or "беспилотный
# летательный аппарат". Separate its manufacturing/development from mere use.
DRONE_PLATFORM = r"(?:беспилотник\w*|бпла|беспилотн\w*\s+(?:летательн\w*\s+)?аппарат\w*)"
AIRCRAFT_OBJECT = re.compile(
    rf"(?:самолет\w*|вертолет\w*|авиационн\w*|авиастроен\w*|"
    rf"авиадвигател\w*|ил-\d+|су-\d+|пд-\d+|мс-\d+|ssj|sj-100|{DRONE_PLATFORM})"
)
AIRCRAFT_PRODUCTION = re.compile(
    r"(?:разработ\w*|изготов\w*|выпуст\w*|производ\w*|постав\w*|"
    r"создал\w*|создан\w*|созда\w*|спроект\w*|испыт\w*|"
    r"сертифицир\w*|модерниз\w*)"
)
DRONE_DEVELOPMENT = (
    r"(?:производ\w*|изготов\w*|разработ\w*|"
    r"создан\w*|созда\w*|спроект\w*|испыт\w*)"
)
DRONE_MANUFACTURING = re.compile(
    rf"(?:{DRONE_DEVELOPMENT}.{{0,90}}{DRONE_PLATFORM}|"
    rf"{DRONE_PLATFORM}.{{0,90}}{DRONE_DEVELOPMENT})"
)
# A report on *defending against* UAVs isn't a report on aircraft design.
DRONE_COUNTERMEASURES = re.compile(
    rf"(?:борьб\w*\s+с|защит\w*.{{0,25}}\s+от|"
    rf"средств\w*\s+против|перехват\w*.{{0,25}})\s*{DRONE_PLATFORM}"
)

# A round-up of several unrelated events does not have one industry topic.
# This is a tagging-only rule; the collector still saves the full material.
DIGEST_HEADLINE = re.compile(
    r"^(?:дайджест\b|главн(?:ое|ые)\b.{0,90}\bза\s+недел\w*\b|"
    r"обзор\s+новост\w*\b|подборка\s+новост\w*\b|"
    r"итоги\s+недел\w*\s*[:\-])"
)
# Polysemy: an explosive sea/land mine is NOT a mineral deposit.
MILITARY_MINE = re.compile(
    r"\b(?:донн\w*|морск\w*|противокорабельн\w*|"
    r"противопехотн\w*|противотанк\w*)\s+мин\w*\b"
    r"|\bмин\w*\s+для\s+(?:морск\w*|флот\w*|корабл\w*)\b"
)
GEOLOGICAL_EVIDENCE = re.compile(
    r"\b(?:добыч\w*|добыва\w*|рудник\w*|руд\w*|"
    r"шахт\w*|месторожд\w*|геолог\w*|ископаем\w*)\b"
)
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
    if re.search(DRONE_PLATFORM, text):
        # Do not mistake a counter-UAV radar or field observation for
        # a new aircraft. Check the manufacturing/development action.
        if DRONE_COUNTERMEASURES.search(normalized(title)):
            return False
        return bool(DRONE_MANUFACTURING.search(normalized(title)))
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
    if DIGEST_HEADLINE.search(normalized(title)):
        event_reason = "news_roundup"
    if event_reason:
        return {"tags": [], "event_reason": event_reason,
                "rejected_categories": [], "production_process": False}

    military_mine = bool(MILITARY_MINE.search(normalized(title)))
    mineral_evidence = bool(GEOLOGICAL_EVIDENCE.search(normalized(title)))
    ship_ok = shipbuilding_evidence(title, summary)
    air_ok = aviation_production_evidence(title, summary)
    scores = sorted(
        [entry for entry in (record.get("ranked") or [])
         if isinstance(entry, dict) and entry.get("tag") is not None
         and entry.get("final_score") is not None],
        key=lambda item: float(item["final_score"]), reverse=True,
    )
    eligible, rejected = [], []
    original_tags = set(record.get("raw_tags_before_event_guard",
                                   record.get("tags") or []))
    for entry in scores:
        tag = entry["tag"]
        disallowed = (
            (tag == "судостроение" and not ship_ok)
            or (tag == "авиация" and not air_ok)
            or (tag == "горнодобывающая промышленность"
                and military_mine and not mineral_evidence)
        )
        if disallowed:
            # Report only tags the original algorithm actually assigned.
            # Old diagnostics misleadingly listed irrelevant low-score tags
            # as "rejected" for almost every industrial article.
            if tag in original_tags:
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
