"""Exact-evidence first pass and batch-only shared-footer detection.

Never mutate source articles. A quotation is selected from an existing
source span, never composed by a language model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re
from processing.abstractive_validator import normalize, sentences

EVENT = re.compile(
    r"\b(?:выпуст\w*|созда\w*|разработ\w*|запуст\w*|постро\w*|заверш\w*|"
    r"постав\w*|произв\w*|добы\w*|заяв\w*|сниз\w*|возобнов\w*|"
    r"предлож\w*|поддерж\w*|предупред\w*|задерж\w*|нараст\w*|"
    r"обсуд\w*|открыл\w*|открыт\w*|представ\w*|выдел\w*|"
    r"модерниз\w*|сократ\w*|вырос\w*|увелич\w*|восстанов\w*|"
    r"получ\w*|приобр\w*|прошл\w*|пройде\w*|состо\w*|"
    r"провел\w*|провед\w*|начал\w*|утверд\w*|принял\w*|"
    r"готовит\w*|готовится|вернул\w*|возвраща\w*|останов\w*|"
    r"заключ\w*|признал\w*|смонтир\w*|отправ\w*|обеспеч\w*|"
    r"подключ\w*|прогнозир\w*|предлага\w*|подтверд\w*|наращива\w*|"
    r"пере[шёе]л\w*|перевыполн\w*|проходит|обязал\w*|награжд\w*|"
    r"удосто\w*|признан\w*|передал\w*|вводят|введен\w*|"
    r"пройдет|проедет|перешли|планир\w*|хочет)\b", re.I)
NON_NEWS = re.compile(r"(?:^|\b)(?:как\s+|обзор\s+|советы\s+|реклама\b|что\s+ломается|почему\s+|топ[-\s]?\d+)", re.I)
NUMBER = re.compile(
    r"(?<!\w)\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?\s*"
    r"(?:%|тыс\.?|млн|млрд|руб\.?|тонн|кг|МВт)(?!\w)", re.I)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
QUOTED = re.compile(r"«[^»]{3,65}»|\"[^\"]{3,65}\"")
WORDS = re.compile(r"[а-яёa-z]{4,}", re.I)
STOP = {"в", "на", "по", "для", "из", "при", "что", "как", "это", "или",
        "новый", "новая", "россии", "российский", "компания", "производства",
        "завод", "от", "и", "с", "об", "год", "уже", "свой"}


def shared_boilerplate(cluster_materials: list[list[dict]], min_documents: int = 2
                       ) -> dict[str, set[str]]:
    """High-frequency same-source long sentences across DISTINCT clusters."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for materials in cluster_materials:
        by_source: dict[str, set[str]] = defaultdict(set)
        for material in materials:
            source = str(material.get("source") or "")
            for sentence in sentences(str(material.get("text") or "")):
                key = normalize(sentence)
                if len(key) >= 45:
                    by_source[source].add(key)
        for source, keys in by_source.items():
            counts[source].update(keys)
    return {
        src: {s for s, n in counter.items()
              if n >= min_documents and (n >= 4 or repeated_count >= 4)}
        for src, counter in counts.items()
        for repeated_count in [sum(n >= min_documents for n in counter.values())]
    }


def clean_working_materials(materials: list[dict],
                            shared: dict[str, set[str]] | None = None
                            ) -> tuple[list[dict], list[dict]]:
    """Never change raw source dicts. Trim only demonstrably repeated suffix."""
    shared = shared or {}
    output, notes = [], []
    for material in materials:
        copy = dict(material)
        original = str(copy.get("text") or "")
        keys = shared.get(str(copy.get("source") or ""), set())
        cursor, cut, boundary = 0, None, ""
        for sentence in sentences(original):
            position = original.find(sentence, cursor)
            if position < 0:
                continue
            cursor = position + len(sentence)
            if position >= 450 and normalize(sentence) in keys:
                cut, boundary = position, sentence[:100]
                break
        if cut is not None and len(original[:cut].strip()) >= 300:
            copy["text"] = original[:cut].rstrip()
            notes.append({
                "source_id": str(copy.get("id")),
                "source": copy.get("source"), "reason": "shared_cross_article_tail",
                "removed_chars": len(original) - cut,
                "boundary_preview": boundary,
            })
        output.append(copy)
    return output, notes


def _tokens(text: str) -> set[str]:
    return {w for w in WORDS.findall(normalize(text)) if w not in STOP}


def extract_grounded_facts(materials: list[dict], max_facts: int = 9
                           ) -> tuple[list[dict], list[str]]:
    """Strict literal anchored evidence, no lossy LLM quote transcription."""
    materials = sorted(materials, key=lambda m: (
        not bool(m.get("is_primary")), str(m.get("id"))))
    if not materials:
        return [], ["no_source_materials"]
    main = materials[0]
    title = str(main.get("title") or "").strip()
    body = str(main.get("text") or "").strip()
    lead = [s for s in sentences(body[:2800]) if len(s) >= 30]
    if len(body) < 120 or not lead:
        return [], ["insufficient_source_material"]
    event = title[:220] if EVENT.search(title) and not NON_NEWS.search(title) else ""
    if not event:
        for s in lead[:3]:
            if EVENT.search(s) and len(_tokens(title) & _tokens(s)) >= 1:
                event = s[:220]
                break
    if not event:
        return [], ["no_supported_news_event"]

    sid = str(main.get("id"))
    facts, seen = [], set()
    def add(kind: str, value: str, quote: str):
        value, quote = value.strip(), quote.strip()
        if (not value or not quote or len(quote) > 260 or len(value) > len(quote)
                or normalize(value) not in normalize(quote)
                or normalize(value) in seen or len(facts) >= max_facts):
            return
        seen.add(normalize(value))
        facts.append({"kind": kind, "value": value, "quote": quote, "source_id": sid})

    add("event", event, title if event in title else next(
        (s for s in lead if event in s), event))

    for match in QUOTED.finditer(title + " " + body[:550]):
        value = match.group()
        context = title if value in title else next(
            (s for s in lead[:4] if value in s), "")
        if context:
            start = context.find(value)
            category = "organization" if re.search(
                r"(?:компани|ао\b|ооо\b|госкорпораци|холдинг|групп|организаци|предприяти|завод)",
                context[max(0, start-55):start+len(value)+12], re.I
            ) else "object"
            add(category, value, context[:260] if value in context[:260] else value)

    title_words = _tokens(title)
    ranked = [(i, len(title_words & _tokens(s)), s)
              for i, s in enumerate(lead[:10])]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    for i, score, sentence in ranked[:6]:
        if len(facts) >= max_facts:
            break
        if i > 2 and score == 0:
            continue
        quote = sentence[:245]
        category = ("outcome" if re.search(
            r"\b(?:позволит|обеспечит|в результате|в итоге|благодаря)\b",
            quote, re.I) else "object")
        add(category, quote, quote)
        for match in NUMBER.finditer(quote):
            add("number", match.group(), quote)
            break

    for snippet in [title] + lead[:3]:
        if len(facts) >= max_facts:
            break
        match = YEAR.search(snippet)
        if match:
            add("date", match.group(), snippet[:260] if match.start() < 260
                else match.group())
            break
    return (facts, []) if len(facts) >= 3 else (
        facts, ["insufficient_supported_facts"])
