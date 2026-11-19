"""Conservative, reproducible checks for locally generated industrial news.

No lexical check or second LLM is proof of factual entailment: passing here
only produces a draft for human review, never permission to publish.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from os.path import commonprefix
import re

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"])")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?\s*%?(?!\w)")
QUOTES_RE = re.compile(r"«([^»]+)»")
ACRONYM_RE = re.compile(r"(?<!\w)[А-ЯЁA-Z]{2,}(?!\w)")
CAPITAL_RE = re.compile(r"(?<!\w)[А-ЯЁ][а-яё]{3,}(?!\w)")
TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
MONTHS = ("январ", "феврал", "март", "апрел", "мая", "мае", "май",
          "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр")
UNITS = ("млн", "млрд", "тыс", "руб", "тонн", "кг", "км", "мвт", "гвт",
         "гектар", "процент", "станк", "линий", "лини", "метр", "литр",
         "мощност", "год", "квт", "шт", "%", "единиц")
FUTURE_RE = re.compile(r"\b(?:планиру\w*|намерен\w*|предстоит|планах|планируется|собирается)\b", re.I)
DONE_RE = re.compile(r"\b(?:запустил\w*|начал\w* работу|вв[её]л\w*|построил\w*|выпустил\w*|завершил\w*|открыл\w*|создал\w*)\b", re.I)
NEGATION_RE = re.compile(r"\b(?:не|нет|без|отменил\w*|отказал\w*)\b", re.I)
RISE_RE = re.compile(r"\b(?:выросл\w*|увеличил\w*|повысил\w*|рост|увеличени\w*)\b", re.I)
FALL_RE = re.compile(r"\b(?:сократил\w*|снизил\w*|уменьшил\w*|падени\w*|снижени\w*)\b", re.I)
CAUSE_RE = re.compile(r"\b(?:из-за|благодаря|поскольку|вследствие|по причине)\b", re.I)
COMMON_STARTS = {"Компания", "Предприятие", "Завод", "Производство", "Организация",
                 "В", "На", "Для", "По", "Также", "Кроме", "Это", "При", "Согласно"}
TECHNICAL_OBJECT_STEMS = ("реактор", "турбин", "генератор", "двигател", "станк",
                          "самолет", "вертолет", "подлод", "танкер", "контейнер",
                          "спутник", "ракет", "конвейер", "электростанц",
                          "комбайн", "беспилотник", "экскаватор", "тепловоз",
                          "трансформатор", "компрессор", "робот", "рудник", "шахт")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("ё", "е").casefold()).strip()


def sentences(text: str) -> list[str]:
    return [x.strip() for x in SENTENCE_RE.split((text or "").strip()) if x.strip()]


def _stem_match(a: str, b: str) -> bool:
    """Case endings of the same long Cyrillic name, not arbitrary synonyms."""
    a, b = normalize(a), normalize(b)
    if a == b:
        return True
    if not a or not b or a[:5] != b[:5]:
        return False
    prefix = len(commonprefix([a, b]))
    return prefix >= max(5, min(len(a), len(b)) - 3)


def _inflected_token_in_source(word: str, source: str) -> bool:
    # Only candidates with a sufficiently long common lexical base.
    return any(_stem_match(word, match.group())
               for match in TOKEN_RE.finditer(source)
               if len(match.group()) >= 5)


def _inflected_phrase_in_source(value: str, source: str) -> bool:
    """A quoted name may be inflected, but may not change its word sequence."""
    words = TOKEN_RE.findall(value)
    if not words:
        return False
    source_words = TOKEN_RE.findall(source)
    count = len(words)
    return any(all(_stem_match(a, b) for a, b in zip(
                words, source_words[i:i+count]))
               for i in range(len(source_words) - count + 1))


def _number_key(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("\u00a0", " ")).replace(",", ".").rstrip("%")


def _all_sources(materials: list[dict]) -> str:
    return "\n".join((str(m.get("title") or "") + "\n" + str(m.get("text") or ""))
                     for m in materials)


def _unit_after(source: str, match: re.Match) -> str:
    tail = source[match.end():match.end()+38].lower().lstrip(" .,;:()-\n")
    if source[match.start():match.end()].strip().endswith("%"):
        return "%"
    for token in re.findall(r"[а-яёa-z%]+", tail, re.I)[:3]:
        for unit in UNITS:
            if token.startswith(unit):
                return unit
        if token in {"около", "более", "почти", "свыше", "до", "от", "и", "на", "в"}:
            continue
        break
    return ""


def _number_contexts(text: str) -> dict[str, set[str]]:
    contexts: dict[str, set[str]] = {}
    for match in NUMBER_RE.finditer(text):
        contexts.setdefault(_number_key(match.group()), set()).add(_unit_after(text, match))
    return contexts


@dataclass
class Validation:
    passed: bool
    issues: list[str]
    flags: list[str]
    chars: int
    sentence_count: int
    copied_sentence_ratio: float
    max_source_sentence_similarity: float
    repeated_sentence_count: int
    readability_proxy: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_summary(summary: str, materials: list[dict],
                     *, max_chars: int = 800) -> Validation:
    """Flag unsupported named anchors, numerical bindings and polarity changes.

    Source-wide checks avoid incorrectly requiring every source number/entity to
    appear in the shortened output. Fail closed on detected risks, NOT on an
    unverified claim that a text is factually correct.
    """
    summary = (summary or "").strip()
    source = _all_sources(materials)
    norm_source = normalize(source)
    parts = sentences(summary)
    issues: list[str] = []
    flags: list[str] = []
    if not source.strip():
        issues.append("empty_source")
    if not summary or len(summary) > max_chars:
        issues.append("empty_or_overlength")
    if summary and len(summary) < 80:
        flags.append("very_short")
    if len(parts) < 3 or len(parts) > 5:
        flags.append("outside_target_3_to_5_sentences")

    orig_numbers = _number_contexts(source)
    new_numbers = _number_contexts(summary)
    for number, units in new_numbers.items():
        if number not in orig_numbers:
            issues.append("unsupported_number:" + number)
        elif (not (len(number) == 4 and number.isdecimal() and 1900 <= int(number) <= 2100)
              and any(u and u not in orig_numbers[number] for u in units)):
            issues.append("number_unit_mismatch:" + number)
    for material in materials:
        headline = normalize(str(material.get("title") or ""))
        if len(headline) >= 28 and headline in normalize(summary):
            issues.append("headline_copied_verbatim")
            break
    for stem in TECHNICAL_OBJECT_STEMS:
        if (re.search(r"\b" + re.escape(stem), normalize(summary))
                and not re.search(r"\b" + re.escape(stem), norm_source)):
            issues.append("unsupported_technical_object:" + stem)
    for quoted in QUOTES_RE.findall(summary):
        if (normalize(quoted) not in norm_source
                and not _inflected_phrase_in_source(quoted, source)):
            issues.append("unsupported_quoted_name:" + quoted[:60])
    original_acronyms = set(ACRONYM_RE.findall(source))
    for acronym in ACRONYM_RE.findall(summary):
        if acronym not in original_acronyms:
            issues.append("unsupported_acronym:" + acronym)
    for match in CAPITAL_RE.finditer(summary):
        name = match.group()
        if name in COMMON_STARTS:
            continue
        if (name.casefold().replace("ё", "е") not in norm_source
                and not _inflected_token_in_source(name, source)):
            # Capitals in the middle of a sentence are particularly likely to be
            # named places/organizations; initial generic words are ambiguous.
            prefix = summary[:match.start()]
            if prefix and not re.search(r"[.!?]\s*$", prefix):
                issues.append("unsupported_proper_name:" + name)
    for month in MONTHS:
        if month in {"мая", "мае", "май"}:
            if (re.search(r"\b(?:мая|мае|май)\b", normalize(summary))
                    and not re.search(r"\b(?:мая|мае|май)\b", norm_source)):
                issues.append("unsupported_month:май")
            continue
        if month in normalize(summary) and month not in norm_source:
            issues.append("unsupported_month:" + month)
    # Main-event tense/forecast guards: source headlines are first-class facts.
    # They catch completed-vs-ongoing claims missed by the same LLM verifier.
    main_title = normalize(str(materials[0].get("title") or "")) if materials else ""
    first_sentence = normalize(parts[0]) if parts else ""
    if (re.search(r"\bзаверша(?:ется|ются|ют|ет)\b", main_title)
            and re.search(r"\b(?:завершил\w*|закончил\w*|окончил\w*)\b",
                          first_sentence)):
        issues.append("main_event_ongoing_changed_to_completed")
    if (re.search(r"\b(?:прогнозиру\w*|ожида\w*|планиру\w*)\b", main_title)
            and re.search(r"\b(?:зафиксировал\w*|достиг\w*|составил\w*|получил\w*)\b",
                          first_sentence)):
        issues.append("forecast_changed_to_observed_result")
    # These are coarse global guards. A claim-by-claim verification follows.
    if FUTURE_RE.search(source) and not DONE_RE.search(source) and DONE_RE.search(summary):
        issues.append("planned_event_changed_to_completed")
    if not NEGATION_RE.search(source) and NEGATION_RE.search(summary):
        issues.append("introduced_negation")
    if NEGATION_RE.search(source) and not NEGATION_RE.search(summary):
        flags.append("source_negation_not_represented")
    if RISE_RE.search(source) and not FALL_RE.search(source) and FALL_RE.search(summary):
        issues.append("growth_changed_to_decline")
    if FALL_RE.search(source) and not RISE_RE.search(source) and RISE_RE.search(summary):
        issues.append("decline_changed_to_growth")
    if CAUSE_RE.search(summary) and not CAUSE_RE.search(source):
        issues.append("unsupported_explicit_cause")

    src_sentences = [x for m in materials for x in sentences(
        (m.get("title") or "") + ". " + (m.get("text") or ""))]
    sims = [max((SequenceMatcher(None, normalize(part), normalize(original)).ratio()
                 for original in src_sentences), default=0.0) for part in parts]
    copied = sum(score >= 0.88 and len(TOKEN_RE.findall(part)) >= 8
                 for part, score in zip(parts, sims))
    repeated = sum(SequenceMatcher(None, normalize(a), normalize(b)).ratio() >= 0.83
                   for i, a in enumerate(parts) for b in parts[:i])
    if repeated:
        issues.append("repeated_sentences")
    if len(parts) >= 2 and copied / len(parts) > 0.50:
        flags.append("mostly_extractive")
    if len(parts) > 0 and sims and max(sims) > 0.93:
        flags.append("near_verbatim_sentence")
    # Sentence-length proxy ONLY; grammar and naturalness require human scoring.
    avg_words = len(TOKEN_RE.findall(summary)) / max(1, len(parts))
    readability_proxy = round(max(0.0, 1.0 - max(0.0, abs(avg_words - 16) - 8) / 20), 3) if parts else 0.0
    return Validation(
        passed=not issues, issues=list(dict.fromkeys(issues)),
        flags=list(dict.fromkeys(flags)), chars=len(summary),
        sentence_count=len(parts),
        copied_sentence_ratio=round(copied / max(1, len(parts)), 3),
        max_source_sentence_similarity=round(max(sims, default=0.0), 3),
        repeated_sentence_count=repeated,
        readability_proxy=readability_proxy,
    )


def check_semantic_verdict(answer: object, summary: str,
                           materials: list[dict]) -> list[str]:
    """LLM referee may *reject*, but cannot independently certify truth.

    Every cited evidence quote must literally exist in the submitted sources;
    one judgement per summary sentence is required. Verdict is advisory.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get("claims"), list):
        return ["semantic_verdict_missing"]
    claims = answer["claims"]
    parts = sentences(summary)
    if len(claims) != len(parts):
        return ["semantic_claim_count_mismatch"]
    source = normalize(_all_sources(materials))
    errors = []
    for i, (claim, part) in enumerate(zip(claims, parts), 1):
        if not isinstance(claim, dict) or normalize(str(claim.get("claim", ""))) != normalize(part):
            errors.append(f"semantic_claim_{i}_not_aligned")
            continue
        if claim.get("verdict") != "supported":
            errors.append(f"semantic_claim_{i}_" + str(claim.get("verdict", "unknown")))
        quote = claim.get("evidence", "")
        if not isinstance(quote, str) or len(quote.strip()) < 12 or normalize(quote) not in source:
            errors.append(f"semantic_claim_{i}_unsupported_evidence")
    return errors
