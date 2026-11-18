"""Conservative, deterministic event identity gate (stage 3).

SBERT is a candidate retriever, not an event-identity oracle. These rules deliberately
prefer missed merges to false merges. They use the headline and opening lead; no LLM,
DB, publisher, summarizer or tagger dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
import re

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CANDIDATE_THRESHOLD = 0.67
MERGE_THRESHOLD = 0.79

COMPANIES = {
    "оск": r"\b(?:оск|объединенн\w* судостроительн\w* корпораци\w*)\b",
    "чмз": r"\b(?:чмз|челябинск\w* механическ\w* завод\w*)\b",
    "ростех": r"\bростех\w*\b",
    "роснефть": r"\bроснефт\w*\b",
    "газпром": r"\bгазпром\w*\b",
    "норникель": r"\b(?:норникел\w*|норильск\w* никел\w*)\b",
    "русал": r"\bрусал\w*\b",
    "селигдар": r"\bселигдар\w*\b",
    "крэt": r"\b(?:крет|концерн радиоэлектронн\w* технолог\w*)\b",
    "свеза": r"\bсвез\w*\b",
    "тмк": r"\b(?:тмк|трубн\w* металлургическ\w* компани\w*)\b",
    "одк": r"\b(?:одк|объединенн\w* двигателестроительн\w* корпораци\w*)\b",
    "оак": r"\b(?:оак|объединенн\w* авиастроительн\w* корпораци\w*)\b",
    "алроса": r"\bалрос\w*\b",
    "росатом": r"\bросатом\w*\b",
    "пролетарский завод": r"\bпролетарск\w* завод\w*\b",
}
ACTIONS = {
    "production": r"\b(?:выпуст\w*|изготов\w*|произвел\w*|достиг\w* отметк\w*|перешагнул\w* рубеж\w*|производ\w* парти\w*|собрал\w* парти\w*)\b",
    "construction": r"\b(?:постро\w*|строительств\w*|возвод\w*|начал\w* строит\w*)\b",
    "launch": r"\b(?:запуст\w*|ввел\w* в эксплуатаци\w*|открыл\w* производств\w*)\b",
    "modernization": r"\b(?:модерниз\w*|усовершенств\w*|улучшил\w*|обновил\w*|модернизаци\w*)\b",
    "development": r"\b(?:разработ\w*|создал\w*|спроектир\w*)\b",
    "presentation": r"\b(?:представил\w*|презентовал\w*|продемонстрир\w*)\b",
    "tests": r"\b(?:испытан\w*|испытал\w*|протестир\w*)\b",
    "delivery": r"\b(?:поставил\w*|поставк\w*|передал\w* заказчик\w*|отгруз\w*)\b",
    "contract": r"\b(?:контракт\w*|соглашени\w*|договор\w*|подписал\w*)\b",
    "training": r"\b(?:подготовк\w* наставник\w*|обучени\w* наставник\w*|программ\w* наставничеств\w*)\b",
    "report": r"\b(?:итог\w*|результат\w*|показател\w*|объем\w* производств\w*)\b",
}
OBJECTS = {
    "autocranes": r"\bавтокран\w*\b",
    "workshop": r"\b(?:цех\w*|производственн\w* корпус\w*|здани\w* завод\w*)\b",
    "mentoring": r"\b(?:наставник\w*|наставничеств\w*)\b",
    "ships": r"\b(?:судн\w*|судов\w*|корабл\w*|ледокол\w*|фрегат\w*|корвет\w*)\b",
    "engines": r"\b(?:электродвигател\w*|двигател\w*|турбин\w*)\b",
    "aircraft": r"\b(?:самолет\w*|вертолет\w*|мс.21\b|су.34\b)\b",
    "tram": r"\b(?:трамва\w*|вагон\w*|локомотив\w*|тепловоз\w*)\b",
    "mining": r"\b(?:рудник\w*|шахт\w*|месторождени\w*)\b",
    "metal": r"\b(?:сталь\w*|металл\w*|прокат\w*|титан\w*)\b",
    "gold": r"\bзолот\w*\b",
    "coal": r"\bугл\w*\b",
    "transformer": r"\bтрансформатор\w*\b",
    "equipment": r"\b(?:оборудовани\w*|станк\w*|комплекс\w*)\b",
    "pipeline": r"\b(?:трубопровод\w*|газопровод\w*)\b",
    "plant": r"\b(?:завод\w*|фабрик\w*|предприяти\w*)\b",
}
# General categories do not override a more specific headline object.
GENERIC_OBJECTS = {"equipment", "plant"}
_MONTH = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_QUANTITY = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(тыс(?:яч\w*)?|млн|млрд)?\s*"
    r"(автокран\w*|суд\w*|вагон\w*|двигател\w*|станк\w*|единиц\w*|штук\w*|"
    r"тонн\w*|км|мвт|гвт|рубл\w*|руб\.?|%|процент\w*)\b"
)
_DATE = re.compile(r"\b([0-3]?\d)\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йя]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+(20\d\d)\b")
_MODEL = re.compile(r"\b[a-zа-я]{1,12}-\d{2,5}(?:[-/][a-zа-я0-9]+)?\b")
_STAGE = re.compile(r"\b(?:перв\w*|втор\w*|треть\w*|четверт\w*)\s+(?:этап\w*|очеред\w*|парти\w*)")
_PLANNED = re.compile(r"\b(?:планиру\w*|намерен\w*|собирает\w*|план\w* постро\w*|план\w* выпуст\w*)\b")
_STOP = {"россии", "российский", "компания", "завод", "новый", "новые", "предприятие",
         "выпуск", "выпустил", "производство", "система", "для", "на", "при", "из", "это"}


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def day(value: object) -> date | None:
    if isinstance(value, datetime):
        if value.tzinfo:
            value = value.astimezone(timezone.utc)
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
    return None


def _hits(patterns: dict[str, str], text: str) -> frozenset[str]:
    return frozenset(name for name, pattern in patterns.items() if re.search(pattern, text))


def _headline(row: object) -> str:
    return normalize(getattr(row, "title", "") if not isinstance(row, dict) else row.get("title"))


def _field(row: object, key: str):
    return row.get(key) if isinstance(row, dict) else getattr(row, key, None)


@dataclass(frozen=True)
class EventFacts:
    companies: frozenset[str]
    actions: frozenset[str]
    objects: frozenset[str]
    quantities: tuple[tuple[str, float], ...]
    event_date: date | None
    publication_date: date | None
    models: frozenset[str]
    stage: str
    planned: bool
    headline_tokens: frozenset[str]
    title: str


def extract_facts(row: object) -> EventFacts:
    title = _headline(row)
    lead = normalize(_field(row, "text"))[:320]
    short = title + " " + lead
    companies = _hits(COMPANIES, short)
    actions = _hits(ACTIONS, title) or _hits(ACTIONS, lead)
    objects = _hits(OBJECTS, title)
    if not objects:
        objects = _hits(OBJECTS, lead[:160])
    objects = objects - GENERIC_OBJECTS if objects - GENERIC_OBJECTS else objects
    amounts = []
    for value, multiplier, unit in _QUANTITY.findall(title + " " + lead[:180]):
        amount = float(value.replace(",", "."))
        if multiplier.startswith("тыс"):
            amount *= 1_000
        elif multiplier == "млн":
            amount *= 1_000_000
        elif multiplier == "млрд":
            amount *= 1_000_000_000
        unit = "autocranes" if unit.startswith("автокран") else (
            "ships" if unit.startswith("суд") else
            "units" if unit.startswith(("единиц", "штук")) else
            "tonnes" if unit.startswith("тонн") else
            "rub" if unit.startswith("руб") else unit[:5]
        )
        amounts.append((unit, amount))
    parsed_date = None
    match = _DATE.search(title)
    if match:
        m = normalize(match.group(2))
        month = next((number for key, number in _MONTH.items() if m.startswith(key)), None)
        if month:
            try:
                parsed_date = date(int(match.group(3)), month, int(match.group(1)))
            except ValueError:
                pass
    models = frozenset(m.group(0).replace(" ", "-") for m in _MODEL.finditer(title)
                       if not m.group(0).startswith("20"))
    stage_match = _STAGE.search(title)
    tokens = frozenset(t for t in re.findall(r"[а-яa-z0-9]{4,}", title)
                       if t not in _STOP)
    return EventFacts(companies, actions, objects, tuple(sorted(set(amounts))),
                      parsed_date, day(_field(row, "published_at")),
                      models, stage_match.group(0) if stage_match else "",
                      bool(_PLANNED.search(title)), tokens, title)


@dataclass(frozen=True)
class EventDecision:
    merge: bool
    similarity: float
    reason: str


def technical_specs(row: object) -> frozenset[tuple[str, float]]:
    """Distinctive measurements from an article opening, excluding bare years.

    2500 W + 600 kg are stronger evidence than a shared industry name.
    These are NOT used without company, action and object corroboration.
    """
    text = normalize(_field(row, "title")) + " " + normalize(_field(row, "text"))[:10000]
    pattern = re.compile(
        r"\b(\d+(?:[.,]\d+)?)\s*(квт|вт|мвт|гвт|кг|грамм\w*|мм|"
        r"л\.?\s*с\.?|километр\w*|метр\w*)\b"
    )
    result = set()
    for raw, unit in pattern.findall(text):
        value = float(raw.replace(",", "."))
        if unit.startswith("грамм"):
            unit = "г"
        if unit.startswith("кило"):
            unit = "км"
        if unit.startswith("метр"):
            unit = "м"
        result.add((unit, value))
    return frozenset(result)


def same_event(a: object, b: object, similarity: float,
               *, candidate_threshold: float = CANDIDATE_THRESHOLD,
               merge_threshold: float = MERGE_THRESHOLD) -> EventDecision:
    """Hard factual vetoes first; mere shared sector/company/number is insufficient."""
    score = float(similarity)
    if not math.isfinite(score) or score < candidate_threshold:
        return EventDecision(False, score if math.isfinite(score) else 0.0, "low_embedding")
    left, right = extract_facts(a), extract_facts(b)
    corroborating_specs = technical_specs(a) & technical_specs(b)
    strong_specs = len(corroborating_specs) >= 2
    if left.companies and right.companies and not left.companies.intersection(right.companies):
        return EventDecision(False, score, "different_company")
    if left.event_date and right.event_date and left.event_date != right.event_date:
        return EventDecision(False, score, "different_event_date")
    if left.models and right.models and not left.models.intersection(right.models):
        return EventDecision(False, score, "different_model")
    if left.stage and right.stage and left.stage != right.stage:
        return EventDecision(False, score, "different_stage")
    if left.planned != right.planned:
        return EventDecision(False, score, "plan_vs_completed")
    if left.actions and right.actions and not left.actions.intersection(right.actions):
        # Product development reported as a presentation can describe the
        # same unveiling. Require strong embedding + all remaining hard gates.
        compatible = (left.actions | right.actions) == {"development", "presentation"}
        if not compatible or (score < 0.88 and not (strong_specs and score >= 0.71)):
            return EventDecision(False, score, "different_action")
    if left.objects and right.objects and not left.objects.intersection(right.objects):
        return EventDecision(False, score, "different_object")
    quantities_a, quantities_b = {}, {}
    for unit, amount in left.quantities:
        quantities_a.setdefault(unit, set()).add(amount)
    for unit, amount in right.quantities:
        quantities_b.setdefault(unit, set()).add(amount)
    for unit in quantities_a.keys() & quantities_b.keys():
        if not any(math.isclose(x, y, rel_tol=0.01)
                   for x in quantities_a[unit] for y in quantities_b[unit]):
            return EventDecision(False, score, "different_quantity")
    if left.publication_date and right.publication_date:
        gap = abs((left.publication_date - right.publication_date).days)
        if gap > 45 or (gap > 14 and ("production" in left.actions or "production" in right.actions)):
            return EventDecision(False, score, "publication_gap")
    common_tokens = left.headline_tokens & right.headline_tokens
    # Both company AND action AND object needed for ordinary merges. Very close
    # paraphrases can omit company in one title, but still need action + object.
    company_ok = bool(left.companies & right.companies)
    action_ok = bool(left.actions & right.actions) or (left.actions | right.actions) == {"development", "presentation"}
    object_ok = bool(left.objects & right.objects)
    title_equal = left.title == right.title and bool(left.title)
    if not action_ok or not object_ok:
        # Text-only mirrors with identical headlines and a known company may pass.
        if title_equal and company_ok and score >= 0.94:
            return EventDecision(True, score, "identical_headline_verified")
        return EventDecision(False, score, "insufficient_action_object_evidence")
    if not company_ok and score < 0.90:
        return EventDecision(False, score, "company_not_confirmed")
    if len(common_tokens) < 2 and score < 0.90 and not strong_specs:
        return EventDecision(False, score, "weak_headline_evidence")
    threshold = merge_threshold if company_ok else max(0.90, merge_threshold)
    if company_ok and action_ok and object_ok and strong_specs:
        threshold = min(threshold, 0.71)
    if score < threshold:
        return EventDecision(False, score, "below_merge_threshold")
    return EventDecision(True, score, "shared_technical_specs" if strong_specs else "event_facts_agree")


def embedding_text(row: object) -> str:
    """Title repeated once; only the lead, not the unrelated full-article tail."""
    title = str(_field(row, "title") or "")
    lead = str(_field(row, "text") or "")[:500]
    return f"{title}. {title}. {lead}"


def representative_key(row: object) -> tuple:
    """Prefer informative article, then deterministic date/URL tie breakers."""
    title = str(_field(row, "title") or "")
    body = str(_field(row, "text") or "")
    quality = min(len(body), 4500) + min(len(title), 180) * 4
    quality += 250 if day(_field(row, "published_at")) else 0
    return (-quality, str(_field(row, "published_at") or ""), str(_field(row, "url") or ""))


def cosine(vec1, vec2) -> float:
    """No sklearn DLL dependency (works with numpy arrays or Python lists)."""
    if vec1 is None or vec2 is None:
        return 0.0
    a, b = list(vec1), list(vec2)
    if len(a) != len(b) or not a:
        return 0.0
    dot = math.fsum(float(x)*float(y) for x,y in zip(a,b))
    na = math.sqrt(math.fsum(float(x)**2 for x in a))
    nb = math.sqrt(math.fsum(float(y)**2 for y in b))
    return dot / (na*nb) if na and nb else 0.0


def choose_existing_cluster(row: object, embedding: list[float],
                            clusters: list[dict], *, threshold: float = MERGE_THRESHOLD):
    """Complete-link factual checks prevent single-link chaining and centroid drift.

    Each cluster dict contains 'items' (row+embedding tuples) and stable 'id'.
    Reject if ANY member contradicts the incoming article. Pick the highest
    MINIMUM member similarity, not the greatest centroid similarity.
    """
    candidates = []
    rejections = []
    for cluster in clusters:
        items = cluster.get("items", [])
        if not items:
            continue
        outcomes = [(other, same_event(row, other, cosine(embedding, other_embedding),
                                       merge_threshold=threshold))
                    for other, other_embedding in items]
        refused = [result for _, result in outcomes if not result.merge]
        if refused:
            strongest = max((result.similarity for _, result in outcomes), default=0.0)
            rejections.append((strongest, refused[0].reason, cluster["id"]))
            continue
        minimum = min(result.similarity for _, result in outcomes)
        candidates.append((minimum, cluster["id"], cluster))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], str(item[1])))
        return candidates[0][2], EventDecision(True, candidates[0][0], "all_members_agree")
    rejections.sort(key=lambda item: (-item[0], str(item[2])))
    if rejections:
        return None, EventDecision(False, rejections[0][0], rejections[0][1])
    return None, EventDecision(False, 0.0, "no_candidates")
