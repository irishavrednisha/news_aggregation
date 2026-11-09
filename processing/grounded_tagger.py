"""Deterministic industry tags grounded in the headline's principal event.

Pure Python, no Qwen, no DB, no model inference. SBERT scores are used only
to rank categories that have evidence in the event. Unclear cases abstain.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from processing.tagging_event_guard import nonindustrial_event_reason


@dataclass(frozen=True)
class TagDecision:
    tags: tuple[str, ...]
    outcome: str  # tagged, reject, review
    reason: str


def norm(s: str) -> str:
    return " ".join(str(s or "").casefold().replace("ё", "е").split())


EVENT = re.compile(
    r"\b(?:разработ\w*|созда\w*|создан\w*|изготов\w*|выпуст\w*|"
    r"выпуск\w*|производ\w*|запуст\w*|ввел\w*|постро\w*|"
    r"строительств\w*|модерниз\w*|усовершенств\w*|внедр\w*|"
    r"осво\w*|испыт\w*|спуст\w*|спущ\w*|заклад\w*|"
    r"реконстру\w*|восстанавлива\w*|восстанов\w*|"
    r"автоматиз\w*|переоснащ\w*|технологи\w*\s+ремонт\w*)\b"
)
OFFTOPIC = re.compile(
    r"\b(?:котировк\w*|дивиденд\w*|акционер\w*|страхов\w*\s+взнос\w*|"
    r"ипотек\w*|финансировани\w*|"
    r"гохран\w*.{0,70}приобрет\w*|приобрет\w*.{0,70}гохран\w*|"
    r"фонд\w*\s+для\s+рекультивац\w*|"
    r"обязат\w*.{0,80}фонд\w*|"
    r"протест\w*|забастовк\w*|"
    r"подвергл\w*\s+удар\w*|"
    r"праздничн\w*\s+акц\w*)\b"
)
FOREIGN = re.compile(
    r"(?:^naval group\b|^saab\b|^финлянди\w*\b|"
    r"^в\s+(?:южной\s+корее|бангладеш|ираке|японии|сша|франции)\b|"
    r"\bевропейск\w*\s+(?:корвет\w*|эсмин\w*|фрегат\w*)|"
    r"\bдатск\w*\s+флот\w*)"
)
RUSSIA_PARTICIPANT = re.compile(
    r"росси\w*|отечественн\w*|ростех|оск\b|росатом|"
    r"роснефт\w*|норникел\w*|русал|северстал\w*"
)

# Strong objects: require them to be mentioned in the event headline, not
# simply somewhere in a multi-topic article or company profile.
RULES = (
    ("судостроение", r"судостроен\w*|судоверф\w*|верф\w*|"
     r"судн\w*|ледокол\w*|теплоход\w*|танкер\w*|траулер\w*|"
     r"корвет\w*|фрегат\w*|эсмин\w*|подводн\w*\s+лодк\w*|"
     r"корабл\w*|катер\w*"),
    ("авиация", r"самолет\w*|вертолет\w*|авиадвигател\w*|"
     r"авиастроен\w*|летательн\w*\s+аппарат\w*|бпла\b|"
     r"ил-114|мс-21|sj-100|ssj|беспилотник\w*"),
    ("железнодорожное машиностроение", r"вагон\w*|локомотив\w*|"
     r"тепловоз\w*|электровоз\w*|электропоезд\w*|"
     r"железнодорожн\w*\s+подвижн\w*\s+состав\w*"),
    ("автомобильная промышленность", r"автозавод\w*|автомобил\w*|"
     r"автобус\w*|электробус\w*|троллейбус\w*|грузовик\w*"),
    ("атомная отрасль", r"аэс\b|атомн\w*\s+(?:электростанци\w*|"
     r"реактор\w*|энергоблок\w*)|ядерн\w*\s+(?:топлив\w*|реактор\w*)"),
    ("нефтегазовая промышленность", r"нефт\w*|газопровод\w*|"
     r"скважин\w*|бурени\w*|буров\w*|"
     r"сжиженн\w*\s+газ\w*|спг\b"),
    ("горнодобывающая промышленность", r"рудник\w*|шахт\w*|"
     r"золотодобыч\w*|горнодобыч\w*|"
     r"добыч\w*\s+(?:руд\w*|угл\w*|золот\w*|алмаз\w*)|"
     r"горн\w*\s+(?:пород\w*|масс\w*)"),
    ("металлургия", r"металлург\w*|сталеплав\w*|выплавк\w*|"
     r"доменн\w*|сталелитейн\w*|производств\w*\s+"
     r"(?:стали|алюминия|никеля|меди)"),
    ("химическая промышленность", r"химическ\w*\s+"
     r"(?:производств\w*|завод\w*|реактор\w*)|"
     r"удобрени\w*|полимер\w*|полипропилен\w*|нефтехим\w*"),
    ("электротехническое направление", r"радиоэлектрон\w*|"
     r"микросхем\w*|sim-карт\w*|sim-чип\w*|полупроводник\w*|"
     r"трансформатор\w*|предусилител\w*|электродвигател\w*|"
     r"электронн\w*\s+(?:компонент\w*|плат\w*)"),
    ("машиностроение", r"станк\w*|машиностроен\w*|"
     r"насос\w*|компрессор\w*|турбин\w*|"
     r"промышленн\w*\s+двигател\w*"),
    ("оборонная промышленность", r"вооружени\w*|бронетехник\w*|"
     r"ракетн\w*\s+комплекс\w*|"
     r"военн\w*\s+(?:техник\w*|оборудовани\w*)"),
    ("ТЭК", r"электростанци\w*|электрогенераци\w*|"
     r"энергоблок\w*|теплоэнергет\w*|электросет\w*|"
     r"электроэнерги\w*"),
    ("инжиниринг", r"инжиниринг\w*|"
     r"инженерн\w*\s+(?:проект\w*|разработк\w*)"),
    ("технологии", r"технологи\w*\s+(?:ремонт\w*|лить\w*|"
     r"печати|обработк\w*|сварк\w*)|разработан\w*\s+метод\w*"),
    ("инновации", r"запатентовал\w*|"
     r"инновационн\w*\s+(?:материал\w*|разработк\w*|технологи\w*)"),
)
COMPILED = tuple((tag, re.compile(rf"(?:{expression})"))
                 for tag, expression in RULES)
GENERIC_PRODUCTION = re.compile(
    r"(?:запуст\w*|осво\w*|налаж\w*|расшир\w*|"
    r"серийн\w*).{0,95}(?:производств\w*|выпуск\w*|"
    r"технологическ\w*\s+лини\w*)|"
    r"(?:производств\w*|выпуск\w*).{0,60}"
    r"(?:запущ\w*|запуст\w*|модерниз\w*|восстанов\w*)"
)
FACTORY_ACTION = re.compile(
    r"(?:завод\w*|фабрик\w*|предприяти\w*).{0,75}"
    r"(?:восстанов\w*|восстанавлива\w*|запуст\w*|откр\w*|постро\w*)|"
    r"(?:запуст\w*|откр\w*|постро\w*|восстанов\w*).{0,70}"
    r"(?:завод\w*|фабрик\w*|предприяти\w*)"
)


def select_tags(title: str, summary: str = "", *,
                ranked: list[dict] | None = None,
                max_tags: int = 2) -> TagDecision:
    """Ground tags to actual headline action; never tag from SBERT alone."""
    if max_tags < 1:
        raise ValueError("max_tags должен быть положительным")
    headline, lead = norm(title), norm(summary)[:650]
    if not headline:
        return TagDecision((), "review", "empty_title")
    event_reason = nonindustrial_event_reason(title)
    if event_reason:
        return TagDecision((), "reject", event_reason)
    if OFFTOPIC.search(headline):
        return TagDecision((), "reject", "nonindustrial_main_event")
    if FOREIGN.search(headline) and not RUSSIA_PARTICIPANT.search(headline):
        return TagDecision((), "reject", "foreign_project_without_russia")
    if not EVENT.search(headline):
        return TagDecision((), "review", "no_industrial_action_in_headline")

    # Score breaks ties only among categories supported by event words.
    scores = {
        str(row["tag"]): float(row.get("final_score") or 0)
        for row in (ranked or [])
        if isinstance(row, dict) and row.get("tag") is not None
    }
    candidates = {}
    for tag, pattern in COMPILED:
        if pattern.search(headline):
            candidates[tag] = 3.0 + scores.get(tag, 0)
    if GENERIC_PRODUCTION.search(headline):
        candidates["технологическое производство"] = (
            3.15 + scores.get("технологическое производство", 0)
        )
    if FACTORY_ACTION.search(headline):
        candidates["заводы"] = 3.1 + scores.get("заводы", 0)
    if not candidates:
        return TagDecision((), "review", "industry_not_explicit")

    # Prevent incidental related concepts from becoming separate industries.
    if "судостроение" in candidates and re.search(
        r"судн\w*|ледокол\w*|теплоход\w*|"
        r"корвет\w*|эсмин\w*|корабл\w*|фрегат\w*", headline
    ):
        candidates.pop("авиация", None)
        candidates.pop("электротехническое направление", None)
    if "нефтегазовая промышленность" in candidates and not re.search(
        r"рудник\w*|руд\w*|шахт\w*|горн\w*", headline
    ):
        candidates.pop("горнодобывающая промышленность", None)
    if "электротехническое направление" in candidates and not re.search(
        r"металлург\w*|выплавк\w*|сталеплав\w*", headline
    ):
        candidates.pop("металлургия", None)
    tags = tuple(sorted(candidates, key=lambda t: (-candidates[t], t))[:max_tags])
    return TagDecision(tags, "tagged", "evidence_in_principal_event")
