"""Narrow, high-confidence factual checks for Qwen summaries.

No regex can certify meaning. These checks only *reject* detected source
contradictions and preserve the existing human-approval requirement.
Unlike shared Gemma validators, this file can evolve independently.
"""
from __future__ import annotations

import re

from processing.abstractive_validator import NUMBER_RE, normalize


def _number_key(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("\u00a0", " ")).replace(",", ".").rstrip("%")


_UNIT_ALIASES = (
    (re.compile(r"^тонн\w*\b", re.I), "tonne"),
    (re.compile(r"^т\b", re.I), "tonne"),
    (re.compile(r"^метр\w*\b", re.I), "metre"),
    (re.compile(r"^м\b", re.I), "metre"),
)


def _explicit_physical_units(text: str) -> dict[str, set[str]]:
    """Resolve abbreviations only with an explicit unit immediately after a number.

    Never infer a physical unit from a narrative or a nearby unrelated figure.
    Other units remain covered by the original strict validator.
    """
    result: dict[str, set[str]] = {}
    for number in NUMBER_RE.finditer(text):
        following = text[number.end():number.end()+24].lstrip(" \t\u00a0")
        for pattern, canonical in _UNIT_ALIASES:
            if pattern.match(following):
                result.setdefault(_number_key(number.group()), set()).add(canonical)
                break
    return result


def equivalent_explicit_units(number: str, generated: str, original: str) -> bool:
    """Support 70 т == 70 тонн and 1000 м == 1000 метров, not 1000 км."""
    left = _explicit_physical_units(generated).get(number, set())
    right = _explicit_physical_units(original).get(number, set())
    return bool(left and right and left == right)


def focused_fact_issues(summary: str, materials: list[dict]) -> list[str]:
    """Spot dangerous, source-contradicting temporal or attribute bindings.

    Rules are deliberately narrow; full semantic verification remains human.
    """
    text = "\n".join(str(m.get("text") or "") for m in materials)
    norm_summary, norm_source = normalize(summary), normalize(text)
    issues = []

    # Planned jobs are not already created jobs. The source often combines a
    # completed plant launch with *future* jobs in the very same paragraph.
    future_jobs = re.findall(
        r"\bбуд(?:ет|ут)\s+создан\w*\s+(\d{1,6})\s+рабоч\w*\s+мест\w*",
        norm_source)
    for count in future_jobs:
        count = re.escape(count)
        if re.search(
            r"\b(?:создан\w*|создал\w*|появил\w*|"
            r"позволил\w*\s+созда(?:ть|ни\w*))\s+"
            + count + r"\s+рабоч\w*\s+мест\w*",
            norm_summary,
        ):
            issues.append("planned_jobs_reported_as_created")
            break

    # Explicit exception to a supposedly universal transport shutdown.
    # The summary must mention the exception, not just the "all sections" claim.
    source_exception = re.search(
        r"\bтолько\s+работ\w*\s+([а-яё-]{5,})\s+шлюз\w*"
        r"\s+продлен\w*\s+до", norm_source)
    universal_closure = re.search(
        r"(?:движени\w*\s+суд\w*\s+прекраща\w*|"
        r"судоходств\w*\s+прекраща\w*)\s+на\s+всех\s+участк\w*",
        norm_summary)
    if (source_exception and universal_closure
            and source_exception.group(1)[:5] not in norm_summary
            and not re.search(r"\b(?:исключени\w*|кроме|однако)\b", norm_summary)):
        issues.append("source_exception_omitted")

    # Catch a wrong object-material association: grip made from polymer
    # cannot support a claim that the stock is made from polymer.
    generated_stock = re.search(
        r"\bполимерн\w*\s+приклад\w*\b|"
        r"\bприклад\w*\s+из\s+(?:[а-яё-]+\s+){0,2}полимер\w*\b",
        norm_summary)
    original_stock = re.search(
        r"\bполимерн\w*\s+приклад\w*\b|"
        r"\bприклад\w*\s+из\s+(?:[а-яё-]+\s+){0,2}полимер\w*\b|"
        r"\bприклад\w*\s+выполнен\w*\s+из\s+(?:[а-яё-]+\s+){0,2}полимер\w*\b",
        norm_source)
    if generated_stock and not original_stock:
        issues.append("unsupported_material_binding:polymer_stock")

    return issues
