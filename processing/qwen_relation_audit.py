"""Conservative, inspectable relationship checks for Stage 4 Qwen.

These checks can request editorial review, never certify factual correctness.
Keep them separate from generation and from industrial-topic filtering.
Unknown subjects/metrics are deliberately left unknown rather than guessed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from processing.abstractive_validator import NUMBER_RE, normalize, sentences

_NUMERIC = re.compile(r"(?<!\w)(\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?)\s*(%)?")
_COMPLETION = re.compile(r"\b(?:завершил\w*|законч\w*|окончил\w*|модернизировал\w*)\b")
_CURRENT_YEAR = re.compile(r"\b(?:в текущем году|в этом году|в нынешнем году)\b")
_FACTUAL_SPENDING = re.compile(
    r"\b(?:обошл\w*|потратил\w*|затратил\w*|расход\w*|"
    r"стоимост\w*\s+внедрени\w*|вложил\w*)\b"
)
_CONDITIONAL = re.compile(
    r"\b(?:условн\w*\s+расч\w*|предполож\w*|"
    r"если\s+самосвал|пример\w*\s+расч\w*|"
    r"не\s+оценк\w*\s+фактическ\w*)\b"
)
_POWER = re.compile(r"\bмощност\w*\b")
_VOLTAGE = re.compile(r"\b(?:напряжени\w*|вольт\w*)\b")
_VOLT = re.compile(r"(?<!\w)(?:\d+(?:[.,]\d+)?)\s*В\b")
_YEAR = re.compile(r"\b20\d{2}\b")
_SCOPE_STRONG = re.compile(r"\b(?:крепк\w*\s+алкогол\w*|напитк\w*\s+крепче\s+9\s*%)\b")
_SCOPE_GENERAL = re.compile(r"\b(?:производств\w*|выпуск\w*|продаж\w*)\s+алкогол\w*(?:\s*\(|\b)")
_REPORTING = re.compile(r"\b(?:публикац\w*|сообщени\w*|отчёт\w*|новост\w*)\b")

_METRICS = {
    "export": re.compile(r"\bэкспорт\w*\b"),
    "loading": re.compile(r"\b(?:погруз\w*|погруж\w*|отгруз\w*)\b"),
    "sales": re.compile(r"\b(?:продаж\w*|реализац\w*|спрос\w*)\b"),
    "investment": re.compile(r"\b(?:инвестиц\w*|вложени\w*|вложил\w*)\b"),
    "cost": re.compile(r"\b(?:расход\w*|затрат\w*|обошл\w*|потратил\w*)\b"),
    "economic_effect": re.compile(r"\b(?:экономи\w*|сэконом\w*|эффект\w*)\b"),
    "power": _POWER,
    "voltage": _VOLTAGE,
    "production": re.compile(r"\b(?:производств\w*|выпуск\w*|розлив\w*)\b"),
}


@dataclass(frozen=True)
class RelationFact:
    value: str
    metric: str | None
    unit: str | None
    subject_scope: str | None
    status: str
    source_id: str
    excerpt: str


def _number(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(",", ".")


def _scope(context: str) -> str | None:
    if _SCOPE_STRONG.search(context):
        return "strong_alcohol"
    if _SCOPE_GENERAL.search(context):
        return "all_alcohol"
    return None


def _metric(context: str) -> str | None:
    """Find an explicit nearby indicator, not a global document keyword."""
    candidates = []
    for kind, pattern in _METRICS.items():
        for match in pattern.finditer(context):
            candidates.append((match.end(), kind))
    if not candidates:
        return None
    # "экспортные отгрузки" and "отгрузки на экспорт" describe export.
    last = max(candidates)
    if last[1] == "loading" and _METRICS["export"].search(context[-65:]):
        return "export"
    if last[1] == "export" and re.search(r"\bобщ\w*\s+(?:объ[её]м\w*\s+)?(?:погруз\w*|отгруз\w*)", context):
        # Explicit "overall loading" takes precedence over a remote mention of export.
        if context.rfind("общ") > context.rfind("экспорт"):
            return "loading"
    return last[1]


def _unit(text: str, match: re.Match) -> str | None:
    if (match.group(2) or "") == "%":
        return "%"
    tail = text[match.end():match.end()+24]
    m = re.match(r"\s*(млн|млрд|тыс\.?|квт|мвт|кг|км|тонн\w*|т\b|рубл\w*|в\b|вольт\w*)", tail, re.I)
    if not m:
        return None
    val = normalize(m.group(1))
    if val in ("т", "тонна", "тонн", "тонны") or val.startswith("тонн"):
        return "tonne"
    if val in ("в",) and tail.strip().startswith("В"):
        return "volt"
    return val


def _sentences(text: str) -> list[str]:
    return sentences(text.replace("\n", " "))


def _extract_facts(text: str, sid: str) -> list[RelationFact]:
    result = []
    for sent in _sentences(text):
        norm = normalize(sent)
        for number in _NUMERIC.finditer(sent):
            value = _number(number.group(1))
            if _YEAR.fullmatch(value):
                continue
            before = normalize(sent[max(0, number.start()-145):number.start()])
            after = normalize(sent[number.end():min(len(sent), number.end()+50)])
            # Avoid borrowing a metric across clauses with different subjects.
            clause = re.split(r"(?:[,;]|\bа\b|\bтогда\s+как\b)", before)[-1]
            indicator = _metric(clause) or _metric(before[-100:])
            local = before[-95:] + " " + after[:35]
            status = "conditional" if _CONDITIONAL.search(local) else "stated"
            result.append(RelationFact(
                value=value, metric=indicator, unit=_unit(sent, number),
                subject_scope=_scope(before[-140:]), status=status,
                source_id=sid, excerpt=sent[:300]))
    return result


def _respective_percent_facts(material: dict) -> list[RelationFact]:
    """Resolve two ordered percentage changes when source says 'respectively'.

    The preceding sentence must explicitly identify both loading and export.
    If the ordered antecedents are not identifiable, infer nothing.
    """
    result = []
    lines = _sentences(str(material.get("text") or ""))
    for i in range(1, len(lines)):
        current, preceding = normalize(lines[i]), normalize(lines[i-1])
        if "соответственно" not in current:
            continue
        amounts = [_number(m.group(1)) for m in _NUMERIC.finditer(lines[i]) if m.group(2)]
        if len(amounts) != 2:
            continue
        if not (_METRICS["loading"].search(preceding)
                and _METRICS["export"].search(preceding)):
            continue
        l = _METRICS["loading"].search(preceding)
        e = _METRICS["export"].search(preceding)
        if not l or not e:
            continue
        kinds = ("loading", "export") if l.start() < e.start() else ("export", "loading")
        for value, kind in zip(amounts, kinds):
            result.append(RelationFact(
                value=value, metric=kind, unit="%", subject_scope=None,
                status="stated", source_id=str(material.get("id")),
                excerpt=lines[i][:300]))
    return result


def audit_relations(summary: str, materials: list[dict]) -> dict:
    """Return conservative blockers, extracted facts and unknown bindings.

    No guessed true/false entailment, no model calls, no publication decisions.
    """
    problems: list[str] = []
    source_facts: list[RelationFact] = []
    generated = _extract_facts(summary, "generated")
    for m in materials:
        sid = str(m.get("id") or "")
        body = str(m.get("text") or "")
        title = str(m.get("title") or "")
        source_facts.extend(_extract_facts(body, sid))
        source_facts.extend(_extract_facts(title, sid + ":title"))
        source_facts.extend(_respective_percent_facts(m))

    # Resolve ordered references in preference to the unanchored percentage
    # expressions in the sentence that contains 'respectively'.
    for fact in generated:
        if fact.unit != "%" or fact.metric not in ("loading", "export"):
            continue
        explicit = [f for f in source_facts if f.value == fact.value
                    and f.unit == "%" and f.metric in ("loading", "export")]
        if not explicit:
            continue
        distinct = {f.metric for f in explicit}
        if fact.metric not in distinct:
            problems.append(f"metric_binding_mismatch:{fact.value}:{fact.metric}")
        elif len(distinct) > 1:
            problems.append(f"metric_binding_ambiguous:{fact.value}")

    # A reported investment, expense, revenue or saving is not interchangeable
    # with the same number attached to a different financial role.
    financial_roles = {"investment", "cost", "economic_effect", "sales"}
    for fact in generated:
        if fact.metric not in financial_roles:
            continue
        values = {f.metric for f in source_facts if f.value == fact.value
                  and f.metric in financial_roles}
        if values and fact.metric not in values:
            problems.append("financial_role_mismatch:" + fact.value + ":" + fact.metric)

    # Scope belongs to the measurement, not merely the article. Check only
    # distinctive non-percent figures where scope is explicitly stated.
    for fact in generated:
        if fact.subject_scope != "strong_alcohol" or fact.unit == "%":
            continue
        alternatives = [f for f in source_facts if f.value == fact.value
                        and f.subject_scope is not None]
        if alternatives and all(x.subject_scope == "all_alcohol" for x in alternatives):
            problems.append("quantity_scope_mismatch:" + fact.value)

    # Physics: volts are voltage, never power. V is case-sensitive in source
    # and summary; lowercase 'в' is a Russian preposition, not a unit.
    for sentence in _sentences(summary):
        if _VOLT.search(sentence) and _POWER.search(normalize(sentence)):
            problems.append("dimension_mismatch:power_vs_voltage")
            break

    # A hypothetical cost is not a completed investment. Use context of each
    # matching number, not just an article-wide 'if'.
    source_text = "\n".join(str(m.get("text") or "") for m in materials)
    norm_source = normalize(source_text)
    if _FACTUAL_SPENDING.search(normalize(summary)):
        for num in _NUMERIC.finditer(summary):
            value = _number(num.group(1))
            if _YEAR.fullmatch(value):
                continue
            numeric_occurrences = [m for m in _NUMERIC.finditer(source_text)
                                   if _number(m.group(1)) == value]
            if not numeric_occurrences:
                continue  # Baseline checker already handles unsupported numbers.
            status = []
            for m in numeric_occurrences:
                vicinity = normalize(source_text[max(0, m.start()-430):m.end()+150])
                # An explicit completed spend in the same sentence takes
                # precedence over a nearby *different* hypothetical example.
                left = max(source_text.rfind(".", 0, m.start()),
                           source_text.rfind("!", 0, m.start()),
                           source_text.rfind("?", 0, m.start()))
                right = source_text.find(".", m.end())
                sentence = normalize(source_text[left+1:right if right != -1 else None])
                actual_spend = re.search(
                    r"\b(?:инвестировал\w*|инвестировано|вложил\w*|"
                    r"потратил\w*|затратил\w*|фактически\s+потрачено)\b",
                    sentence)
                status.append(bool(_CONDITIONAL.search(vicinity))
                              and not bool(actual_spend))
            if all(status):
                problems.append("conditional_amount_presented_as_spending:" + value)

    # Relative 'this year' is tied to publication year, not to the source
    # event year. Only flag when the source has a single explicit completion year.
    published_years = {str(m.get("published_at") or "")[:4] for m in materials}
    if len(published_years) == 1:
        publication_year = next(iter(published_years))
        if _YEAR.fullmatch(publication_year):
            completion_years = set()
            for line in _sentences(source_text):
                if not _COMPLETION.search(normalize(line)):
                    continue
                for year in re.findall(r"\b20\d{2}\b", line):
                    completion_years.add(year)
            if (len(completion_years) == 1
                    and publication_year not in completion_years
                    and any(_CURRENT_YEAR.search(normalize(line))
                            and _COMPLETION.search(normalize(line))
                            for line in _sentences(summary))):
                problems.append("completed_event_moved_to_publication_year")

    return {
        "issues": list(dict.fromkeys(problems)),
        "generated_facts": [asdict(f) for f in generated],
        "source_facts": [asdict(f) for f in source_facts],
        "verification_level": "heuristic_review_only",
    }
