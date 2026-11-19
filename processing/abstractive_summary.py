"""Two-stage, strictly local Gemma summarization of *event clusters*.

TextRank is retained as a baseline only. Nothing in this module overwrites
News.text, Cluster.summary_text, moderator decisions, or the MAX queue.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Callable

from processing.abstractive_validator import (
    check_semantic_verdict, normalize, sentences, validate_summary,
)
from processing.extractive_rewriter import ollama_rewrite

ALLOWED_MODELS = frozenset({"gemma3:4b", "gemma3:1b", "gemma2:2b"})
DEFAULT_MODEL = "gemma3:4b"
PARAMETERS = {"temperature": 0, "num_ctx": 4096, "num_predict": 1024}
KINDS = frozenset({"organization", "event", "object", "location", "date", "number", "outcome"})
FACT_FORMAT = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": sorted(KINDS)},
            "value": {"type": "string"}, "quote": {"type": "string"},
            "source_id": {"type": "string"}},
            "required": ["kind", "value", "quote", "source_id"]}},
    },
    "required": ["facts"],
}
SUMMARY_FORMAT = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}
VERDICT_FORMAT = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "object",
        "properties": {"claim": {"type": "string"}, "verdict": {
            "type": "string", "enum": ["supported", "unsupported", "uncertain"]},
            "evidence": {"type": "string"}},
        "required": ["claim", "verdict", "evidence"]}}},
    "required": ["claims"],
}

FACT_SYSTEM = """Ты извлекаешь ТОЛЬКО непосредственно подтвержденные факты одной промышленной новости.
Даны документы одного событийного кластера. Не делай выводы, не объединяй разные
события и не додумывай причины. Верни JSON facts (не более 15).
Поля kind: organization, event, object, location, date, number, outcome.
Каждый факт: kind, value, quote, source_id. Значение value должно быть ДОСЛОВНЫМ
непрерывным фрагментом цитаты quote; quote — ДОСЛОВНЫМ непрерывным фрагментом
указанного документа, не более 260 символов. source_id скопируй точно.
Факт о событии (event) обязателен. Если факт неизвестен, не добавляй его.
Не добавляй предположения, рекламу и оценки. Никакого текста вне JSON."""

SUMMARY_SYSTEM = """Напиши НОВУЮ самостоятельную связную новость на естественном русском языке.
Вход: подтвержденные ДОСЛОВНЫМИ цитатами факты и короткие фрагменты оригинала.
Опирайся ТОЛЬКО на эти факты и тексты: не добавляй новые организации, объекты,
цифры, даты, место, цели, причины, результат или статус завершения.
Напиши сводку целиком заново, не заменяй пословно каждый исходный оборот.
Начни с главного события, затем контекст и значимые детали. 3–5 предложений,
до 800 символов. Без повтора заголовка целиком, списков, рекламы, канцеляризмов
и дублирования смыслов. Соблюдай различия между планом и фактом, ростом и
падением. Если достоверной связной сводки сделать нельзя, верни пустую summary.
Верни только JSON: {"summary": "..."}."""

VERDICT_SYSTEM = """Проверь каждое предложение сводки на предмет фактической опоры.
Верни JSON claims: ПО ОДНОМУ элементу на предложение В ИСХОДНОМ ПОРЯДКЕ.
claim = предложение сводки дословно. verdict = supported, unsupported, uncertain.
Для supported поле evidence должно содержать ДОСЛОВНЫЙ фрагмент исходных
документов, который поддерживает ВСЕ утверждения предложения. Если оно состоит
из нескольких независимых фактов без единой цитаты, ответь uncertain. В поле
evidence не придумывай цитат. При любом сомнении выбирай uncertain.
Никакого текста вне JSON."""


def _source_text(material: dict) -> str:
    return str(material.get("title") or "") + "\n" + str(material.get("text") or "")


def _selected_sources(materials: list[dict], max_chars: int = 5300) -> list[dict]:
    """Bound the local 4096-token context; prefer primary, then new sources."""
    ordered = sorted(materials, key=lambda m: (not bool(m.get("is_primary")), str(m.get("id"))))
    result = []
    remaining = max_chars
    for material in ordered[:5]:
        if remaining < 350:
            break
        body = _source_text(material)
        # Reserve space for other sources; never silently pass giant scraped pages.
        size = min(len(body), 2100, remaining)
        result.append({"source_id": str(material.get("id")), "text": body[:size]})
        remaining -= size
    return result


def _validate_facts(response: object, materials: list[dict]) -> tuple[list[dict], list[str]]:
    if not isinstance(response, dict) or not isinstance(response.get("facts"), list):
        return [], ["facts_bad_json_schema"]
    source_by_id = {str(m.get("id")): normalize(_source_text(m)) for m in materials}
    facts, issues, keys = [], [], set()
    if len(response["facts"]) > 15:
        issues.append("facts_over_limit")
    for i, item in enumerate(response["facts"][:15], 1):
        if not isinstance(item, dict) or not {"kind", "value", "quote", "source_id"}.issubset(item):
            issues.append(f"fact_{i}_wrong_structure")
            continue
        kind, value, quote, source_id = (item.get(k) for k in ("kind", "value", "quote", "source_id"))
        if not all(isinstance(x, str) for x in (kind, value, quote, source_id)):
            issues.append(f"fact_{i}_wrong_type")
            continue
        value, quote = value.strip(), quote.strip()
        norm_value, norm_quote = normalize(value), normalize(quote)
        if (kind not in KINDS or not source_by_id.get(source_id) or len(quote) > 260
                or len(quote) < 8 or not norm_value or norm_value not in norm_quote
                or norm_quote not in source_by_id[source_id]):
            issues.append(f"fact_{i}_unverifiable_source_quote")
            continue
        key = (kind, norm_value)
        if key in keys:
            continue
        keys.add(key)
        facts.append({"kind": kind, "value": value, "quote": quote, "source_id": source_id})
    if not any(x["kind"] == "event" for x in facts):
        issues.append("missing_supported_event")
    if len(facts) < 3:
        issues.append("insufficient_supported_facts")
    return facts, issues


def _messages_for_facts(materials: list[dict]) -> list[dict]:
    return [{"role": "system", "content": FACT_SYSTEM}, {
        "role": "user", "content": json.dumps(_selected_sources(materials), ensure_ascii=False)}]


def _messages_for_summary(facts: list[dict], materials: list[dict],
                          previous_issues: list[str]) -> list[dict]:
    content = {"facts": facts, "source_fragments": _selected_sources(materials),
               "previous_validation_issues": previous_issues}
    return [{"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]


def _messages_for_verdict(summary: str, materials: list[dict]) -> list[dict]:
    return [{"role": "system", "content": VERDICT_SYSTEM},
            {"role": "user", "content": json.dumps({
                "source_fragments": _selected_sources(materials), "summary": summary,
            }, ensure_ascii=False)}]


def _model_check(model: str):
    if model not in ALLOWED_MODELS:
        raise ValueError("Only local 1B/2B/4B Gemma tags are allowed; no cloud/large model")


@dataclass
class SummaryOutcome:
    cluster_id: int | str
    source_materials: list[dict]
    source_sha256: str
    facts: list[dict]
    textrank_summary: str
    gemma_summary: str
    status: str
    model: str
    model_parameters: dict
    validation: dict
    attempts: list[dict]
    created_utc: str

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_event_cluster(
    cluster_id: int | str, materials: list[dict], textrank_summary: str,
    *, model: str = DEFAULT_MODEL, request: Callable | None = None,
    max_retries: int = 2,
) -> SummaryOutcome:
    """Return an auditable draft or summary_review; never approve publication."""
    _model_check(model)
    if not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be 0..2")
    materials = [{"id": str(m.get("id")), "url": str(m.get("url") or ""), "source": str(m.get("source") or ""),
                  "title": str(m.get("title") or ""), "text": str(m.get("text") or ""),
                  "published_at": str(m.get("published_at") or ""),
                  "is_primary": bool(m.get("is_primary"))}
                 for m in materials]
    digest = hashlib.sha256(json.dumps(materials, ensure_ascii=False,
                                      sort_keys=True).encode("utf-8")).hexdigest()
    created = datetime.now(timezone.utc).isoformat()
    outcome = SummaryOutcome(
        cluster_id=cluster_id, source_materials=materials, source_sha256=digest,
        facts=[], textrank_summary=textrank_summary or "", gemma_summary="",
        status="summary_review", model=model, model_parameters=dict(PARAMETERS),
        validation={"passed": False, "issues": []}, attempts=[], created_utc=created)
    if (not materials or not any(len((m["text"] or "").strip()) >= 120
                                 for m in materials)):
        outcome.validation["issues"] = ["insufficient_source_material"]
        return outcome
    call = request if request is not None else ollama_rewrite
    try:
        raw = call(_messages_for_facts(materials), FACT_FORMAT, model=model)
        facts, fact_issues = _validate_facts(raw, materials)
    except Exception as exc:
        facts, fact_issues = [], ["fact_extraction_error:" + type(exc).__name__]
    outcome.facts = facts
    if fact_issues:
        outcome.validation["issues"] = fact_issues
        return outcome

    prior_issues: list[str] = []
    for attempt_no in range(1, max_retries + 2):
        entry: dict = {"attempt": attempt_no, "summary": "", "validation": {},
                       "semantic_verdict": None}
        outcome.attempts.append(entry)
        try:
            result = call(_messages_for_summary(facts, materials, prior_issues),
                          SUMMARY_FORMAT, model=model)
            if not isinstance(result, dict) or not isinstance(result.get("summary"), str):
                raise ValueError("summary_bad_json_schema")
            candidate = result["summary"].strip()
        except Exception as exc:
            entry["validation"] = {"passed": False, "issues": [
                "generation_error:" + type(exc).__name__]}
            prior_issues = entry["validation"]["issues"]
            continue
        entry["summary"] = candidate
        check = validate_summary(candidate, materials)
        entry["validation"] = check.to_dict()
        issues = list(check.issues)
        if not candidate:
            issues.append("empty_generation")
        if check.copied_sentence_ratio > .50:
            issues.append("mostly_extractive")
        if not issues:
            try:
                verdict = call(_messages_for_verdict(candidate, materials),
                               VERDICT_FORMAT, model=model)
                entry["semantic_verdict"] = verdict
                issues += check_semantic_verdict(verdict, candidate, materials)
            except Exception as exc:
                issues.append("semantic_verifier_error:" + type(exc).__name__)
        if issues:
            entry["validation"]["issues"] = list(dict.fromkeys(issues))
        entry["validation"]["passed"] = not issues
        outcome.gemma_summary = candidate  # rejected candidate remains diagnostic, not published
        outcome.validation = entry["validation"]
        if not issues:
            outcome.status = "summary_draft"  # NEVER 'ready' or 'published'
            break
        prior_issues = issues

    if not outcome.attempts:
        outcome.validation["issues"] = ["no_generation_attempt"]
    elif not outcome.gemma_summary:
        outcome.validation = outcome.attempts[-1]["validation"]
    return outcome
