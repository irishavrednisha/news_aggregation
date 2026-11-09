"""TextRank -> optional local Gemma draft with fail-closed publication policy.

This is an orchestration layer, not a factual entailment verifier.
A successful model rewrite is *always* review_required before MAX.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from processing.extractive_rewriter import rewrite_extractive, rewrite_extractive_whole


@dataclass(frozen=True)
class SummaryDraft:
    extractive: str
    text: str
    rewritten: bool
    review_required: bool
    warnings: tuple[str, ...] = ()


def rewrite_summary_draft(
    extractive: str, *, backend: str = "gemma", model: str = "gemma3:4b",
    request: Callable | None = None,
    rewrite_mode: str = "whole",
) -> SummaryDraft:
    backend = backend.strip().casefold()
    if backend not in {"gemma", "textrank"}:
        raise ValueError("NEWS_SUMMARY_BACKEND должен быть gemma или textrank")
    text = (extractive or "").strip()
    if not text:
        return SummaryDraft("", "", False, False, ("empty_extractive",))
    if backend == "textrank":
        return SummaryDraft(text, text, False, False)
    if not model.startswith(("gemma3:", "gemma2:")) or "cloud" in model.casefold():
        raise ValueError("NEWS_GEMMA_MODEL: допускаются только локальные gemma3:/gemma2:")
    if rewrite_mode not in ("whole", "aligned"):
        raise ValueError("NEWS_GEMMA_REWRITE_MODE должен быть whole или aligned")
    rewrite = rewrite_extractive_whole if rewrite_mode == "whole" else rewrite_extractive
    result = rewrite(text, model=model, request=request)
    # Mechanical guards cannot prove semantic equivalence.
    return SummaryDraft(
        extractive=result.extractive,
        text=result.text,
        rewritten=result.rewritten,
        review_required=True,  # all Gemma attempts require source review, even on fallback
        warnings=tuple(result.warnings),
    )


def tagged_summary_status(tags: list[str], *, review_pending: bool) -> str:
    """Tagging can reject a draft but must never approve a Gemma rewrite."""
    if not tags:
        return "not_for_publication"
    return "review_required" if review_pending else "ready"
