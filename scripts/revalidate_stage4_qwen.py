"""Re-check an old (v1) Qwen report without Ollama or regeneration.

This does NOT change historical model output or prove factual correctness.
It builds new literal source-backed claims from existing Qwen summary text.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.qwen_abstractive_summary import (
    VERSION, attach_evidence, canonical_materials, source_digest,
    validate_response,
)


def check_report(report: dict) -> dict:
    if (report.get("stage") != 4
            or not isinstance(report.get("clusters"), list)):
        raise ValueError("Expected an existing Stage 4 Qwen diagnostic report")
    results = []
    counts = Counter()
    for previous in report["clusters"]:
        row = dict(previous)
        original = canonical_materials(row.get("source_materials") or [])
        if row.get("source_sha256") != source_digest(original):
            raise ValueError("Source digest mismatch: " + str(row.get("cluster_id")))
        # Only the originally generated candidate, never a new model inference.
        summary = row.get("qwen_summary") or ""
        recent_raw = next(
            (attempt.get("response") for attempt in reversed(row.get("attempts") or [])
             if isinstance(attempt.get("response"), dict)), {})
        if summary:
            raw = {
                "main_event": row.get("main_event") or recent_raw.get("main_event", ""),
                "is_news_event": True,
                "summary": summary,
                "uncertainties": row.get("uncertainties", []),
                # A missing model flag is not a permission for publication.
                "requires_review": recent_raw.get("requires_review", True),
            }
            rebuilt = attach_evidence(raw, original)
            check = validate_response(rebuilt, original)
            if row.get("source_coverage", {}).get("omitted_ids"):
                check["issues"].append("source_count_exceeds_context")
            check["passed"] = not check["issues"]
            row["claims_v2"] = rebuilt["claims"]
            row["validation_v2"] = check
            row["status_v2"] = "summary_draft" if check["passed"] else "summary_review"
            counts["revalidated_summaries"] += 1
            if check["passed"]:
                counts["mechanically_clear_not_approved"] += 1
            else:
                counts["still_review"] += 1
            counts.update("issue:" + issue for issue in check["issues"])
        else:
            row["claims_v2"] = []
            row["validation_v2"] = {
                "passed": False, "issues": ["no_generated_summary"] +
                row.get("validation", {}).get("issues", [])}
            row["status_v2"] = "summary_review"
            counts["no_generated_summary"] += 1
        results.append(row)
    return {
        "stage": 4,
        "kind": "qwen_v1_rechecked_with_literal_evidence_v2",
        "source_report_created_utc": report.get("created_utc"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": VERSION,
        "reused_original_qwen_text_without_ollama": True,
        "database_writes": False,
        "publisher_called": False,
        "editorial_approval_required": True,
        "new_independent_holdout": False,
        "note": ("Only mechanical revalidation of old generated summaries; "
                 "semantic truth, actor/action bindings, grammar and event "
                 "completeness still require independent human review."),
        "metrics": {
            "rows": len(results),
            "revalidated_summaries": counts["revalidated_summaries"],
            "mechanically_clear_not_approved": counts["mechanically_clear_not_approved"],
            "still_review": counts["still_review"],
            "no_generated_summary": counts["no_generated_summary"],
            "issues": {k.removeprefix("issue:"): v for k, v in counts.items()
                       if k.startswith("issue:")},
        },
        "clusters": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve() == args.input_json.resolve():
        parser.error("Do not overwrite original Qwen evidence")
    if args.output.suffix.lower() != ".json":
        parser.error("Output must be JSON")
    report = json.loads(args.input_json.read_text(encoding="utf-8-sig"))
    corrected = check_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(corrected["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
