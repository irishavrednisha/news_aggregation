"""Replay source-cleaning and literal fact extraction on an existing Stage 4 JSON.

No Ollama/torch/Postgres dependencies. This is only evidence-preflight,
NOT a rerun of 100 Gemma generations or a factual-accuracy measurement.

PowerShell:
  .\.venv\Scripts\python.exe scripts\audit_stage4_preflight.py --input collector_test_results\stage4_abstractive_100.json --output collector_test_results\stage4_preflight_v2.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.stage4_evidence import (
    shared_boilerplate, clean_working_materials, extract_grounded_facts,
)


def replay(report: dict) -> dict:
    if report.get("stage") != 4 or not isinstance(report.get("clusters"), list):
        raise ValueError("Expected Stage 4 report with clusters[]")
    rows = report["clusters"]
    shared = shared_boilerplate([r.get("source_materials") or [] for r in rows])
    result = []
    for row in rows:
        original = row.get("source_materials") or []
        working, cleanup = clean_working_materials(original, shared)
        facts, issues = extract_grounded_facts(working)
        result.append({
            "cluster_id": row.get("cluster_id"),
            "title": original[0].get("title", "") if original else "",
            "source": original[0].get("source", "") if original else "",
            "original_status": row.get("status"),
            "working_source_chars": sum(len(str(m.get("text") or "")) for m in working),
            "removed_source_chars": sum(note["removed_chars"] for note in cleanup),
            "source_cleanup_notes": cleanup,
            "anchored_facts": facts,
            "fact_preflight_issues": issues,
            "could_attempt_gemma": bool(facts) and not issues,
        })
    issues = Counter(x for row in result for x in row["fact_preflight_issues"])
    counts = {
        "real_clusters_checked": len(result),
        "previous_gemma_drafts": sum(r.get("status") == "summary_draft" for r in rows),
        "eligible_for_gemma_generation_preflight": sum(r["could_attempt_gemma"] for r in result),
        "no_supported_news_event": issues["no_supported_news_event"],
        "insufficient_source_material": issues["insufficient_source_material"],
        "clusters_with_tail_cleanup": sum(bool(r["source_cleanup_notes"]) for r in result),
        "total_removed_source_chars": sum(r["removed_source_chars"] for r in result),
        "potentially_unsupported_event_heuristic": True,
        "new_gemma_generation_calls": 0,
        "independent_factual_accuracy": None,
        "important": ("Only source-grounding preflight was replayed. "
                      "The rewrite and semantic validator still require a fresh "
                      "local Gemma test and human verification."),
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True, "metrics": counts,
        "repeated_snippet_counts_by_source": {
            source: len(parts) for source, parts in shared.items() if parts
        }, "clusters": result,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay Stage 4 fact preflight without Ollama")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("Refusing to overwrite original audit JSON")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = replay(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print("No model inference, database writes, or publication occurred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
