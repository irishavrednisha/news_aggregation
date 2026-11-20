"""Experimental claimwise local Gemma second reader with per-cluster checkpoints.

Reads existing Stage4 Qwen JSON, never regenerates, writes DB, or publishes.
This script does not treat model "supported" as proof. All old batch reports
remain untouched. By default, do not cache malformed verdicts on --resume.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage4_semantic_audit import _save, _source_rows
from processing.qwen_semantic_claimwise import (
    VERSION, review_cluster_claimwise,
)

MODEL = "gemma3:4b"


def _signature(row: dict) -> str:
    return hashlib.sha256(json.dumps(
        [VERSION, row.get("cluster_id"), row.get("source_sha256"),
         row.get("qwen_summary"), MODEL], ensure_ascii=False,
        sort_keys=True).encode("utf-8")).hexdigest()


def _cacheable(item: dict) -> bool:
    if item.get("version") != VERSION:
        return False
    problems = item.get("issues", [])
    return not any(
        reason.startswith("reviewer_unavailable:")
        or reason.endswith("_invalid_fact_schema")
        or reason.endswith("_invalid_fact_verdict")
        or reason.endswith("_invalid_fact_quote_indices")
        or reason.endswith("_empty_fact_reason")
        or reason.endswith("_supported_without_quoted_evidence")
        or reason.startswith("language_invalid_")
        or reason.startswith("language_language_")
        for reason in problems
    )


def run(input_path: Path, output_path: Path, *, limit: int | None = None,
        resume: bool = False, request=None) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite original Qwen report")
    if output_path.suffix.lower() != ".json":
        raise ValueError("Output must be .json")
    raw = input_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = _source_rows(json.loads(raw.decode("utf-8-sig")))
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    chosen = rows if limit is None else rows[:limit]
    prior = {}
    if resume and output_path.exists():
        old = json.loads(output_path.read_text(encoding="utf-8-sig"))
        if old.get("input_sha256") != digest or old.get("version") != VERSION:
            raise ValueError("Different input/version; refusing unsafe resume")
        prior = {x.get("signature"): x for x in old.get("results", [])}
    report = {
        "stage": 4, "kind": "stage4_local_semantic_claimwise_advisory",
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": digest, "model": MODEL, "local_only": True,
        "read_only": True, "database_writes": False, "publisher_called": False,
        "auto_publication": False, "human_approval_required": True,
        "editorial_accuracy_measured": False,
        "results": [], "complete": False, "stopped_reason": None,
    }
    for row in chosen:
        sig = _signature(row)
        cached = prior.get(sig)
        if cached and _cacheable(cached):
            result = dict(cached)
        else:
            result = review_cluster_claimwise(row, request=request)
            result["signature"] = sig
        report["results"].append(result)
        _save(output_path, report)
        if any(x.startswith("reviewer_unavailable:") for x in result["issues"]):
            report["stopped_reason"] = "local_reviewer_unavailable"
            break
    else:
        report["complete"] = True
    allissues = Counter(reason for item in report["results"]
                        for reason in item.get("issues", []))
    report["metrics"] = {
        "requested": len(chosen), "processed": len(report["results"]),
        "model_calls": sum(x.get("model_calls", 0) for x in report["results"]),
        "advisory_no_issues": sum(
            x.get("status") == "advisory_no_issues_detected"
            for x in report["results"]),
        "review_required": sum(x.get("status") == "review_required"
                               for x in report["results"]),
        "valid_model_claim_verdicts": sum(
            sum(1 for c in (x.get("answer") or {}).get("claims", [])
                if not c.get("not_model_generated"))
            for x in report["results"]),
        "supported": sum(
            sum(c.get("verdict") == "supported" for c in
                (x.get("answer") or {}).get("claims", [])
                if not c.get("not_model_generated"))
            for x in report["results"]),
        "contradicted": sum(
            sum(c.get("verdict") == "contradicted" for c in
                (x.get("answer") or {}).get("claims", []))
            for x in report["results"]),
        "insufficient": sum(
            sum(c.get("verdict") == "insufficient" for c in
                (x.get("answer") or {}).get("claims", []))
            for x in report["results"]),
        "issues": dict(allissues),
    }
    report["schema_all_valid"] = not any(
        any(token in x for token in (
            "schema", "quote_indices", "empty_fact_reason",
            "invalid_fact_verdict", "language_flag_without_example",
            "supported_without_quoted_evidence"))
        for x in allissues
    )
    _save(output_path, report)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    try:
        out = run(args.input_json, args.output, limit=args.limit,
                  resume=args.resume)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    print(json.dumps({
        "complete": out["complete"],
        "stopped_reason": out["stopped_reason"],
        "schema_all_valid": out["schema_all_valid"],
        "metrics": out["metrics"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0 if out["complete"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
