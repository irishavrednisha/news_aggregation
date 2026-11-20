"""Opt-in local Gemma review of saved Stage-4 Qwen summaries.

One command, atomic checkpoints and --resume. No Qwen regeneration, SQL,
MAX or external paid API. The result cannot approve publication.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.qwen_semantic_reviewer import REVIEW_MODEL, review_cluster


def _source_rows(document: dict) -> list[dict]:
    if document.get("kind") == "qwen_one_run_offline_diagnostics":
        document = document.get("revalidation") or {}
    if document.get("stage") != 4 or not isinstance(document.get("clusters"), list):
        raise ValueError("Input must contain saved Stage-4 Qwen clusters")
    return document["clusters"]


def _signature(row: dict) -> str:
    payload = [row.get("cluster_id"), row.get("source_sha256"),
               row.get("qwen_summary"), REVIEW_MODEL]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run(input_path: Path, output_path: Path, *, limit: int | None = None,
        resume: bool = False, request=None) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must differ")
    if output_path.suffix.lower() != ".json":
        raise ValueError("Output must be .json")
    original = input_path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    rows = _source_rows(json.loads(original.decode("utf-8-sig")))
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    chosen = rows[:limit] if limit is not None else rows
    prior = {}
    if resume and output_path.exists():
        earlier = json.loads(output_path.read_text(encoding="utf-8-sig"))
        if earlier.get("input_sha256") != digest:
            raise ValueError("Input file changed; refusing unsafe resume")
        prior = {str(x.get("signature")): x for x in earlier.get("results", [])}
    report = {
        "stage": 4, "kind": "stage4_local_semantic_advisory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": digest,
        "model": REVIEW_MODEL, "read_only": True, "database_writes": False,
        "publisher_called": False, "auto_publication": False,
        "human_approval_required": True,
        "editorial_accuracy_measured": False,
        "results": [], "complete": False, "stopped_reason": None,
    }
    for row in chosen:
        sig = _signature(row)
        cached = prior.get(sig)
        if cached and cached.get("status") in ("advisory_no_issues_detected",
                                               "review_required") and not any(
                x.startswith("reviewer_unavailable") for x in cached.get("issues", [])):
            review = dict(cached)
        else:
            review = review_cluster(row, request=request)
            review["signature"] = sig
        report["results"].append(review)
        _save(output_path, report)
        if any(i.startswith("reviewer_unavailable:") for i in review["issues"]):
            report["stopped_reason"] = "local_reviewer_unavailable"
            _save(output_path, report)
            break
    else:
        report["complete"] = True
    report["metrics"] = {
        "requested": len(chosen),
        "processed": len(report["results"]),
        "review_required": sum(x["status"] == "review_required"
                               for x in report["results"]),
        "advisory_no_issues": sum(x["status"] == "advisory_no_issues_detected"
                                  for x in report["results"]),
        "issues": dict(Counter(
            reason for r in report["results"] for reason in r["issues"])),
    }
    _save(output_path, report)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    try:
        result = run(a.input_json, a.output, limit=a.limit, resume=a.resume)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    print(json.dumps({
        "complete": result["complete"],
        "stopped_reason": result["stopped_reason"],
        "metrics": result["metrics"],
        "output": str(a.output),
    }, ensure_ascii=False, indent=2))
    return 0 if result["complete"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
