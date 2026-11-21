"""One-command, resumable, read-only Stage-4 Qwen+Gemma acceptance.

Example:
 .\.venv\Scripts\python.exe scripts\stage4_final_acceptance.py --input collector_test_results\holdout_manifest.json

No SQL, HTTP collection, MAX publishing, training or automatic fact approval.
Requires local Ollama with qwen3.5:9b-q4_K_M and gemma3:4b.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.qwen_abstractive_summary import source_digest
from processing.qwen_gemma_final import GEMMA_MODEL, QWEN_MODEL, VERSION, summarize_final

DEFAULT_DIR = ROOT / "collector_test_results" / "stage4_qwen_gemma_final"
FINISHED = {"ready_for_moderation", "needs_manual_review", "not_for_publication"}


def atomic_write(path: Path, payload: object) -> None:
    """Write a complete JSON snapshot before atomically replacing old content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_input(path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(report, dict) or report.get("stage") != 4
            or not isinstance(report.get("clusters"), list)
            or not report["clusters"]):
        raise ValueError("Input must be Stage 4 JSON with nonempty clusters")
    seen = set()
    for row in report["clusters"]:
        if (not isinstance(row, dict) or "cluster_id" not in row or
                not isinstance(row.get("source_materials"), list)):
            raise ValueError("Each event requires cluster_id and source_materials")
        key = str(row["cluster_id"])
        if key in seen:
            raise ValueError("Duplicate cluster_id in input: " + key)
        seen.add(key)
    return report, report["clusters"]


def quality(rows: list[dict], total: int, tests: dict, input_sha: str) -> dict:
    counts = Counter(row["status"] for row in rows)
    selected = Counter(row.get("selected_version", "none") for row in rows)
    runtime = [row.get("processing_time", 0) for row in rows]
    issues = Counter(issue for row in rows
                     for issue in row.get("validation_issues", []))
    return {
        "stage": 4, "kind": "qwen_gemma_final_acceptance_report",
        "pipeline_version": VERSION, "input_sha256": input_sha,
        "processed": len(rows), "expected": total,
        "complete_requested_run": len(rows) == total,
        "ready_for_moderation": counts["ready_for_moderation"],
        "needs_manual_review": counts["needs_manual_review"],
        "not_for_publication": counts["not_for_publication"],
        "processing_error": counts["processing_error"],
        "selected_versions": dict(selected),
        "model_requests": {
            "qwen": sum(x.get("model_requests", {}).get("qwen", 0) for x in rows),
            "gemma_fact_review": sum(x.get("model_requests", {}).get("gemma_fact_review", 0) for x in rows),
            "gemma_edit": sum(x.get("model_requests", {}).get("gemma_edit", 0) for x in rows),
        },
        "total_processing_seconds": round(sum(runtime), 2),
        "mean_processing_seconds": round(sum(runtime) / len(runtime), 2) if runtime else None,
        "most_common_guard_issues": dict(issues.most_common(20)),
        "offline_tests": tests,
        "human_factual_accuracy": None,
        "human_language_quality": None,
        "human_event_completeness": None,
        "human_gemma_introduced_errors": None,
        "human_multisource_coherence": None,
        "independent_human_evaluation_complete": False,
        "model_calls_attempted": any(row.get("model_requests", {}).get("qwen", 0) for row in rows),
        "qwen_drafts_obtained": sum(bool(row.get("qwen_summary")) for row in rows),
        "gemma_reviews_obtained": sum(bool(row.get("gemma_fact_review")) for row in rows),
        "gemma_edits_obtained": sum(bool(row.get("gemma_edited_summary")) for row in rows),
        "auto_publication_allowed": False,
        "ready_for_stage5_data_contract": (
            len(rows) == total and not counts["processing_error"]
            and bool(tests.get("passed"))
        ),
        "limitations": [
            "Automated guards and Gemma labels do not establish factual truth.",
            "Manual blind labels are required before claiming factual or stylistic accuracy.",
            "ready_for_moderation is not approval for MAX publishing.",
            "Independent sample provenance must be verified outside this runner.",
        ],
    }


def ollama_preflight() -> list[str]:
    """Avoid making dozens of slow failing requests when local models are absent."""
    try:
        req = Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urlopen(req, timeout=6) as response:
            data = json.load(response)
        names = {str(m.get("name") or m.get("model") or "") for m in data.get("models", [])}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return ["ollama_not_available:" + type(exc).__name__]
    return ["missing_ollama_model:" + model for model in (QWEN_MODEL, GEMMA_MODEL)
            if model not in names]


def test_suite() -> dict:
    patterns = ("test_stage4_qwen*.py",)
    records = []
    for pattern in patterns:
        cmd = [sys.executable, "-m", "unittest", "discover",
               "-s", "tests", "-p", pattern]
        try:
            proc = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=180,
                encoding="utf-8", errors="replace", check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            records.append({"pattern": pattern, "passed": False,
                            "error": type(exc).__name__})
            break
        records.append({"pattern": pattern, "passed": proc.returncode == 0,
                        "exit_code": proc.returncode,
                        "output_tail": (proc.stdout + proc.stderr)[-2500:]})
        if proc.returncode:
            break
    return {"passed": len(records) == len(patterns) and all(
        x["passed"] for x in records), "suites": records}


def generate(input_path: Path, out_dir: Path, *, request=None,
             run_tests: bool = True) -> dict:
    """Read-only input; every completed event is committed to atomic JSON."""
    _, events = load_input(input_path)
    input_digest = sha256(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "stage4_final_results.json"
    metrics_path = out_dir / "stage4_quality_report.json"
    tests = test_suite() if run_tests else {
        "passed": False, "suites": [], "skipped_in_injected_test": True,
    }
    if run_tests and not tests["passed"]:
        report = quality([], len(events), tests, input_digest)
        atomic_write(metrics_path, report)
        return report
    if run_tests and request is None:
        errors = ollama_preflight()
        if errors:
            report = quality([], len(events), tests, input_digest)
            report["preflight_errors"] = errors
            atomic_write(metrics_path, report)
            return report

    previous = {}
    if data_path.exists():
        saved = json.loads(data_path.read_text(encoding="utf-8"))
        if saved.get("input_sha256") != input_digest:
            raise ValueError("Input changed; choose a new out-dir, frozen output preserved")
        previous = {str(x["cluster_id"]): x for x in saved["clusters"]}

    ordered = []
    for row in events:
        cid = row["cluster_id"]
        materials = row["source_materials"]
        digest = source_digest(materials)
        cached = previous.get(str(cid))
        if (cached and cached.get("source_sha256") == digest
                and cached.get("status") in FINISHED
                and cached.get("model_versions", {}).get("pipeline") == VERSION):
            result = cached
        else:
            result = summarize_final(cid, materials, request=request)
            if result["source_sha256"] != digest:
                raise AssertionError("Source fingerprint changed during processing")
        ordered.append(result)
        snapshot = {
            "stage": 4, "kind": "qwen_gemma_final_results",
            "pipeline_version": VERSION, "input_sha256": input_digest,
            "input_path": str(input_path), "automatic_publication": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "clusters": ordered,
        }
        # On resume, retain all previously completed rows whose input is unchanged.
        # Interrupted current event is regenerated; last complete JSON stays valid.
        atomic_write(data_path, snapshot)
        report = quality(ordered, len(events), tests, input_digest)
        atomic_write(metrics_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Stage 4 manifest with source_materials")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    try:
        report = generate(args.input, args.out_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print("Stage 4 acceptance error:", str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # A failed model call is reported, not silently presented as accepted.
    return 0 if (report["complete_requested_run"] and
                 report["offline_tests"]["passed"] and
                 report["processing_error"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
