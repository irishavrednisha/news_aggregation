"""One-command read-only extractive Stage 4 run.

PowerShell:
 .\.venv\Scripts\python.exe scripts\stage4_extractive_acceptance.py --input collector_test_results\holdout_manifest.json

Never changes SQL, training data, published posts or the old abstractive results.
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

from processing.qwen_abstractive_summary import MODEL, source_digest
from processing.extractive_summarizer import VERSION, summarize_extractive

DEFAULT_DIR = ROOT / "collector_test_results" / "stage4_extractive_v1_1"
FINISHED = {"ready_for_moderation", "needs_manual_review", "not_for_publication"}


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_input(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(data, dict) or data.get("stage") != 4
            or not isinstance(data.get("clusters"), list) or not data["clusters"]):
        raise ValueError("Expected Stage-4 manifest with nonempty clusters list")
    seen = set()
    for item in data["clusters"]:
        if (not isinstance(item, dict) or "cluster_id" not in item
                or not isinstance(item.get("source_materials"), list)):
            raise ValueError("Each cluster requires cluster_id and source_materials")
        cid = str(item["cluster_id"])
        if cid in seen:
            raise ValueError("Duplicate cluster ID: " + cid)
        seen.add(cid)
    return data["clusters"]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offline_tests() -> dict:
    suite = []
    for pattern in ("test_stage4_extractive.py", "test_stage4_qwen*.py"):
        try:
            run = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests",
                 "-p", pattern], cwd=ROOT, capture_output=True, text=True,
                timeout=180, encoding="utf-8", errors="replace", check=False,
            )
            suite.append({
                "suite": pattern, "passed": run.returncode == 0,
                "tail": (run.stdout + run.stderr)[-1800:],
            })
        except (OSError, subprocess.TimeoutExpired) as exc:
            suite.append({"suite": pattern, "passed": False,
                          "error": type(exc).__name__})
        if not suite[-1]["passed"]:
            break
    return {"passed": len(suite) == 2 and all(s["passed"] for s in suite),
            "suites": suite}


def preflight() -> list[str]:
    try:
        with urlopen(Request("http://127.0.0.1:11434/api/tags"), timeout=6) as response:
            data = json.load(response)
        names = {m.get("name") or m.get("model") for m in data.get("models", [])}
        if MODEL not in names:
            return ["missing_qwen_model:" + MODEL]
        return []
    except (OSError, ValueError, KeyError) as exc:
        return ["ollama_unavailable:" + type(exc).__name__]


def quality(rows: list[dict], expected: int, tests: dict, digest: str) -> dict:
    statuses = Counter(x["status"] for x in rows)
    issues = Counter(reason for x in rows for reason in x.get("review_reasons", []))
    calls = sum(x.get("model_requests", {}).get("qwen", 0) for x in rows)
    exact = all(all(
        ref["text"] == next(m for m in item["source_materials"]
                            if m["id"] == ref["source_id"])["text"][
                                ref["start_char"]:ref["end_char"]]
        for ref in item.get("extractive_sentences", [])
    ) for item in rows)
    return {
        "stage": 4, "kind": "extractive_acceptance_report",
        "pipeline_version": VERSION, "input_sha256": digest,
        "processed": len(rows), "expected": expected,
        "complete_requested_run": len(rows) == expected,
        "ready_for_moderation": statuses["ready_for_moderation"],
        "needs_manual_review": statuses["needs_manual_review"],
        "not_for_publication": statuses["not_for_publication"],
        "processing_error": statuses["processing_error"],
        "strict_source_sentence_provenance": exact,
        "extractive_summaries_obtained": sum(bool(x.get("final_summary")) for x in rows),
        "qwen_requests": calls,
        "gemma_requests": 0,
        "mean_processing_seconds": (round(sum(x.get("processing_time", 0) for x in rows)
                                          / len(rows), 2) if rows else None),
        "most_common_review_reasons": dict(issues.most_common(20)),
        "offline_tests": tests,
        "independent_human_evaluation_complete": False,
        "human_completeness": None,
        "human_readability": None,
        "human_source_accuracy": None,
        "auto_publication_allowed": False,
        "ready_for_stage5_data_contract": (
            len(rows) == expected and not statuses["processing_error"]
            and bool(tests.get("passed")) and exact),
        "limitations": [
            "Verbatim sentence provenance does not guarantee article/source accuracy.",
            "Extracts may be incoherent or omit context; human moderation required.",
            "This runner never publishes to MAX or writes PostgreSQL.",
        ],
    }


def generate(input_path: Path, out_dir: Path, *, request=None,
             run_tests: bool = True) -> dict:
    events = load_input(input_path)
    digest = file_digest(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "stage4_extractive_results.json"
    report_file = out_dir / "stage4_extractive_quality_report.json"
    tests = offline_tests() if run_tests else {
        "passed": False, "skipped_in_injected_test": True, "suites": [],
    }
    if run_tests and not tests["passed"]:
        report = quality([], len(events), tests, digest)
        atomic_write(report_file, report)
        return report
    if run_tests and request is None:
        missing = preflight()
        if missing:
            report = quality([], len(events), tests, digest)
            report["preflight_errors"] = missing
            atomic_write(report_file, report)
            return report

    saved = {}
    if results_file.exists():
        previous = json.loads(results_file.read_text(encoding="utf-8"))
        if (previous.get("input_sha256") != digest
                or previous.get("pipeline_version") != VERSION):
            raise ValueError("Output belongs to another input/version; use new --out-dir")
        saved = {str(x["cluster_id"]): x for x in previous["clusters"]}

    ordered = []
    for item in events:
        cid = item["cluster_id"]
        original = item["source_materials"]
        old = saved.get(str(cid))
        if (old and old.get("source_sha256") == source_digest(original)
                and old.get("status") in FINISHED
                and old.get("model_versions", {}).get("pipeline") == VERSION):
            row = old
        else:
            row = summarize_extractive(cid, original, request=request)
            if row["source_sha256"] != source_digest(original):
                raise AssertionError("Source digest unexpectedly changed")
        ordered.append(row)
        snapshot = {
            "stage": 4, "kind": "extractive_results",
            "pipeline_version": VERSION, "input_sha256": digest,
            "input_path": str(input_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "automatic_publication": False, "clusters": ordered,
        }
        atomic_write(results_file, snapshot)
        report = quality(ordered, len(events), tests, digest)
        atomic_write(report_file, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    options = parser.parse_args(argv)
    try:
        result = generate(options.input, options.out_dir)
    except (OSError, ValueError, KeyError) as exc:
        print("Extractive acceptance error:", exc, file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(not (result["complete_requested_run"]
                    and result["offline_tests"]["passed"]
                    and result["processing_error"] == 0))


if __name__ == "__main__":
    raise SystemExit(main())
