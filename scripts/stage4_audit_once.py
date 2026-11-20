"""One offline Stage-4 audit: regression tests plus immutable Qwen report replay.

This wrapper does not call Ollama, PostgreSQL or MAX and never edits input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.revalidate_stage4_qwen import check_report

TEST_GROUPS = (
    "test_stage4_qwen*.py",
    "test_stage4_independent_holdout.py",
)


def run_audit(input_json: Path, output: Path, *, skip_tests: bool = False) -> dict:
    """Read inputs once, write a new atomic JSON report, including failures."""
    input_json = input_json.resolve()
    output = output.resolve()
    if input_json == output:
        raise ValueError("Refusing to overwrite input")
    if output.suffix.lower() != ".json":
        raise ValueError("Output must end in .json")
    source = json.loads(input_json.read_text(encoding="utf-8-sig"))
    tests = []
    if not skip_tests:
        for pattern in TEST_GROUPS:
            command = [sys.executable, "-m", "unittest", "discover",
                       "-s", "tests", "-p", pattern, "-v"]
            proc = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True,
                timeout=180, check=False)
            tests.append({
                "pattern": pattern, "return_code": proc.returncode,
                "passed": proc.returncode == 0,
                "stdout": proc.stdout[-40000:],
                "stderr": proc.stderr[-40000:],
            })
    report = {
        "stage": 4, "kind": "qwen_one_run_offline_diagnostics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_json), "source_report_is_immutable": True,
        "ollama_called": False, "database_writes": False, "max_publications": 0,
        "editorial_approval_required": True,
        "tests": tests,
        "all_executed_tests_passed": all(x["passed"] for x in tests),
        "revalidation": None,
        "revalidation_error": None,
        "independent_accuracy_measured": False,
    }
    try:
        report["revalidation"] = check_report(source)
    except Exception as exc:
        # Save the diagnostics as a failed report; do not fake revalidation.
        report["revalidation_error"] = type(exc).__name__ + ": " + str(exc)
    report["successful"] = (report["all_executed_tests_passed"]
                             and report["revalidation_error"] is None)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", required=True, type=Path,
                   help="Existing qwen_vs_gemma_41_mixed.json on your PC")
    p.add_argument("--output", type=Path,
                   default=Path("collector_test_results/stage4_relation_audit_full.json"))
    p.add_argument("--skip-tests", action="store_true",
                   help="Replay only, with explicit test skip recorded")
    args = p.parse_args(argv)
    try:
        result = run_audit(args.input_json, args.output, skip_tests=args.skip_tests)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        p.error(str(exc))
    status = {
        "output_json": str(args.output),
        "successful": result["successful"],
        "tests": [{"pattern": t["pattern"], "passed": t["passed"]}
                  for t in result["tests"]],
        "revalidation_error": result["revalidation_error"],
        "metrics": ((result["revalidation"] or {}).get("metrics")),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if result["successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
