"""Replay saved Stage 4 sources through local Qwen and compare with Gemma.

Read-only: no collector, DB connection, migrations, or MAX publisher.
Compatible with gemma_holdout.json, stage4_abstractive_v5_100.json and
holdout_manifest.json (as long as they contain clusters/source_materials).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.qwen_abstractive_summary import (
    MODEL, VERSION, canonical_materials, source_digest, summarize_cluster,
)
from processing.stage4_evidence import clean_working_materials, shared_boilerplate


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict) or report.get("stage") != 4:
        raise ValueError(f"Not a Stage 4 JSON: {path}")
    rows = report.get("clusters")
    if not isinstance(rows, list) or any(
            not isinstance(r, dict) or not isinstance(r.get("source_materials"), list)
            for r in rows):
        raise ValueError("Expected clusters with preserved source_materials")
    return report


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def baseline_index(report: dict) -> dict[str, dict]:
    return {str(row["cluster_id"]): row for row in report["clusters"]}


def match_gemma(row: dict, baseline: dict[str, dict]) -> tuple[str, str]:
    """Never present two summaries as paired when source snapshots differ."""
    candidate = baseline.get(str(row["cluster_id"]))
    if candidate is None:
        return "", "baseline_not_found"
    source_hash = source_digest(row["source_materials"])
    if source_hash != source_digest(candidate["source_materials"]):
        return "", "baseline_source_mismatch"
    summary = str(candidate.get("gemma_summary") or "")
    return summary, "matched" if summary else "baseline_summary_missing"


def resource_snapshot() -> dict:
    """Best-effort measurements, not fabricated VRAM peak utilization."""
    output = {}
    for name, cmd in (
        ("ollama_ps", ["ollama", "ps"]),
        ("nvidia_smi", ["nvidia-smi", "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits"]),
    ):
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=8, check=False, encoding="utf-8",
                                       errors="replace")
            output[name] = {"returncode": completed.returncode,
                            "stdout": completed.stdout.strip()[:1500],
                            "stderr": completed.stderr.strip()[:250]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            output[name] = {"error": type(exc).__name__}
    output["note"] = "Snapshots before/after, NOT a peak-VRAM measurement."
    return output


def evaluate(rows: list[dict]) -> dict:
    status = Counter(r["status"] for r in rows)
    paired = [r for r in rows if r.get("baseline_match") == "matched"]
    elapsed = [a["elapsed_seconds"] for r in rows
               for a in r.get("attempts", []) if "elapsed_seconds" in a]
    tps = [a.get("ollama_metrics", {}).get("tokens_per_second")
           for r in rows for a in r.get("attempts", [])
           if isinstance(a.get("ollama_metrics"), dict)
           and a["ollama_metrics"].get("tokens_per_second") is not None]
    issues = Counter(issue for r in rows
                     for issue in r.get("validation", {}).get("issues", []))
    return {
        "processed": len(rows), "paired_same_source": len(paired),
        "qwen_generated": sum(bool(r.get("qwen_summary")) for r in rows),
        "gemma_generated_in_paired": sum(bool(r.get("gemma_summary")) for r in paired),
        "qwen_mechanical_drafts": status["summary_draft"],
        "qwen_review": status["summary_review"],
        "mean_ollama_request_seconds": round(sum(elapsed)/len(elapsed), 3) if elapsed else None,
        "mean_generation_tokens_per_second": round(sum(tps)/len(tps), 2) if tps else None,
        "issues": dict(issues.most_common()), "human_factual_accuracy": None,
        "human_russian_quality": None, "human_main_event_coverage": None,
        "warning": ("Draft status and quote matching are NOT factual truth labels. "
                    "All results require human editorial verification."),
    }


def make_report(rows: list[dict], *, input_sha: str, baseline_sha: str | None,
                model: str, options: dict, max_input_chars: int,
                created_utc: str, source_path: str) -> dict:
    return {
        "stage": 4, "kind": "qwen_vs_gemma_diagnostic",
        "pipeline_version": VERSION, "model": model,
        "model_parameters": options, "max_input_chars": max_input_chars,
        "created_utc": created_utc, "source_file": source_path,
        "input_sha256": input_sha, "baseline_sha256": baseline_sha,
        "read_only": True, "database_writes": False, "max_publications": 0,
        "training_started": False, "automatic_publication": False,
        "editorial_approval_required": True,
        "independent_holdout_claimed": False,
        "metrics": evaluate(rows), "clusters": rows,
    }


def emit_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["cluster_id", "source_urls", "source_sha256",
                         "gemma_summary", "gemma_status", "qwen_summary",
                         "qwen_status", "baseline_match", "review_issues",
                         "total_seconds"])
        for row in rows:
            writer.writerow([
                row["cluster_id"],
                " | ".join(m.get("url", "") for m in row["source_materials"]),
                row["source_sha256"], row.get("gemma_summary", ""),
                row.get("gemma_status", ""), row.get("qwen_summary", ""),
                row["status"], row.get("baseline_match", ""),
                " | ".join(row.get("validation", {}).get("issues", [])),
                sum(a.get("elapsed_seconds", 0) for a in row.get("attempts", [])),
            ])


def emit_markdown(rows: list[dict], path: Path) -> None:
    parts = [
        "# Qwen3.5 9B vs Gemma 3 4B — диагностическое сравнение",
        "",
        "Сопоставление по исходным хешам. Даже прошедшая механическую проверку "
        "сводка НЕ утверждена редактором. Результат НЕ является слепым holdout.",
        "",
    ]
    for row in rows:
        title = row["source_materials"][0].get("title", "") if row["source_materials"] else ""
        parts.extend([
            f"## Cluster {row['cluster_id']}: {title}",
            "",
            "**Источники:** " + ", ".join(m.get("url", "") for m in row["source_materials"]),
            "",
            "**Gemma:** " + (row.get("gemma_summary") or "Нет результата на идентичных исходниках"),
            "",
            "**Qwen:** " + (row.get("qwen_summary") or "Нет результата"),
            "",
            "**Статус Qwen:** " + row["status"] + " (не разрешает публикацию)",
            "",
            "**Проблемы:** " + (", ".join(row.get("validation", {}).get("issues", [])) or "механически не выявлены"),
            "",
        ])
    path.write_text("\n".join(parts), encoding="utf-8")


def emit_review_template(rows: list[dict], path: Path) -> None:
    if path.exists():
        return  # Do not erase human labels on --resume.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["cluster_id", "source_sha256", "faithful_facts",
                         "natural_russian", "main_event_covered",
                         "quotes_represent_claims", "source_quality", "notes"])
        for row in rows:
            writer.writerow([row["cluster_id"], row["source_sha256"],
                             "", "", "", "", "", ""])


def run(input_path: Path, output: Path, *, baseline_path: Path | None = None,
        model: str = MODEL, options: dict | None = None, max_input_chars: int = 6000,
        limit: int = 91, resume: bool = False, max_retries: int = 1,
        request=None, measure_gpu: bool = True) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    for forbidden in (input_path, baseline_path):
        if forbidden is not None and output.resolve() == forbidden.resolve():
            raise ValueError("Output cannot overwrite preserved sources or Gemma results")
    source = load_report(input_path)
    baseline = load_report(baseline_path) if baseline_path else source
    baseline_rows = baseline_index(baseline)
    source_sha = file_sha(input_path)
    baseline_sha = file_sha(baseline_path) if baseline_path else None
    opts = {"temperature": 0, "num_ctx": 4096, "num_predict": 1400}
    opts.update(options or {})
    processed, started = {}, datetime.now(timezone.utc).isoformat()
    if resume:
        old = json.loads(output.read_text(encoding="utf-8"))
        if (old.get("pipeline_version") != VERSION or old.get("model") != model
                or old.get("input_sha256") != source_sha
                or old.get("baseline_sha256") != baseline_sha
                or old.get("model_parameters") != opts
                or old.get("max_input_chars") != max_input_chars):
            raise ValueError("Resume configuration or input changed")
        started = old["created_utc"]
        processed = {str(r["cluster_id"]): r for r in old["clusters"]}
    selected = source["clusters"][:limit]
    if len({str(x.get("cluster_id")) for x in selected}) != len(selected):
        raise ValueError("Duplicate cluster ID in saved source manifest")
    shared = shared_boilerplate([r["source_materials"] for r in selected])
    rows = []
    hardware_before = resource_snapshot() if measure_gpu else {}
    for i, raw in enumerate(selected, 1):
        cluster_id, materials = raw["cluster_id"], raw["source_materials"]
        digest = source_digest(materials)
        previous = processed.get(str(cluster_id))
        previous_issues = (previous or {}).get("validation", {}).get("issues", [])
        completed = bool(previous and (
            previous.get("qwen_summary")
            or "model_classified_no_news_event" in previous_issues
            or "insufficient_source_material" in previous_issues))
        if previous and previous.get("source_sha256") == digest and completed:
            row = previous
            print(f"[{i}/{len(selected)}] #{cluster_id} reused", flush=True)
        else:
            working, cleanup_notes = clean_working_materials(materials, shared)
            row = summarize_cluster(
                cluster_id, materials, working_materials=working,
                model=model, options=opts, max_input_chars=max_input_chars,
                request=request, max_retries=max_retries)
            row["source_cleanup_notes"] = cleanup_notes
            summary, match = match_gemma(raw, baseline_rows)
            row["gemma_summary"] = summary
            row["baseline_match"] = match
            row["gemma_status"] = baseline_rows.get(str(cluster_id), {}).get(
                "status", "") if match == "matched" else ""
            print(f"[{i}/{len(selected)}] #{cluster_id} {row['status']} "
                  f"issues={row['validation'].get('issues', [])}", flush=True)
        rows.append(row)
        checkpoint(output, make_report(
            rows, input_sha=source_sha, baseline_sha=baseline_sha, model=model,
            options=opts, max_input_chars=max_input_chars, created_utc=started,
            source_path=str(input_path)))
    result = make_report(
        rows, input_sha=source_sha, baseline_sha=baseline_sha, model=model,
        options=opts, max_input_chars=max_input_chars, created_utc=started,
        source_path=str(input_path))
    result["hardware_before"] = hardware_before
    result["hardware_after"] = resource_snapshot() if measure_gpu else {}
    checkpoint(output, result)
    emit_csv(rows, output.with_suffix(".csv"))
    emit_markdown(rows, output.with_suffix(".md"))
    emit_review_template(rows, output.with_name(output.stem + "_human_review.csv"))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare local Qwen with saved Gemma snapshots, read-only")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--gemma-json", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "collector_test_results" /
                        "stage4_qwen_vs_gemma_91.json")
    parser.add_argument("--limit", type=int, default=91)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-input-chars", type=int, default=6000)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=1400)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-gpu-snapshot", action="store_true")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".json" or args.max_input_chars < 2000:
        parser.error("Output must end in .json and max-input-chars must be >= 2000")
    if not args.model.startswith("qwen3.5:"):
        parser.error("Only a local qwen3.5: model is supported")
    from processing.qwen_abstractive_summary import LocalOllama
    result = run(args.input_json, args.output, baseline_path=args.gemma_json,
                 model=args.model, options={
                     "temperature": 0, "num_ctx": args.num_ctx,
                     "num_predict": args.num_predict},
                 max_input_chars=args.max_input_chars, limit=args.limit,
                 max_retries=args.max_retries, resume=args.resume,
                 request=LocalOllama(timeout=args.timeout),
                 measure_gpu=not args.no_gpu_snapshot)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output.resolve()}")
    return 0 if result["metrics"]["qwen_generated"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
