"""Fresh Qwen Stage-4 holdout: collect -> generate -> blind human score.

No automatic ground truth. A new sample must be >=40 independently new,
source-dated full-HTML events and include >=5 real multi-source event clusters.
Database Stage-3 membership is only candidate grouping; a human must confirm
the sources truly describe one event. No writes to PostgreSQL, MAX, master or
legacy Qwen summaries. The 41 used during debugging are excluded.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage4_independent_holdout import (
    atomic_json, run_holdout, clean_url, fingerprint, title_duplicate,
)
from scripts.score_stage4_holdout import confidence_interval

VERSION = "qwen-fresh-independent-v1"
FIELDS = ("faithful_facts", "natural_russian", "main_event_covered",
          "meaningful_news_event", "source_quality", "event_coherence")
LABELS = {"pass", "fail", "na"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _multi(row: dict) -> bool:
    materials = row.get("source_materials") or []
    return len(materials) > 1 and len(
        {clean_url(x.get("url", "")) for x in materials}) > 1 and len(
        {str(x.get("source") or "") for x in materials}) > 1


def check_manifest(manifest: dict, *, minimum: int, multi_minimum: int) -> dict:
    records = manifest.get("clusters") or []
    errors = []
    if (manifest.get("stage") != 4 or
            manifest.get("kind") != "stage4-independent-holdout-v1"):
        errors.append("wrong_manifest")
    if manifest.get("max_age_days") is None:
        errors.append("missing_freshness_window")
    if manifest.get("selection_stats", {}).get("missing_publication_dates", 1):
        errors.append("missing_publication_dates")
    if not manifest.get("previous_files"):
        errors.append("missing_previous_exclusion_files")
    if (manifest.get("minimum_sample_met") is not True or
            len(records) < minimum):
        errors.append("fewer_than_required_events")
    multis = [row for row in records if _multi(row)]
    if len(multis) < multi_minimum:
        errors.append("fewer_than_required_multi_document_events")
    urls, texts = set(), set()
    for row in records:
        if not row.get("source_materials"):
            errors.append("missing_source_materials")
            continue
        for m in row["source_materials"]:
            url, body = clean_url(m.get("url")), fingerprint(m)
            if not url or not body or url in urls or body in texts:
                errors.append("duplicate_or_missing_source_evidence")
            urls.add(url)
            texts.add(body)
    return {
        "ready_for_model_generation": not errors,
        "errors": list(dict.fromkeys(errors)),
        "unique_events": len(records), "multi_document_events": len(multis),
        "human_multi_event_identity_verified": False,
        "independent_accuracy_measured": False,
        "not_publication_approval": True,
    }


def collect(*, exclusions: list[Path], out_dir: Path, mode: str,
            target: int = 100, minimum: int = 40, multi_minimum: int = 5,
            source_limit: int = 15, source_cap: int = 15,
            max_days: int = 30, diagnostics: Path | None = None,
            resume: bool = False) -> dict:
    if len(exclusions) < 1:
        raise ValueError("At least one previous Stage 4 report is required")
    if not 0 <= multi_minimum <= minimum <= target:
        raise ValueError("Require 0 <= multi_minimum <= minimum <= target")
    if not 1 <= source_limit <= 15 or source_cap < 1 or max_days < 1:
        raise ValueError("Invalid age/source limits")
    manifest = run_holdout(
        previous_reports=exclusions, out_dir=out_dir, target=target,
        minimum=minimum, max_days=max_days, source_limit=source_limit,
        source_cap=source_cap, mode=mode, diagnostics=diagnostics,
        allow_undated=False, from_config=False,
        collect_only=True, resume=resume,
        recover_missing_rostec_dates=True,
    )
    gate = check_manifest(manifest, minimum=minimum, multi_minimum=multi_minimum)
    path = out_dir / "qwen_holdout_gate.json"
    atomic_json(path, {
        "version": VERSION, "manifest_sha256": _sha(out_dir/"holdout_manifest.json"),
        "minimum": minimum, "multi_minimum": multi_minimum, **gate,
    })
    return gate


def generate(*, out_dir: Path, resume: bool = False, request=None,
             measure_gpu: bool = True) -> dict:
    manifest_path = out_dir/"holdout_manifest.json"
    gate = _load(out_dir/"qwen_holdout_gate.json")
    if gate.get("version") != VERSION or gate.get("manifest_sha256") != _sha(manifest_path):
        raise ValueError("The frozen manifest changed after selection")
    manifest = _load(manifest_path)
    checked = check_manifest(
        manifest, minimum=gate["minimum"],
        multi_minimum=gate["multi_minimum"])
    if not checked["ready_for_model_generation"]:
        raise ValueError("The independent sample gate did not pass: " +
                         ",".join(checked["errors"]))
    from scripts.compare_stage4_qwen import run
    from processing.qwen_abstractive_summary import MODEL
    output = out_dir/"qwen_holdout.json"
    result = run(
        input_path=manifest_path, output=output, limit=len(manifest["clusters"]),
        model=MODEL, request=request, measure_gpu=measure_gpu,
        resume=resume, max_retries=1,
    )
    if (not result.get("complete_requested_run")
            or len(result["clusters"]) != len(manifest["clusters"])):
        # Do not build a partial human review packet and mistake it for full.
        return {"complete": False, "stop_reason": result.get("stopped_reason"),
                "processed": len(result.get("clusters", []))}
    write_blind_packets(manifest, result, out_dir)
    return {"complete": True, "sample_size": len(result["clusters"]),
            "multi_document_events": checked["multi_document_events"],
            "output": str(output), "blind_review_ready": True}


def write_blind_packets(manifest: dict, result: dict, directory: Path) -> None:
    """Freeze a review packet without exposing model issues, status or model ID.

    Private key contains automatic statuses. Never overwrite any human inputs
    or a pre-existing partial packet.
    """
    packet, labels, keyfile = (
        directory/"blind_qwen_review.jsonl",
        directory/"qwen_review_labels.csv",
        directory/"private_qwen_review_key.json")
    if all(x.is_file() for x in (packet, labels, keyfile)):
        previous = _load(keyfile)
        if (previous.get("manifest_sha256") != _sha(directory/"holdout_manifest.json")
                or previous.get("qwen_report_sha256") != _sha(directory/"qwen_holdout.json")):
            raise ValueError("The frozen blind labels belong to a different report")
        return
    if any(x.exists() for x in (packet, labels, keyfile)):
        raise FileExistsError("Partial blind review packet found; do not overwrite")
    originals = {str(x["cluster_id"]): x for x in manifest["clusters"]}
    entries = list(result["clusters"])
    random.Random(20261121).shuffle(entries)
    key = {}
    with packet.open("w", encoding="utf-8") as fh, labels.open(
            "w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.writer(cf, delimiter=";")
        writer.writerow(["review_id", *FIELDS, "error_details"])
        for i, row in enumerate(entries, 1):
            rid = f"N{i:03d}"
            cid = str(row["cluster_id"])
            docs = originals[cid]["source_materials"]
            key[rid] = {
                "cluster_id": row["cluster_id"],
                "automatic_status": row.get("status"),
                "source_sha256": row.get("source_sha256"),
                "multi_document": _multi(originals[cid]),
            }
            fh.write(json.dumps({
                "review_id": rid,
                "title": docs[0].get("title", ""),
                "source_documents": [{k: m.get(k) for k in (
                    "source", "url", "title", "text", "published_at"
                )} for m in docs],
                "summary": row.get("qwen_summary") or "",
                "multi_document": _multi(originals[cid]),
                "review_instructions": (
                    "Read ALL source documents. Check EVERY sentence and event "
                    "stage, numerical indicator, unit and actor. For multi-document "
                    "cases verify all sources actually report the same event. "
                    "Never label sources or draft automatically from the model. "
                    "Empty summary -> na for all dimensions; single-source -> "
                    "event_coherence=na; missing evidence -> source_quality=fail."),
            }, ensure_ascii=False) + "\n")
            writer.writerow([rid, "", "", "", "", "", "", ""])
    atomic_json(keyfile, {
        "version": VERSION, "manifest_sha256": _sha(directory/"holdout_manifest.json"),
        "qwen_report_sha256": _sha(directory/"qwen_holdout.json"),
        "mapping": key,
    })


def score(*, out_dir: Path, minimum: int | None = None,
          multi_minimum: int | None = None) -> dict:
    manifest = _load(out_dir/"holdout_manifest.json")
    report = _load(out_dir/"qwen_holdout.json")
    secret = _load(out_dir/"private_qwen_review_key.json")
    if (secret.get("manifest_sha256") != _sha(out_dir/"holdout_manifest.json")
            or secret.get("qwen_report_sha256") != _sha(out_dir/"qwen_holdout.json")):
        raise ValueError("Frozen manifest / report differs from review key")
    with (out_dir/"qwen_review_labels.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle, delimiter=";"))
    mapping = secret["mapping"]
    if (len(labels) != len(mapping)
            or len({r.get("review_id") for r in labels}) != len(mapping)
            or {r.get("review_id") for r in labels} != set(mapping)):
        raise ValueError("Missing, duplicated or foreign review IDs")
    rows = {str(r["cluster_id"]): r for r in report["clusters"]}
    if set(rows) != {str(r["cluster_id"]) for r in manifest["clusters"]}:
        raise ValueError("Generated rows differ from held-out event manifest")
    completed, missing, positives, multi_positives = [], [], 0, 0
    drafts_ok, drafts_bad, reviews_good = [], [], []
    errors = {}
    for label in labels:
        rid = label["review_id"]
        meta = mapping[rid]
        row = rows[str(meta["cluster_id"])]
        if (meta["automatic_status"] != row["status"]
                or meta["source_sha256"] != row["source_sha256"]):
            raise ValueError("Source or automatic status changed during review")
        values = {field: (label.get(field) or "").strip().lower()
                  for field in FIELDS}
        if any(v not in LABELS | {""} for v in values.values()):
            raise ValueError("Only pass/fail/na are valid")
        if not (row.get("qwen_summary") or "").strip():
            if any(v not in ("", "na") for v in values.values()):
                raise ValueError("Absent summary must be na/blank: " + rid)
            continue
        if (any(not values[field] or values[field] == "na"
                for field in FIELDS if field != "event_coherence")
                or (values["event_coherence"] not in ("pass", "fail")
                    if meta["multi_document"] else values["event_coherence"] != "na")):
            missing.append(rid)
            continue
        valid = all(values[f] == "pass"
                    for f in FIELDS if f != "event_coherence") and (
                        values["event_coherence"] == "pass"
                        if meta["multi_document"] else True)
        if values["meaningful_news_event"] == "pass" and values["source_quality"] == "pass":
            positives += 1
            if meta["multi_document"] and values["event_coherence"] == "pass":
                multi_positives += 1
        completed.append(rid)
        errors[rid] = (label.get("error_details") or "")[:1000]
        if row["status"] == "summary_draft":
            (drafts_ok if valid else drafts_bad).append(rid)
        if row["status"] == "summary_review" and valid:
            reviews_good.append(rid)
    n = minimum if minimum is not None else _load(
        out_dir/"qwen_holdout_gate.json")["minimum"]
    m = multi_minimum if multi_minimum is not None else _load(
        out_dir/"qwen_holdout_gate.json")["multi_minimum"]
    n_drafts = sum(r["status"] == "summary_draft" for r in rows.values())
    n_generated = sum(bool(r.get("qwen_summary")) for r in rows.values())
    audit_ready = (not missing and len(completed) == n_generated
                   and positives >= n and multi_positives >= m)
    metrics = {
        "stage": 4, "kind": VERSION + "-human-score",
        "sample_size": len(rows), "generated_summaries": n_generated,
        "fully_labeled_summaries": len(completed), "unlabeled_ids": missing,
        "manually_verified_industrial_events": positives,
        "manually_verified_multi_document_events": multi_positives,
        "minimum_required_positive_events": n,
        "minimum_required_multi_document_events": m,
        "drafts_human_pass": len(drafts_ok),
        "drafts_human_fail": len(drafts_bad),
        "false_accepted_review_ids": drafts_bad,
        "correct_summaries_requiring_extra_review": reviews_good,
        "draft_precision_95_wilson": confidence_interval(
            len(drafts_ok), len(drafts_ok)+len(drafts_bad)),
        "all_drafts_human_labeled": len(drafts_ok)+len(drafts_bad) == n_drafts,
        "all_generated_human_labeled": len(completed) == n_generated,
        "independent_evaluation_complete": audit_ready,
        "human_reviewer_identity_verified": False,
        "not_publication_approval": True,
        "claim": ("Review values come ONLY from manually completed CSV. "
                  "Even high draft precision does not authorize publishing. "
                  "Missing labels or positive/multi-source thresholds block finalization."),
        "error_notes": errors,
    }
    atomic_json(out_dir/"qwen_human_score.json", metrics)
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--exclude", type=Path, action="append", required=True)
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--mode", choices=("both", "live", "database", "diagnostics"),
                   default="both")
    c.add_argument("--diagnostics-json", type=Path)
    c.add_argument("--target", type=int, default=100)
    c.add_argument("--minimum", type=int, default=40)
    c.add_argument("--min-multisource", type=int, default=5)
    c.add_argument("--max-age-days", type=int, default=30)
    c.add_argument("--limit-per-source", type=int, default=15)
    c.add_argument("--max-per-source", type=int, default=15)
    c.add_argument("--resume", action="store_true")
    g = sub.add_parser("generate")
    g.add_argument("--out-dir", type=Path, required=True)
    g.add_argument("--resume", action="store_true")
    g.add_argument("--no-gpu-snapshot", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            result = collect(exclusions=args.exclude, out_dir=args.out_dir,
                             mode=args.mode, target=args.target,
                             minimum=args.minimum, multi_minimum=args.min_multisource,
                             max_days=args.max_age_days,
                             source_limit=args.limit_per_source,
                             source_cap=args.max_per_source,
                             diagnostics=args.diagnostics_json, resume=args.resume)
            code = 0 if result["ready_for_model_generation"] else 2
        elif args.command == "generate":
            result = generate(out_dir=args.out_dir, resume=args.resume,
                              measure_gpu=not args.no_gpu_snapshot)
            code = 0 if result.get("complete") else 4
        else:
            result = score(out_dir=args.out_dir)
            code = 0 if result["independent_evaluation_complete"] else 3
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
