"""Stage 4 local audit: 100 real PostgreSQL event clusters, no publication.

Examples (PowerShell):
  .\.venv\Scripts\python.exe scripts\run_stage4_summaries.py --limit 100
  .\.venv\Scripts\python.exe scripts\run_stage4_summaries.py --limit 3 --min-clusters 1
  .\.venv\Scripts\python.exe scripts\run_stage4_summaries.py --limit 100 --persist

Default: read-only PostgreSQL + atomic JSON checkpoints. --persist appends
separate audit rows after applying migration; it never edits original news
or approved posts. Models run only against Ollama on 127.0.0.1.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.abstractive_summary import (
    ALLOWED_MODELS, DEFAULT_MODEL, PIPELINE_VERSION, source_digest, summarize_event_cluster,
)
from processing.abstractive_validator import normalize
from processing.stage4_evidence import shared_boilerplate, clean_working_materials

DEFAULT_OUTPUT = ROOT / "collector_test_results" / "stage4_abstractive_100.json"


def metrics(rows: list[dict], min_clusters: int) -> dict:
    statuses = Counter(row["status"] for row in rows)
    attempted = [row for row in rows if row.get("attempts")]
    valid = [row for row in rows if row["status"] == "summary_draft"]
    cleaned = sum(bool(row.get("source_cleanup_notes")) for row in rows)
    no_event = sum("no_supported_news_event" in row.get("validation", {}).get("issues", []) for row in rows)
    failing = Counter(issue for row in rows
                      for issue in row.get("validation", {}).get("issues", []))
    copied = [row.get("validation", {}).get("copied_sentence_ratio", 0)
              for row in rows if row.get("gemma_summary")]
    repeats = [row.get("validation", {}).get("repeated_sentence_count", 0)
               for row in rows if row.get("gemma_summary")]
    readability = [row.get("validation", {}).get("readability_proxy", 0)
                   for row in rows if row.get("gemma_summary")]
    baseline_sim = [SequenceMatcher(None, normalize(row.get("textrank_summary", "")),
                                    normalize(row.get("gemma_summary", ""))).ratio()
                    for row in rows if row.get("gemma_summary")]
    return {
        "clusters_processed": len(rows),
        "required_clusters": min_clusters,
        "minimum_sample_met": len(rows) >= min_clusters,
        "summary_drafts": statuses["summary_draft"],
        "summary_review": statuses["summary_review"],
        "source_cleanup_clusters": cleaned,
        "no_supported_news_event": no_event,
        "source_integrity_review": sum(
            any(str(x).startswith("source_internal_")
                for x in row.get("validation", {}).get("issues", []))
            for row in rows),
        "language_quality_review": sum(
            any(str(x).startswith("language_")
                for x in row.get("validation", {}).get("issues", []))
            for row in rows),
        "semantic_excess_claim_rows_ignored": sum(
            "semantic_extra_claims_ignored" in
            row.get("validation", {}).get("flags", [])
            for row in rows),
        "model_generation_attempted": len(attempted),
        "validator_passed_with_llm_advisory": len(valid),
        "validator_pass_rate_of_all": round(len(valid)/max(1, len(rows)), 4),
        "mean_copy_sentence_ratio": round(sum(copied)/len(copied), 3) if copied else None,
        "mean_repeated_sentence_pairs": round(sum(repeats)/len(repeats), 3) if repeats else None,
        "mean_sentence_length_readability_proxy": round(sum(readability)/len(readability), 3) if readability else None,
        "mean_textrank_to_gemma_similarity": round(
            sum(baseline_sim)/len(baseline_sim), 3) if baseline_sim else None,
        "issues": dict(failing.most_common()),
        "automated_quality_metrics_are_not_accuracy": True,
        "manual_factual_accuracy": None,
        "manual_readability": None,
        "manual_event_coverage": None,
        "note": ("Mechanical checks plus Gemma's self-evaluation are NOT independent "
                 "semantic truth labels. Every draft requires human approval."),
    }


def representative(rows: list[dict], count: int = 10) -> list[dict]:
    if not rows:
        return []
    # First show examples from different statuses, then divergent and copied
    # drafts, then evenly spaced rows; never claim these are random blind labels.
    order = []
    for status in ("summary_draft", "summary_review"):
        order += [i for i, row in enumerate(rows) if row["status"] == status][:3]
    divergent = sorted(range(len(rows)), key=lambda i:
        SequenceMatcher(None, normalize(rows[i].get("textrank_summary", "")),
                        normalize(rows[i].get("gemma_summary", ""))).ratio()
        if rows[i].get("gemma_summary") else 2)
    order.extend(divergent[:4])
    order.extend(round(i*(len(rows)-1)/max(1, count-1)) for i in range(count))
    chosen = list(dict.fromkeys(order))[:count]
    return [{
        "cluster_id": rows[i]["cluster_id"],
        "title": rows[i]["source_materials"][0]["title"]
                 if rows[i]["source_materials"] else "",
        "before_textrank": rows[i]["textrank_summary"],
        "after_gemma_candidate": rows[i]["gemma_summary"],
        "status": rows[i]["status"],
        "validator_issues": rows[i]["validation"].get("issues", []),
        "validator_flags": rows[i]["validation"].get("flags", []),
    } for i in chosen]


def assemble(rows: list[dict], *, min_clusters: int, model: str,
             started_utc: str, persisted: bool) -> dict:
    return {
        "stage": 4, "created_utc": started_utc,
        "model": model, "pipeline_version": PIPELINE_VERSION,
        "local_only": True, "training_started": False,
        "changed_legacy_news_or_posts": False,
        "persist_audit_to_db": persisted,
        "all_drafts_require_human_moderation": True,
        "metrics": metrics(rows, min_clusters),
        "representative_before_after": representative(rows),
        "clusters": rows,
    }


def checkpoint(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    os.replace(tmp, path)


def _materials(session, cluster_id: int) -> list[dict]:
    from database.models import News
    members = (session.query(News).filter(News.cluster_id == cluster_id)
               .order_by(News.is_primary.desc(), News.id.asc()).all())
    return [{
        "id": n.id, "title": n.title, "url": n.url, "text": n.text,
        "is_primary": n.is_primary,
        "published_at": n.published_at.isoformat() if n.published_at else None,
        "source": getattr(n.source, "name", ""),
    } for n in members]


def _textrank(materials: list[dict]) -> tuple[str, list[str]]:
    from processing.summarizer import build_textrank_summary
    if not materials:
        return "", ["no_cluster_members"]
    text = "\n".join((m["title"] or "") + ". " + (m["text"] or "")[:9000]
                     for m in materials[:5])
    try:
        return build_textrank_summary(
            text, title=materials[0]["title"], sentence_count=4,
            max_chars=800), []
    except Exception as exc:
        return "", ["textrank_error:" + type(exc).__name__]


def _persist(session, row: dict):
    from database.summary_models import SummaryRun
    session.add(SummaryRun(
        cluster_id=row["cluster_id"],
        source_sha256=row["source_sha256"],
        source_materials=row["source_materials"],
        extracted_facts=row["facts"],
        textrank_summary=row["textrank_summary"],
        gemma_summary=row["gemma_summary"],
        model=row["model"],
        model_parameters=row["model_parameters"],
        validation=row["validation"],
        attempts=row["attempts"],
        status=row["status"],
    ))
    session.commit()


def run(*, limit: int, min_clusters: int, model: str, max_retries: int,
        output: Path, persist: bool, resume: bool = False,
        cluster_ids: list[int] | None = None, input_json: Path | None = None) -> dict:
    """Use either unchanged legacy PostgreSQL or the exact previous 100 source snapshots."""
    if input_json and persist:
        raise ValueError("JSON replay is read-only: --persist is disabled")
    if input_json and input_json.resolve() == output.resolve():
        raise ValueError("Never overwrite the previous source JSON")
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    previous = {}
    if resume:
        if not output.is_file():
            raise FileNotFoundError("Cannot resume: no existing JSON report")
        report = json.loads(output.read_text(encoding="utf-8"))
        if (report.get("stage") != 4 or report.get("model") != model
                or report.get("pipeline_version") != PIPELINE_VERSION):
            raise ValueError("Cannot resume: model/stage/pipeline mismatch")
        if persist and not report.get("persist_audit_to_db"):
            raise ValueError("Cannot resume with --persist from a read-only report")
        started = report.get("created_utc", started)
        previous = {str(row["cluster_id"]): row for row in report.get("clusters", [])}
    session = None
    if input_json:
        original = json.loads(input_json.read_text(encoding="utf-8"))
        if original.get("stage") != 4 or not isinstance(original.get("clusters"), list):
            raise ValueError("Input JSON must be the original Stage 4 report")
        source_sets = {}
        for row in original["clusters"]:
            ident = int(row["cluster_id"])
            if (not cluster_ids or ident in cluster_ids) and ident not in source_sets:
                source_sets[ident] = row["source_materials"]
                if len(source_sets) >= limit:
                    break
    else:
        from database.db import get_session
        from database.models import Cluster, News
        from sqlalchemy import inspect
        session = get_session()
        if persist and not inspect(session.get_bind()).has_table("summary_runs"):
            session.close()
            raise RuntimeError("Apply 'python -m alembic upgrade head' to a BACKED-UP DB")
        try:
            query = (session.query(Cluster.id).join(News, News.cluster_id == Cluster.id)
                     .distinct().order_by(Cluster.id.desc()))
            if cluster_ids:
                query = query.filter(Cluster.id.in_(cluster_ids))
            ids = [r[0] for r in query.limit(limit).all()]
            source_sets = {cluster_id: _materials(session, cluster_id) for cluster_id in ids}
        except Exception:
            session.close()
            raise
    try:
        shared = shared_boilerplate(list(source_sets.values()))
        for idx, (cluster_id, materials) in enumerate(source_sets.items(), 1):
            cached = previous.get(str(cluster_id))
            if (cached and cached.get("source_sha256") == source_digest(materials)
                    and cached.get("model") == model
                    and cached.get("pipeline_version") == PIPELINE_VERSION):
                rows.append(cached)
                print(f"[{idx}/{len(source_sets)}] #{cluster_id} reused", flush=True)
                continue
            working, cleanup_notes = clean_working_materials(materials, shared)
            baseline, textrank_flags = _textrank(working)
            outcome = summarize_event_cluster(
                cluster_id, materials, baseline, model=model,
                max_retries=max_retries, working_materials=working,
                cleanup_notes=cleanup_notes,
            ).to_dict()
            if textrank_flags:
                outcome["validation"].setdefault("flags", []).extend(textrank_flags)
            rows.append(outcome)
            if persist:
                try:
                    _persist(session, outcome)
                except Exception:
                    session.rollback()
                    raise
            checkpoint(output, assemble(rows, min_clusters=min_clusters,
                                        model=model, started_utc=started,
                                        persisted=persist))
            print(f"[{idx}/{len(source_sets)}] #{cluster_id} {outcome['status']} "
                  f"{len(outcome['gemma_summary'])} chars; "
                  f"issues={outcome['validation'].get('issues', [])}", flush=True)
    finally:
        if session is not None:
            session.close()
    final = assemble(rows, min_clusters=min_clusters, model=model,
                     started_utc=started, persisted=persist)
    final["source_input_mode"] = "previous_real_json" if input_json else "postgresql"
    checkpoint(output, final)
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Local Gemma source-grounded summary audit; no publishing")
    parser.add_argument("--input-json", type=Path, default=None,
                        help="Replay the EXACT original 100 real cluster source snapshots")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-clusters", type=int, default=100)
    parser.add_argument("--cluster-id", type=int, action="append", default=[])
    parser.add_argument("--model", choices=sorted(ALLOWED_MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse successful or review results when source and model match")
    parser.add_argument("--persist", action="store_true",
                        help="Append audit rows to migrated DB. No legacy writes.")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.min_clusters < 1 or not 0 <= args.max_retries <= 2:
        parser.error("Positive limits, max-retries 0..2 required")
    if args.input_json and args.persist:
        parser.error("--persist cannot be used with --input-json")
    if args.input_json and args.input_json.resolve() == args.output.resolve():
        parser.error("Refusing to overwrite the input JSON")
    if args.output.suffix.casefold() != ".json":
        parser.error("Output must be a JSON file")
    report = run(limit=args.limit, min_clusters=args.min_clusters, model=args.model,
                 max_retries=args.max_retries, output=args.output,
                 persist=args.persist, resume=args.resume, cluster_ids=args.cluster_id,
                 input_json=args.input_json)
    stats = report["metrics"]
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Report: {args.output.resolve()}")
    if not stats["minimum_sample_met"]:
        print("INSUFFICIENT REAL CLUSTERS: do not declare Stage 4 complete.")
        return 2
    if stats["model_generation_attempted"] == 0:
        print("NO GEMMA GENERATION: run Ollama and inspect model availability.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
