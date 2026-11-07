"""Read-only comparison of role-aware tag challenger vs production SBERT.

Usage:
  python check_tag_role_challenger.py collector_test_results/tagging_audit_100.json --reference tagging_reference_100.csv --include-drafts
  python check_tag_role_challenger.py collector_test_results/new_tagging_audit.json

No PyTorch model loading, PostgreSQL writes, collector or MAX publication.
This candidate is intentionally NOT connected to the production tagger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from processing.tag_role_challenger import candidate_tags
from processing.tagging_event_guard import nonindustrial_event_reason
from evaluate_tagging import read_references, metrics


def compare(audit: dict) -> dict:
    changed = []
    rows = []
    for row in audit.get("results", []):
        if not row.get("url"):
            continue
        original = row.get("raw_tags_before_event_guard", row.get("tags") or [])
        event = nonindustrial_event_reason(row.get("title") or "")
        baseline = [] if event else original
        experimental = candidate_tags(row)
        if set(baseline) != set(experimental["tags"]):
            changed.append(row["url"])
        rows.append({
            "source": row.get("source"),
            "title": row.get("title"),
            "url": row["url"],
            "baseline_tags": baseline,
            "candidate_tags": experimental["tags"],
            "event_reason": experimental["event_reason"],
            "rejected_categories": experimental["rejected_categories"],
            "production_process_evidence": experimental["production_process"],
        })
    return {
        "read_only": True,
        "production_tagger_changed": False,
        "warning": "Exploratory hypothesis; labels must be manually verified and test articles should be new.",
        "articles": len(rows),
        "changed": len(changed),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline challenger for industrial tag roles"
    )
    parser.add_argument("audit", type=Path)
    parser.add_argument("--reference", type=Path, default=None,
                        help="Optional reviewed labels CSV/JSON")
    parser.add_argument("--include-drafts", action="store_true",
                        help="Experimental metrics only; not validated gold")
    parser.add_argument("--output", type=Path,
                        default=Path("collector_test_results/tag_role_challenger.json"))
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    result = compare(audit)
    if args.reference:
        reference = read_references(args.reference, include_drafts=args.include_drafts)
        entries = [(r, reference[r["url"]]) for r in result["results"]
                   if r["url"] in reference]
        if entries:
            before = metrics(entries, lambda r: r["baseline_tags"])
            after = metrics(entries, lambda r: r["candidate_tags"])
            result["evaluation"] = {
                "label_status": "DRAFT - NOT VERIFIED" if args.include_drafts else "verified",
                "matched_articles": len(entries),
                "baseline": {k: v for k, v in before.items() if k not in ("errors", "per_tag")},
                "candidate": {k: v for k, v in after.items() if k not in ("errors", "per_tag")},
            }
        else:
            result["evaluation"] = {
                "matched_articles": 0,
                "warning": "No verified overlapping labels; use --include-drafts only for experiments."
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"Articles: {result['articles']}; changed: {result['changed']}")
    if result.get("evaluation", {}).get("matched_articles"):
        ev = result["evaluation"]
        print(f"Reference: {ev['label_status']}, n={ev['matched_articles']}")
        print("Micro-F1: {:.3f} -> {:.3f}".format(
            ev["baseline"]["micro_f1"], ev["candidate"]["micro_f1"]
        ))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
