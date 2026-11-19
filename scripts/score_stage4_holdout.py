"""Score *independent human* labels of the frozen Stage 4 holdout.

Never infer ground truth from Gemma's automatic status or self-verdict. Missing
ratings remain missing; no result grants permission for automatic publication.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.stage4_independent_holdout import atomic_json

DIMENSIONS = ("faithful_facts", "natural_russian", "main_event_covered",
              "meaningful_news_event", "source_quality")
VALID = {"pass", "fail", "na"}


def confidence_interval(passed: int, count: int) -> dict | None:
    if not count:
        return None
    z = 1.96
    p = passed / count
    den = 1 + z*z/count
    center = (p + z*z/(2*count))/den
    spread = z*sqrt(p*(1-p)/count + z*z/(4*count*count))/den
    return {"fraction": round(p, 4),
            "lower_95_wilson": round(center-spread, 4),
            "upper_95_wilson": round(center+spread, 4)}


def score(*, report: dict, mapping: dict, label_rows: list[dict]) -> dict:
    seen = {}
    for row in label_rows:
        rid = (row.get("review_id") or "").strip()
        if not rid or rid not in mapping or rid in seen:
            raise ValueError("Unknown or repeated review ID: " + rid)
        values = {key: (row.get(key) or "").strip().lower()
                  for key in DIMENSIONS}
        if any(value not in VALID | {""} for value in values.values()):
            raise ValueError(f"Use pass/fail/na only: {rid}")
        seen[rid] = values
    if set(seen) != set(mapping):
        raise ValueError("The labels must match every blind review ID exactly")

    originals = {str(row["cluster_id"]): row for row in report["clusters"]}
    missing, evaluations, false_accepts, missed_good = 0, [], [], []
    for rid, key in mapping.items():
        row = originals.get(str(key["cluster_id"]))
        if row is None or row["status"] != key["automatic_status"]:
            raise ValueError("Private key or model report was modified")
        labels = seen[rid]
        if not (row.get("gemma_summary") or "").strip():
            if any(value not in {"", "na"} for value in labels.values()):
                raise ValueError("Cannot rate an absent summary: " + rid)
            continue
        if any(value == "" for value in labels.values()):
            missing += 1
            continue
        source_ok = labels["source_quality"] == "pass"
        correct = source_ok and all(labels[dim] == "pass"
                                    for dim in DIMENSIONS if dim != "source_quality")
        evaluations.append({"review_id": rid,
                            "automatic_status": row["status"],
                            "human_all_dimensions_pass": correct,
                            "source_sufficient": source_ok,
                            "human_labels": labels})
        if row["status"] == "summary_draft" and not correct:
            false_accepts.append(rid)
        if row["status"] == "summary_review" and correct:
            missed_good.append(rid)
    accepted = [row for row in evaluations
                if row["automatic_status"] == "summary_draft"]
    draft_total = sum(row["status"] == "summary_draft"
                      for row in report["clusters"])
    candidates = sum(bool(row.get("gemma_summary")) for row in report["clusters"])
    passed = sum(row["human_all_dimensions_pass"] for row in accepted)
    return {
        "stage": "4-independent-human-review",
        "sample_size": len(report["clusters"]),
        "model": report.get("model"),
        "pipeline_version": report.get("pipeline_version"),
        "generated_summaries": candidates,
        "automatic_drafts": draft_total,
        "human_labeled_summaries": len(evaluations),
        "human_unlabeled_summaries": missing,
        "human_labeled_automatic_drafts": len(accepted),
        "automatic_drafts_not_yet_labeled": draft_total - len(accepted),
        "drafts_passing_all_human_checks": passed,
        "drafts_with_detected_human_issues": len(false_accepts),
        "draft_precision_estimate": confidence_interval(passed, len(accepted)),
        "good_candidates_rejected_by_automation": len(missed_good),
        "false_accepted_blind_ids": false_accepts,
        "review_candidates_blind_ids": missed_good,
        "all_accepted_drafts_reviewed": len(accepted) == draft_total,
        "all_generated_summaries_reviewed": len(evaluations) == candidates,
        "independent_reviewer_identity_not_verified": True,
        "not_publication_approval": True,
        "scoring_note": (
            "Only human-provided labels count. Incomplete review is not "
            "evidence of accuracy; even a clean score does not authorize publishing."),
        "evaluations": evaluations,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)
    directory = args.out_dir
    with (directory/"review_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle, delimiter=";"))
    result = score(
        report=json.loads((directory/"gemma_holdout.json").read_text(encoding="utf-8")),
        mapping=json.loads((directory/"private_review_key.json").read_text(encoding="utf-8")),
        label_rows=labels)
    atomic_json(directory/"human_review_score.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "evaluations"},
                     ensure_ascii=False, indent=2))
    return 0 if result["all_accepted_drafts_reviewed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
