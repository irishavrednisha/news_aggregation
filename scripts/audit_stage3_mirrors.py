"""Read-only cross-check of real stage-2 exact-copy links and rejected originals.

Exact mirror positives are NOT independent semantic SBERT matches. Never
report them as recall denominators for the event clustering model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def run(input_path: Path) -> dict:
    payload=json.loads(input_path.read_text(encoding="utf-8"))
    rows=payload["real_sample"]["articles"]
    by_url={x["canonical_url"]:x for x in rows}
    copies=[x for x in rows if x.get("reason")=="duplicate_text"]
    checks=[]
    for item in copies:
        parent=by_url.get(item.get("duplicate_of_url"))
        checks.append({
            "source":item["source"],"title":item["title"],
            "url":item["url"],"linked_to":item.get("duplicate_of_url"),
            "original_status":parent.get("status") if parent else "missing",
            "original_reason":parent.get("reason") if parent else "missing",
            "same_title":bool(parent and item["title"]==parent["title"]),
            "same_text":bool(parent and item["text"]==parent["text"]),
            "same_published_at":bool(parent and item["published_at"]==parent["published_at"]),
        })
    classifications=Counter(x["original_status"] for x in checks)
    failed=[x for x in checks if not all(
        x[k] for k in ("same_title","same_text","same_published_at"))]
    return {
        "stage":"stage3_stage2_mirror_provenance_check",
        "read_only":True,"db_writes":0,
        "raw_articles":len(rows),
        "exact_mirror_pairs":len(checks),
        "promoted_news_mirror_links":classifications.get("accepted",0),
        "mirrors_of_rejected_originals":classifications.get("rejected",0),
        "not_exact_mirrors":len(failed),
        "checks":checks,
        "warning":"Rejected originals do NOT appear in the accepted-News stage-3 clusters. "
                  "The OСК mentor-training story and three exact mirrors were rejected at "
                  "stage 2 as corporate_or_hr_event and duplicate_text.",
        "quality_caveat":"All 21 are identical content, NOT additional paraphrase/semantic-positive labels."
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("checks",)},
                     ensure_ascii=False,indent=2))
    if result["not_exact_mirrors"]:
        raise SystemExit("Input data claims a duplicate but text/date/title contradict it")


if __name__=="__main__":
    main()
