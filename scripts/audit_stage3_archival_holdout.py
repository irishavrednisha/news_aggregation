"""Read-only event-identity audit on extra historical real-news holdout.

The fixture is assembled from the user's separate 199-article archive, not the
156 accepted-news corpus. Rejected/review-stage historical records are NOT
promoted to News or counted in production. Matching copy cases do not replace
the need for blinded human annotation or new semantic-paraphrase samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from processing.event_matching import MODEL_NAME, cosine, embedding_text, same_event


def _tokens(s: str) -> Counter:
    return Counter(re.findall(r"[а-яa-z0-9]+", s.casefold()))


def _lexical_cosine(a: str, b: str) -> float:
    ca, cb = _tokens(a), _tokens(b)
    dot = sum(n * cb[t] for t, n in ca.items())
    da = math.sqrt(sum(n*n for n in ca.values()))
    db = math.sqrt(sum(n*n for n in cb.values()))
    return dot / (da*db) if da and db else 0.0


def _verify(fixture: dict):
    rows = fixture.get("articles")
    pairs = fixture.get("pairs")
    if not isinstance(rows, list) or not isinstance(pairs, list) or not rows or not pairs:
        raise ValueError("Nonempty articles and labeled pairs required")
    by_id = {}
    urls = set()
    for item in rows:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), int)
            or not item.get("title") or not item.get("url")
            or item.get("id") in by_id or item["url"] in urls):
            raise ValueError("Invalid or repeated real article")
        by_id[item["id"]] = item
        urls.add(item["url"])
    known = set()
    for pair in pairs:
        if (not isinstance(pair, dict)
            or type(pair.get("same_event")) is not bool
            or pair.get("a_index") not in by_id
            or pair.get("b_index") not in by_id
            or pair["a_index"] == pair["b_index"]):
            raise ValueError("Invalid label, unreferenced article, or self pair")
        key = tuple(sorted((pair["a_index"], pair["b_index"])))
        if key in known:
            raise ValueError("Repeated or reversed labeled pair")
        known.add(key)
    return by_id, pairs


def run(path: Path, *, backend: str = "sbert") -> dict:
    raw = path.read_bytes()
    fixture = json.loads(raw)
    by_id, pairs = _verify(fixture)
    rows = [by_id[k] for k in sorted(by_id)]
    result = {
        "stage": "stage3_archival_holdout", "read_only": True, "db_writes": 0,
        "publisher_called": False, "summarizer_called": False,
        "archive_independence": "different_stage2_harvest_199_not_guaranteed_independent_of_every_old_url",
        "annotation_status": fixture["annotation_status"],
        "backend": backend, "input_sha256": hashlib.sha256(raw).hexdigest(),
        "articles": len(rows), "labeled_pairs": len(pairs),
        "positive_pairs": sum(bool(p["same_event"]) for p in pairs),
        "negative_pairs": sum(not p["same_event"] for p in pairs),
        "results": [],
    }
    if backend == "sbert":
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer(MODEL_NAME)
        vectors = model.encode([embedding_text(x) for x in rows], normalize_embeddings=True)
        baseline_vectors = model.encode(
            [(x["title"] + "\n" + (x.get("text") or "")) for x in rows],
            normalize_embeddings=True
        )
        ids = {item["id"]: n for n,item in enumerate(rows)}
        score = lambda a,b: cosine(vectors[ids[a]], vectors[ids[b]])
        old_score = lambda a,b: cosine(baseline_vectors[ids[a]], baseline_vectors[ids[b]])
        result["model"] = MODEL_NAME
    elif backend == "lexical":
        score = lambda a,b: _lexical_cosine(embedding_text(by_id[a]),embedding_text(by_id[b]))
        old_score = lambda a,b: _lexical_cosine(
            by_id[a]["title"] + " " + by_id[a].get("text", ""),
            by_id[b]["title"] + " " + by_id[b].get("text", ""))
        result["lexical_smoke_only"] = True
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    cm = {"baseline": Counter(), "revised": Counter()}
    for pair in pairs:
        a,b = pair["a_index"],pair["b_index"]
        true = pair["same_event"]
        sim = score(a,b)
        base = old_score(a,b)
        event = same_event(by_id[a],by_id[b],sim)
        predicted = event.merge
        baseline_predicted = base >= 0.70
        item = {
            "a_index": a,"b_index": b,
            "a_title": by_id[a]["title"], "b_title": by_id[b]["title"],
            "same_event": true,
            "embedding_similarity": round(sim,5),
            "baseline_fulltext_similarity":round(base,5),
            "baseline_same_event":baseline_predicted,
            "revised_same_event":predicted, "decision_reason":event.reason,
            "review_evidence":pair.get("evidence",""),
        }
        result["results"].append(item)
        for name,seen in (("baseline",baseline_predicted),("revised",predicted)):
            key = ("true_positive" if true else "false_positive") if seen else (
                  "false_negative" if true else "true_negative")
            cm[name][key]+=1
    result["metrics"] = {
        name: {key: counts[key] for key in (
            "true_positive","false_positive","false_negative","true_negative")}
        for name,counts in cm.items()
    }
    result["quality_caveat"] = (
        "Labels are self-reviewed against real text but NOT blinded independent human "
        "annotations; 5 positives include 3 pairs from the same 3-site Manturovo "
        "event and exact-copy reposts. Keep mirror and semantic quality separate."
    )
    return result


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,
                   default=Path("tests/fixtures/stage3_archival_holdout.json"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--backend",choices=["sbert","lexical"],default="sbert")
    args=p.parse_args()
    result=run(args.input,backend=args.backend)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items()
                      if k in ("backend","articles","labeled_pairs","positive_pairs",
                               "negative_pairs","metrics","annotation_status")},
                     ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
