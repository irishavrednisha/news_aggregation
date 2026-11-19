"""Read-only, repeatable stage-3 baseline/new clustering audit over stage-2 JSON.

No database import, SQL writes, migrations, publication, summary or tagger.
Real quality requires EXTERNAL pair labels; unlabeled predictions are never
reported as precision/recall. Default SBERT; --backend lexical is a smoke test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.event_matching import (
    MODEL_NAME, MERGE_THRESHOLD, choose_existing_cluster, cosine,
    embedding_text, representative_key,
)


def load_articles(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("real_sample"), dict):
        rows = data["real_sample"].get("articles", [])
    elif isinstance(data, dict):
        rows = data.get("articles", data.get("news", []))
    else:
        raise ValueError("Expected articles or stage2 report JSON")
    if not isinstance(rows, list):
        raise ValueError("Invalid article list")
    accepted = [r for r in rows if isinstance(r, dict) and
                r.get("status", "accepted") == "accepted" and
                r.get("url") and r.get("title")]
    # Stable, replayable order; accepted intake may include old industrial articles.
    accepted.sort(key=lambda r: (str(r.get("published_at") or "9999"),
                                 str(r.get("url") or ""), str(r.get("title") or "")))
    seen = set()
    unique = []
    for row in accepted:
        if row["url"] not in seen:
            unique.append(row)
            seen.add(row["url"])
    return len(rows), unique


def raw_exact_copy_links(path: Path, accepted_rows: list[dict]) -> dict[str, list[dict]]:
    """Stage-2 rejected exact mirrors are references, never clustering inputs.

    Return them under their accepted canonical News URL, including translated
    Cyrillic/punycode paths. Duplicate links do not increase cluster_size.
    """
    from processing.exact_duplicates import normalize_url
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("real_sample"), dict):
        raw = data["real_sample"].get("articles", [])
    elif isinstance(data, dict):
        raw = data.get("articles", data.get("news", []))
    else:
        raw = data
    index = {normalize_url(row["url"]): row["url"] for row in accepted_rows}
    links: dict[str, list[dict]] = {}
    for row in raw:
        if not isinstance(row, dict) or not str(row.get("reason", "")).startswith("duplicate_"):
            continue
        original_url = index.get(normalize_url(row.get("duplicate_of_url", "")))
        if not original_url or not row.get("url") or row["url"] == original_url:
            continue
        links.setdefault(original_url, []).append({
            "source": row.get("source") or "",
            "url": row["url"],
            "link_type": "stage2_exact_copy",
        })
    for refs in links.values():
        refs.sort(key=lambda ref: (ref["url"], ref["source"]))
    return links


def _lexical_embedding(text, dimensions=1024):
    """Hashing char 4-grams. NOT SBERT; only for deterministic CI smoke tests."""
    import math
    from collections import Counter
    from processing.event_matching import normalize
    value = " " + normalize(text) + " "
    grams = Counter(value[i:i+4] for i in range(max(0, len(value) - 3)))
    result = [0.0] * dimensions
    for gram, count in grams.items():
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1 if digest[4] & 1 else -1
        result[idx] += sign * (1 + math.log(count))
    mag = math.sqrt(sum(x*x for x in result))
    return [x / mag for x in result] if mag else result


def embeddings(rows, *, backend, model_name):
    baseline_texts = [
        f"{r.get('title', '')}. {r.get('text', '')}" for r in rows
    ]
    candidate_texts = [embedding_text(row) for row in rows]
    if backend == "lexical":
        return (
            [_lexical_embedding(s) for s in baseline_texts],
            [_lexical_embedding(s) for s in candidate_texts]
        )
    if backend != "sbert":
        raise ValueError("Unsupported backend: " + backend)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    baseline = model.encode(baseline_texts, normalize_embeddings=True,
                            show_progress_bar=False).tolist()
    current = model.encode(candidate_texts, normalize_embeddings=True,
                           show_progress_bar=False).tolist()
    return baseline, current


def _centroid(vectors):
    import math
    dim = len(vectors[0])
    summed = [math.fsum(float(v[i]) for v in vectors) / len(vectors)
              for i in range(dim)]
    magnitude = math.sqrt(sum(x*x for x in summed))
    return [x / magnitude for x in summed] if magnitude else summed


def baseline_cluster(rows, vectors, threshold):
    """Reproduce original running-centroid nearest cluster, with fixed input order."""
    groups = []
    decisions = []
    for row, vector in zip(rows, vectors):
        best, maximum = None, 0.0
        for group in groups:
            score = cosine(vector, _centroid(group["vectors"]))
            if score > maximum:
                best, maximum = group, score
        if best is not None and maximum >= threshold:
            best["rows"].append(row)
            best["vectors"].append(vector)
            selected = best["id"]
            reason = "centroid_threshold"
        else:
            selected = "base:" + str(len(groups) + 1).zfill(4)
            groups.append({"id": selected, "rows": [row], "vectors": [vector]})
            reason = "below_threshold"
        decisions.append({"url": row["url"], "title": row["title"],
                          "cluster": selected, "similarity": round(maximum, 5),
                          "reason": reason})
    return groups, decisions


def conservative_cluster(rows, vectors, threshold):
    groups = []
    decisions = []
    for row, vec in zip(rows, vectors):
        selected, decision = choose_existing_cluster(row, vec, groups,
                                                    threshold=threshold)
        if selected is None:
            selected = {"id": "event:" + str(len(groups) + 1).zfill(4),
                        "items": [], "rows": []}
            groups.append(selected)
            action = "new"
        else:
            action = "attached"
        selected["items"].append((row, vec))
        selected["rows"].append(row)
        decisions.append({"url": row["url"], "title": row["title"],
                          "cluster": selected["id"], "action": action,
                          "similarity": round(decision.similarity, 5),
                          "reason": decision.reason,
                          "source": row.get("source", "")})
    return groups, decisions


def group_summary(groups, *, exact_copy_links=None):
    exact_copy_links = exact_copy_links or {}
    histogram = Counter(len(g["rows"]) for g in groups)
    singles = histogram.get(1, 0)
    multisource = sum(
        len({row.get("source") for row in g["rows"]}) >= 2
        for g in groups
    )
    multi_source_with_exact = sum(
        len({r.get("source") for r in group["rows"]} |
            {r.get("source") for article in group["rows"]
             for r in exact_copy_links.get(article["url"], [])}) >= 2
        for group in groups
    )
    return {
        "cluster_count": len(groups),
        "multi_source_clusters_including_exact_copies": multi_source_with_exact,
        "raw_exact_copy_links": sum(
            len(exact_copy_links.get(article["url"], []))
            for group in groups for article in group["rows"]
        ),
        "singleton_clusters": singles,
        "multi_article_clusters": len(groups) - singles,
        "multi_source_clusters": multisource,
        "max_cluster_size": max(histogram, default=0),
        "mean_cluster_size": round(sum(len(g["rows"]) for g in groups) /
                                   len(groups), 4) if groups else 0,
        "cluster_size_distribution": dict(sorted(histogram.items())),
        "clusters": [{
            "id": g["id"],
            "size": len(g["rows"]),
            "primary_url": min(g["rows"], key=representative_key)["url"],
            "source_urls": (
                [{"source": row.get("source"), "url": row["url"],
                  "link_type": "accepted_event_article"} for row in g["rows"]]
                + [ref for row in g["rows"]
                   for ref in exact_copy_links.get(row["url"], [])]
            ),
        } for g in groups],
    }


def compare_pair_labels(rows, baseline, revised, path, input_path):
    """Independent manually curated same_event labels on genuine article URLs."""
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", []) if isinstance(data, dict) else data
    file_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if isinstance(data, dict) and data.get("corpus_sha256"):
        if data["corpus_sha256"] != file_sha:
            raise ValueError("Gold labels belong to a different immutable real corpus")
    input_data = json.loads(input_path.read_text(encoding="utf-8"))
    raw = (input_data.get("real_sample", {}).get("articles", [])
           if isinstance(input_data, dict) else input_data)
    original_urls = [r["url"] for r in raw if isinstance(r, dict)
                     and r.get("status", "accepted") == "accepted"
                     and r.get("url") and r.get("title")]
    if not isinstance(pairs, list):
        raise ValueError("Gold file must contain pairs")
    urls = {r["url"] for r in rows}
    maps = {name: {d["url"]: d["cluster"] for d in decisions}
            for name, decisions in (("baseline", baseline), ("revised", revised))}
    # Count unique, well-formed pair decisions; reversed or repeated labels
    # must never inflate the claimed number of reviewed real examples.
    reviewed_pairs = set()
    for entry in pairs:
        if not isinstance(entry, dict):
            raise ValueError("Every gold pair must be a JSON object")
        if type(entry.get("same_event")) is not bool:
            raise ValueError("Gold pair missing boolean same_event")
        try:
            a = entry["a"] if entry.get("a") else original_urls[entry["a_index"]]
            b = entry["b"] if entry.get("b") else original_urls[entry["b_index"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Gold pair references an invalid accepted article") from exc
        if a not in urls or b not in urls or a == b:
            raise ValueError(f"Gold URL not in accepted corpus or identical: {a} ; {b}")
        pair = tuple(sorted((a, b)))
        if pair in reviewed_pairs:
            raise ValueError(f"Duplicate/reversed gold pair: {a} ; {b}")
        reviewed_pairs.add(pair)
    result = {}
    for name, assignments in maps.items():
        counts = Counter({key: 0 for key in (
            "gold_same_event", "gold_different_event", "correct_merge",
            "false_merge", "missed_merge", "correct_separation"
        )})
        mistakes = []
        for entry in pairs:
            # Compact human-reviewed fixtures may use original accepted row
            # indices, guarded by the SHA256 of the entire source report.
            a = entry.get("a") or original_urls[entry["a_index"]]
            b = entry.get("b") or original_urls[entry["b_index"]]
            if a not in urls or b not in urls or a == b:
                raise ValueError(f"Gold URL not in accepted corpus or identical: {a} ; {b}")
            if type(entry.get("same_event")) is not bool:
                raise ValueError("Gold pair missing boolean same_event")
            predicted = assignments[a] == assignments[b]
            expected = entry["same_event"]
            if expected:
                counts["gold_same_event"] += 1
            else:
                counts["gold_different_event"] += 1
            if predicted and expected:
                counts["correct_merge"] += 1
            elif predicted and not expected:
                counts["false_merge"] += 1
            elif not predicted and expected:
                counts["missed_merge"] += 1
            else:
                counts["correct_separation"] += 1
            if predicted != expected:
                mistakes.append({"a": a, "b": b, "expected_same_event": expected,
                                 "predicted_same_event": predicted,
                                 "evidence": entry.get("evidence")})
        tp, fp, fn = (counts[k] for k in ("correct_merge", "false_merge", "missed_merge"))
        counts["labeled_pairs"] = len(pairs)
        result[name] = {
            "counts": dict(counts),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "mistakes": mistakes,
        }
    result["annotation_status"] = (
        data.get("verification_status", "unspecified")
        if isinstance(data, dict) else "unspecified"
    )
    return result


def run(path, *, gold=None, backend="sbert", model=MODEL_NAME,
        baseline_threshold=0.7, threshold=MERGE_THRESHOLD,
        min_decisions=100):
    if not (0 < baseline_threshold <= 1 and 0 < threshold <= 1):
        raise ValueError("Thresholds must be in (0,1]")
    raw_count, rows = load_articles(path)
    if len(rows) < min_decisions:
        raise ValueError(f"Only {len(rows)} accepted unique articles, need {min_decisions}")
    old_vectors, new_vectors = embeddings(rows, backend=backend, model_name=model)
    old_groups, old_decisions = baseline_cluster(rows, old_vectors, baseline_threshold)
    new_groups, new_decisions = conservative_cluster(rows, new_vectors, threshold)
    gold_report = compare_pair_labels(rows, old_decisions, new_decisions, gold, path)
    exact_links = raw_exact_copy_links(path, rows)
    old_stats, new_stats = (
        group_summary(old_groups, exact_copy_links=exact_links),
        group_summary(new_groups, exact_copy_links=exact_links)
    )
    return {
        "stage": "stage3_event_clustering",
        "read_only": True,
        "database_imported": False,
        "db_writes": 0,
        "publisher_called": False,
        "summarizer_called": False,
        "backend": backend,
        "sbert_model": model if backend == "sbert" else None,
        "lexical_smoke_only": backend != "sbert",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(gold.read_bytes()).hexdigest() if gold else None,
        "input_articles": raw_count,
        "accepted_unique_articles": len(rows),
        "cluster_decisions_checked": len(new_decisions),
        "baseline_threshold": baseline_threshold,
        "revised_threshold": threshold,
        "baseline": old_stats,
        "revised": new_stats,
        "delta_cluster_count": new_stats["cluster_count"] - old_stats["cluster_count"],
        "delta_singletons": new_stats["singleton_clusters"] - old_stats["singleton_clusters"],
        "gold_real_metrics": gold_report,
        "quality_caveat": (
            "No labeled real pairs: FP/FN precision and recall unknown."
            if gold_report is None else (
                "Real pair labels are provisional and NOT independently verified. "
                "Error counts are preliminary; cluster counts do not prove quality."
                if gold_report.get("annotation_status") != "independently_verified" else None
            )
        ),
        "decisions": new_decisions,
        "baseline_decisions": old_decisions,
        "examples": {
            "merged": [d for d in new_decisions if d["action"] == "attached"][:25],
            "separated_near_matches": sorted(
                (d for d in new_decisions if d["action"] == "new"
                 and d["similarity"] >= 0.55),
                key=lambda d: (-d["similarity"], d["url"])
            )[:25],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Stage 2 report (real_sample.articles) or articles array")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold", type=Path, help="Independent manually labeled URL pairs")
    parser.add_argument("--backend", choices=("sbert", "lexical"), default="sbert")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--baseline-threshold", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=MERGE_THRESHOLD)
    parser.add_argument("--min-decisions", type=int, default=100)
    args = parser.parse_args(argv)
    report = run(args.input, gold=args.gold, backend=args.backend,
                 model=args.model, baseline_threshold=args.baseline_threshold,
                 threshold=args.threshold, min_decisions=args.min_decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({
        "checked_real_decisions": report["cluster_decisions_checked"],
        "baseline_clusters": report["baseline"]["cluster_count"],
        "new_clusters": report["revised"]["cluster_count"],
        "baseline_singletons": report["baseline"]["singleton_clusters"],
        "new_singletons": report["revised"]["singleton_clusters"],
        "real_gold_metrics": report["gold_real_metrics"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
