"""Recover independently sourced Stage-4 event evidence from a frozen manifest.

This is a *new* artifact, never modifies the original. Cross-source pairs are
candidate events, not verified semantic equivalence. Every multidoс group must
be independently confirmed by human blind_review.event_coherence before the
final score may claim it is one event.

Optional additional source articles are fetched as FULL HTML with the normal
strict certificate verification of the existing collector. No guessed dates.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage4_independent_holdout import (
    atomic_json, clean_url, date_of, fingerprint, new_material,
    prior_signatures,
)

VERSION = "stage4-multisource-reconcile-v1"


def _ngrams(text: str) -> Counter[str]:
    raw = str(text or "").casefold()
    return Counter(raw[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(raw)-n+1)))


def _vectors(texts: list[str]) -> list[dict[str, float]]:
    counts = [_ngrams(text) for text in texts]
    df = Counter(key for count in counts for key in count)
    n = len(counts)
    output = []
    for count in counts:
        weighted = {g: v * (math.log((1+n)/(1+df[g]))+1)
                    for g,v in count.items()}
        norm = math.sqrt(sum(value*value for value in weighted.values())) or 1
        output.append({g: value/norm for g,value in weighted.items()})
    return output


def _similarity(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a,b = b,a
    return sum(v*b.get(k, 0) for k,v in a.items())


def propose_pairs(rows: list[dict], *, title_min: float = .135,
                  lead_min: float = .20) -> list[dict]:
    """Only match distinct source names, strongest non-overlapping candidates.

    Both thresholds are fixed before running the unseen evaluation; a title
    alone never proves event identity. True event co-reference is scored later
    by an independent human and cannot be silently inferred from these scores.
    """
    docs = [row["source_materials"][0] for row in rows]
    if any(len(row.get("source_materials") or []) != 1 for row in rows):
        raise ValueError("Reconcile accepts only the original single-document manifest")
    titles = _vectors([m.get("title","") for m in docs])
    leads = _vectors([str(m.get("text",""))[:800] for m in docs])
    pairs = []
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            left, right = docs[i], docs[j]
            if left.get("source") == right.get("source"):
                continue
            if clean_url(left.get("url")) == clean_url(right.get("url")):
                continue
            if (urlsplit(clean_url(left.get("url"))).hostname ==
                    urlsplit(clean_url(right.get("url"))).hostname):
                continue
            title_score = _similarity(titles[i], titles[j])
            if title_score < title_min:
                continue
            lead_score = _similarity(leads[i], leads[j])
            if lead_score < lead_min:
                continue
            pairs.append({
                "cluster_ids": [rows[i]["cluster_id"], rows[j]["cluster_id"]],
                "titles": [left.get("title"), right.get("title")],
                "source_names": [left.get("source"), right.get("source")],
                "urls": [left.get("url"), right.get("url")],
                "title_similarity": round(title_score, 4),
                "lead_similarity": round(lead_score, 4),
                "event_identity_verified": False,
                "requires_human_event_identity_review": True,
            })
    return sorted(pairs, key=lambda x: (
        -x["title_similarity"]-x["lead_similarity"],
        str(x["cluster_ids"])))


def _load_prior_from_manifest(document: dict, root: Path) -> dict:
    files = []
    entries = document.get("previous_files") or []
    if not entries:
        raise ValueError("Missing prior Stage-4 exclusion fingerprints")
    for entry in entries:
        p = Path(str(entry.get("path", "")))
        if not p.is_absolute():
            p = root/p
        if not p.is_file():
            raise FileNotFoundError("Exclusion file missing: "+str(p))
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Historical exclusion changed: "+str(p))
        files.append(p)
    return prior_signatures(files)


def _fetch_new_article(spec: dict, *, fetch=None) -> dict:
    # clean_url canonicalization rewrites http:// to https://. Check the
    # *original* scheme first to avoid silently upgrading untrusted links.
    original_url = str(spec.get("url") or "").strip()
    if urlsplit(original_url).scheme != "https":
        raise ValueError("Supplemental source must use HTTPS")
    url = clean_url(original_url)
    if not url:
        raise ValueError("Invalid supplemental URL")
    source = str(spec.get("source") or "").strip()
    if not source or len(source) > 100:
        raise ValueError("Invalid supplemental source name")
    if fetch is None:
        from collectors.data_collector import build_full_article
        fetch = build_full_article
    preview = SimpleNamespace(
        source=source, title=str(spec.get("title") or ""),
        url=url, published_at=None, preview_text="",
    )
    article = fetch(preview)
    if not article or clean_url(getattr(article, "url", "")) != url:
        raise ValueError("Article URL changed while collecting")
    if getattr(article, "text_source", "") != "html":
        raise ValueError("No verifiable full HTML text")
    return {
        "id": "supplement:"+hashlib.sha256(url.encode()).hexdigest()[:18],
        "source": source, "url": url,
        "title": str(getattr(article, "title", "") or ""),
        "text": str(getattr(article, "text", "") or ""),
        "text_method": "html",
        "published_at": (article.published_at.isoformat()
                         if hasattr(article.published_at, "isoformat")
                         else getattr(article, "published_at", None)),
        "is_primary": False,
        "supplement_provenance": "verified_tls_full_article_html",
    }


def reconcile(source: Path, dest: Path, *, supplement_specs: Path | None = None,
              root: Path | None = None, fetch=None) -> dict:
    """Write a NEW manifest and gate; never replace the immutable old sample."""
    original = json.loads(source.read_text(encoding="utf-8-sig"))
    if (original.get("stage") != 4 or
            original.get("kind") != "stage4-independent-holdout-v1"):
        raise ValueError("Expected a saved, frozen Stage 4 holdout manifest")
    old_rows = original.get("clusters")
    if not isinstance(old_rows, list) or not old_rows:
        raise ValueError("Source manifest contains no clusters")
    if dest.resolve() == source.parent.resolve():
        raise ValueError("Cannot modify the original manifest directory")
    if (dest/"holdout_manifest.json").exists() or (dest/"qwen_holdout.json").exists():
        raise FileExistsError("Target artifacts already exist: choose a fresh out-dir")
    if not source.is_file():
        raise FileNotFoundError(source)
    root = root or Path(__file__).resolve().parents[1]
    prior = _load_prior_from_manifest(original, root)
    pairs = propose_pairs(old_rows)
    byid = {str(row["cluster_id"]): dict(row) for row in old_rows}
    if len(byid) != len(old_rows):
        raise ValueError("Duplicate original cluster IDs")
    used, absorbed, accepted = set(), {}, []
    for pair in pairs:
        x, y = map(str, pair["cluster_ids"])
        if x in used or y in used:
            continue
        # Preserve the original source article and its identity. Only group
        # *additional* sources under the primary existing cluster ID.
        primary, other = byid[x], byid[y]
        primary["source_materials"] = (
            list(primary["source_materials"]) + list(other["source_materials"]))
        primary["holdout_provenance"] = "cross_source_candidate_reconciled"
        primary["event_identity_verified"] = False
        primary["candidate_cluster_ids"] = [pair["cluster_ids"][0],
                                            pair["cluster_ids"][1]]
        absorbed[y] = x
        used.update((x,y))
        accepted.append(pair)
    groups = [row for key,row in byid.items() if key not in absorbed]
    supplemental = []
    specs = []
    if supplement_specs is not None:
        data = json.loads(supplement_specs.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
            raise ValueError("Supplemental JSON must contain articles array")
        specs = data["articles"]
    asof = date.fromisoformat(original["evaluation_date"])
    max_days = int(original["max_age_days"])
    known_urls = {clean_url(m.get("url")) for r in groups
                  for m in r["source_materials"]}
    failures = []
    for index,spec in enumerate(specs):
        if not isinstance(spec, dict):
            failures.append({"index":index,"error":"not_an_object"})
            continue
        target = str(spec.get("target_cluster_id"))
        actual = absorbed.get(target, target)
        parent = next((r for r in groups if str(r["cluster_id"]) == actual), None)
        if parent is None:
            failures.append({"index":index,"error":"target_not_found"})
            continue
        try:
            item = _fetch_new_article(spec, fetch=fetch)
            if item["source"] in {m["source"] for m in parent["source_materials"]}:
                raise ValueError("Not an independent source")
            if urlsplit(item["url"]).hostname in {
                    urlsplit(clean_url(m["url"])).hostname
                    for m in parent["source_materials"]}:
                raise ValueError("Same publishing hostname as existing article")
            reason = new_material(item, prior, asof, max_days)
            if reason:
                raise ValueError("Not new/dated/full HTML: "+reason)
            if clean_url(item["url"]) in known_urls:
                raise ValueError("Same source URL already present")
            # A new page must match the selected event in both its title
            # and first part of the source text. Human review still required.
            options = [m for m in parent["source_materials"]]
            references = [item]+options
            tt = _vectors([m.get("title","") for m in references])
            ll = _vectors([str(m.get("text",""))[:800] for m in references])
            scored = [(_similarity(tt[0],tt[j]),_similarity(ll[0],ll[j]))
                      for j in range(1,len(references))]
            if not any(t >= .135 and l >= .20 for t,l in scored):
                raise ValueError("Insufficient article-to-event matching evidence")
            parent["source_materials"].append(item)
            parent["holdout_provenance"] = "cross_source_candidate_reconciled"
            parent["event_identity_verified"] = False
            known_urls.add(clean_url(item["url"]))
            supplemental.append({
                "target_cluster_id": parent["cluster_id"],
                "source": item["source"], "url": item["url"],
                "title": item["title"], "published_at": item["published_at"],
                "similarities": [[round(t,4),round(l,4)] for t,l in scored],
            })
        except Exception as exc:
            failures.append({"index":index,"url":str(spec.get("url",""))[:300],
                             "error":type(exc).__name__,"reason":str(exc)[:210]})
    original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    # Do not reassign historical IDs; the absorbed source IDs are preserved
    # inside the distinct source_materials records and audit pair mapping.
    selection_stats = dict(original.get("selection_stats") or {})
    selection_stats["multi_source_groups"] = sum(
        len({m["source"] for m in row["source_materials"]}) > 1
        for row in groups)
    selection_stats["recovered_pairs"] = len(accepted)
    selection_stats["recovered_new_sources"] = len(supplemental)
    new_manifest = dict(original)
    new_manifest["clusters"] = groups
    new_manifest["sample_size"] = len(groups)
    new_manifest["minimum_sample_met"] = len(groups) >= original["required_minimum"]
    new_manifest["selection_method"] += "; verified cross-source event candidates"
    new_manifest["selection_stats"] = selection_stats
    new_manifest["reconciliation"] = {
        "version": VERSION, "source_manifest_sha256": original_sha,
        "merged_pair_candidates": accepted, "supplemental_verified_html": supplemental,
        "failed_supplements": failures,
        "identity_not_verified_by_model": True,
        "manual_event_coherence_required": True,
    }
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest/"holdout_manifest.json", new_manifest)
    atomic_json(dest/"multi_event_candidates.json", new_manifest["reconciliation"])
    from scripts.stage4_fresh_qwen_eval import VERSION as GATE_VERSION, check_manifest
    from scripts.stage4_fresh_qwen_eval import _sha
    minimum = original["required_minimum"]
    multi = 5
    gate = check_manifest(new_manifest,minimum=minimum,multi_minimum=multi)
    atomic_json(dest/"qwen_holdout_gate.json",{
        "version": GATE_VERSION,
        "manifest_sha256": _sha(dest/"holdout_manifest.json"),
        "minimum":minimum, "multi_minimum":multi, **gate,
    })
    return {"original_sample_size":len(old_rows), "result_sample_size":len(groups),
            "recovered_pairs":len(accepted),
            "supplemental_sources":len(supplemental),
            "supplemental_errors":failures,
            "new_multi_source_groups":selection_stats["multi_source_groups"],
            "gate":gate, "output":str(dest)}


def reconcile_then_generate(source: Path, dest: Path, *,
                            supplement_specs: Path | None = None,
                            generate_after: bool = False,
                            root: Path | None = None, fetch=None) -> dict:
    """An opt-in single command; a failed sample gate NEVER invokes Qwen."""
    result = reconcile(source, dest, supplement_specs=supplement_specs,
                       root=root, fetch=fetch)
    if generate_after:
        if result["gate"]["ready_for_model_generation"]:
            from scripts.stage4_fresh_qwen_eval import generate
            result["generation"] = generate(out_dir=dest)
        else:
            result["generation"] = {"complete": False,
                                     "reason": "insufficient_verified_multisource_events",
                                     "model_called": False}
    return result


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-manifest",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--supplemental-json",type=Path)
    ap.add_argument("--generate",action="store_true",
                    help="Run local Qwen ONLY if 40 events and 5 sources pass gate")
    args=ap.parse_args()
    result = reconcile_then_generate(
        args.input_manifest, args.out_dir,
        supplement_specs=args.supplemental_json,
        generate_after=args.generate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.generate and not result["gate"]["ready_for_model_generation"]:
        raise SystemExit(2)
    if args.generate and not result["generation"].get("complete"):
        raise SystemExit(4)
