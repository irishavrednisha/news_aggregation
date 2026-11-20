"""Read-only independent Stage 4 holdout from NEW real full-HTML news.

Frozen v5 JSON is an exclusion set, never reused for evaluation. Local
collector session=None; no DB writes, no MAX, no insecure TLS. Gemma runs only
if enough unseen source documents are available. Blind human labels are
separate from model verdicts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "stage4-independent-holdout-v1"
WORDS = re.compile(r"[а-яёa-z0-9]+", re.I)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def clean_url(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    query = [(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
             if not key.lower().startswith(("utm_", "pk_"))
             and key.lower() not in {"gclid", "fbclid", "yclid", "ref", "from", "source"}]
    return urlunsplit(("https", parsed.netloc.lower().removeprefix("www."),
                       parsed.path.rstrip("/") or "/", urlencode(sorted(query)), ""))


def tokens(text: str) -> set[str]:
    return {t.replace("ё", "е") for t in WORDS.findall(str(text or "").lower())
            if len(t) >= 3}


def title_duplicate(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", a))
    b_numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", b))
    if a_numbers and b_numbers and a_numbers != b_numbers:
        return False  # Different factory number or output scale may be new event.
    aa, bb = tokens(a), tokens(b)
    if not aa or not bb:
        return False
    jaccard = len(aa & bb) / len(aa | bb)
    return jaccard >= 0.79 or SequenceMatcher(
        None, " ".join(sorted(aa)), " ".join(sorted(bb))).ratio() >= 0.87


def date_of(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def fingerprint(material: dict) -> str:
    text = " ".join(WORDS.findall(str(material.get("text") or "").casefold()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def prior_signatures(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("Previous Stage 4 report required to exclude old examples")
    prior = {"urls": set(), "titles": [], "texts": set(), "source_files": []}
    for path in paths:
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get("stage") != 4 or not isinstance(report.get("clusters"), list):
            raise ValueError(f"Not a Stage 4 report: {path}")
        for row in report["clusters"]:
            for material in row.get("source_materials", []):
                url = clean_url(material.get("url", ""))
                if url:
                    prior["urls"].add(url)
                title = str(material.get("title") or "").strip()
                if title:
                    prior["titles"].append(title)
                sha = fingerprint(material)
                if sha:
                    prior["texts"].add(sha)
        prior["source_files"].append(
            {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    prior["titles"] = list(dict.fromkeys(prior["titles"]))
    return prior


def new_material(material: dict, prior: dict, now: date, max_days: int,
                 *, allow_undated: bool = False) -> str:
    url = clean_url(material.get("url", ""))
    title = str(material.get("title") or "").strip()
    body = str(material.get("text") or "").strip()
    if not url or len(title) < 18 or len(body) < 350:
        return "invalid_or_short_source"
    if material.get("text_method") not in (None, "html"):
        return "rss_preview_not_full_html"
    # Check prior exposure BEFORE missing/stale dates: a previously seen
    # article is never an "undated new candidate". This makes the rejection
    # counts diagnostically meaningful without loosening the freshness gate.
    if url in prior["urls"]:
        return "seen_url"
    if fingerprint(material) in prior["texts"]:
        return "seen_text"
    if any(title_duplicate(title, old) for old in prior["titles"]):
        return "seen_or_near_duplicate_event"
    published = date_of(material.get("published_at"))
    if published is None and not allow_undated:
        return "no_verified_date"
    if published is not None and not 0 <= (now - published).days <= max_days:
        return "not_recent"
    return ""


def load_db(max_clusters: int = 800):
    from database.db import get_session
    from database.models import News
    from scripts.run_stage4_summaries import _materials
    db = get_session()
    try:
        ids = [r[0] for r in
               db.query(News.cluster_id).filter(News.cluster_id.isnot(None))
               .distinct().order_by(News.cluster_id.desc()).limit(max_clusters).all()]
        return [_materials(db, cluster_id) for cluster_id in ids], []
    finally:
        db.close()


def load_live(limit_per_source: int = 10, *, from_config: bool = False):
    from collectors.data_collector import collect_news
    from scripts.diagnose_all_sources import load_sources, load_configured_sources
    sources = load_configured_sources() if from_config else load_sources()
    groups, warnings = [], []
    for source in sorted(sources, key=lambda x: x.name):
        if not source.is_active:
            continue
        try:
            articles = collect_news([source], limit_per_source=limit_per_source,
                                    session=None, apply_filters=False,
                                    allow_unverified_tls=False)
            for idx, article in enumerate(articles):
                groups.append([{
                    "id": f"live:{source.name}:{idx}", "source": article.source,
                    "url": article.url, "title": article.title, "text": article.text,
                    "text_method": article.text_source, "is_primary": True,
                    "published_at": (article.published_at.isoformat()
                                     if hasattr(article.published_at, "isoformat") else None),
                }])
            print(f"[live] {source.name}: {len(articles)} candidates", flush=True)
        except Exception as exc:
            warnings.append(f"{source.name}: {type(exc).__name__}: {str(exc)[:130]}")
    return groups, warnings


def load_diagnostics(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report.get("results"), list):
        raise ValueError("Not diagnose_all_sources JSON")
    groups, warnings = [], []
    for source in report["results"]:
        for i, article in enumerate(source.get("articles") or []):
            if not article.get("text"):
                warnings.append(f"{source.get('name')}: needs --include-text")
                continue
            groups.append([{
                "id": f"diagnostic:{source.get('name')}:{i}",
                "source": source.get("name") or "", "title": article.get("title"),
                "url": article.get("url"), "text": article.get("text"),
                "text_method": article.get("text_method"), "is_primary": True,
                "published_at": article.get("published_at"),
            }])
    return groups, warnings


def recover_rostec_article_dates(groups: list[list[dict]], previous: dict,
                                 now: date, max_days: int,
                                 *, fetch_article=None) -> tuple[list[list[dict]], dict]:
    """Recheck only unseen, undated Rostec records against their official HTML.

    This is strictly read-only.  Never infer a date from the text or a related
    article, invent a timestamp, or rewrite PostgreSQL or the article body.
    Failures leave the source undated and ineligible for the holdout.
    """
    from urllib.parse import urlsplit

    if fetch_article is None:
        from collectors.data_collector import build_full_article
        fetch_article = build_full_article
    from collectors.data_collector import ArticlePreview
    recovered, failures = [], []
    attempted = 0
    updated = []
    for materials in groups:
        new_materials = []
        for m in materials:
            entry = dict(m)
            url = str(entry.get("url") or "")
            host = (urlsplit(url).hostname or "").lower()
            if (entry.get("source") == "Ростех"
                    and host in {"rostec.ru", "www.rostec.ru"}
                    and not date_of(entry.get("published_at"))
                    and clean_url(url) not in previous["urls"]
                    and fingerprint(entry) not in previous["texts"]
                    and not any(title_duplicate(str(entry.get("title") or ""), old)
                                for old in previous["titles"])
                    and len(str(entry.get("text") or "")) >= 350
                    and len(str(entry.get("title") or "")) >= 18):
                attempted += 1
                try:
                    preview = ArticlePreview(
                        source="Ростех", title=str(entry["title"]),
                        url=url, published_at=None, preview_text="")
                    article = fetch_article(preview)
                    date = date_of(getattr(article, "published_at", None))
                    # The page must describe the same article. The collector
                    # may otherwise have parsed a recommendation/sidebar.
                    if (date and 0 <= (now - date).days <= max_days
                            and title_duplicate(
                                entry["title"], str(getattr(article, "title", "")))):
                        entry["published_at"] = (
                            article.published_at.isoformat()
                            if hasattr(article.published_at, "isoformat")
                            else str(article.published_at))
                        entry["date_provenance"] = "official_rostec_article_page"
                        recovered.append({"url": url, "published_at": entry["published_at"]})
                    else:
                        failures.append({"url": url, "reason": (
                            "date_or_title_unverified_or_outside_window")})
                except Exception as exc:
                    failures.append({"url": url, "reason": type(exc).__name__})
            new_materials.append(entry)
        updated.append(new_materials)
    return updated, {
        "attempted": attempted, "recovered": len(recovered),
        "verified": recovered, "unverified": failures,
        "note": ("Unverified dates are NOT accepted. The article page was "
                 "requested with the collector's normal verified TLS policy."),
    }


def select_holdout(groups: list[list[dict]], previous: dict, now: date,
                   *, target: int = 100, max_days: int = 30,
                   max_per_source: int = 15, allow_undated: bool = False):
    excluded, eligible = Counter(), []
    candidate_sources, rejection_by_source = Counter(), {}
    rejection_examples = []
    example_count_per_source = Counter()
    for materials in groups:
        if not materials:
            excluded["empty_group"] += 1
            continue
        site = str(materials[0].get("source") or "unknown")
        candidate_sources[site] += 1
        issues = [new_material(m, previous, now, max_days,
                               allow_undated=allow_undated) for m in materials]
        if any(issues):
            reason = next(issue for issue in issues if issue)
            excluded[reason] += 1
            if site not in rejection_by_source:
                rejection_by_source[site] = Counter()
            rejection_by_source[site][reason] += 1
            if (reason == "no_verified_date" and len(rejection_examples) < 40
                    and example_count_per_source[site] < 2):
                # Minimal audit metadata, no scraped article bodies or guesses.
                source = next((m for m, issue in zip(materials, issues)
                               if issue), materials[0])
                example_count_per_source[site] += 1
                rejection_examples.append({
                    "source": site, "title": str(source.get("title") or "")[:180],
                    "url": str(source.get("url") or "")[:300],
                    "published_at": source.get("published_at"),
                    "text_method": source.get("text_method"),
                    "text_chars": len(str(source.get("text") or "")),
                    "reason": reason,
                })
            continue
        site = str(materials[0].get("source") or "")
        different_sites = len({str(m.get("source") or "") for m in materials})
        recent = max((date_of(m.get("published_at")) or date.min
                      for m in materials))
        eligible.append((materials, site, different_sites > 1, recent))
    # Prioritize actual Stage 3 multi-document clusters, never invent grouping.
    eligible.sort(key=lambda r: (-int(r[2]), -r[3].toordinal(), r[1],
                                 clean_url(r[0][0].get("url", ""))))
    selected, counts = [], Counter()
    seen_urls, seen_titles, seen_text = set(), [], set()
    for materials, site, multi, _ in eligible:
        if counts[site] >= max_per_source:
            excluded["source_cap"] += 1
            continue
        if any(clean_url(m.get("url", "")) in seen_urls
               or fingerprint(m) in seen_text
               or any(title_duplicate(m.get("title", ""), t) for t in seen_titles)
               for m in materials):
            excluded["duplicate_within_holdout"] += 1
            continue
        selected.append({
            "cluster_id": 900000 + len(selected) + 1,
            "source_materials": materials,
            "holdout_provenance": ("database_event_cluster" if
                all(str(m.get("id", "")).isdigit() for m in materials)
                else "new_single_source_article"),
            "event_identity_verified": False,
        })
        counts[site] += 1
        for material in materials:
            seen_urls.add(clean_url(material.get("url", "")))
            seen_text.add(fingerprint(material))
            seen_titles.append(str(material.get("title", "")))
        if len(selected) == target:
            break
    stats = {
        "rejections": dict(excluded), "eligible_before_deduplication": len(eligible),
        "candidate_input_groups": len(groups),
        "candidate_source_counts": dict(candidate_sources),
        "rejections_by_source": {
            source: dict(reasons) for source, reasons
            in sorted(rejection_by_source.items())
        },
        "undated_unseen_examples": rejection_examples,
        "note": ("Overlap screening precedes the date check. An undated "
                 "candidate here is not known to be new in time: manual "
                 "publication-date evidence is still required."),
        "source_counts": dict(counts),
        "multi_source_groups": sum(
            len({m.get("source") for m in x["source_materials"]}) > 1 for x in selected),
        "missing_publication_dates": sum(
            not date_of(m.get("published_at"))
            for x in selected for m in x["source_materials"]),
    }
    return selected, stats


def build_manifest(selected, stats, *, target, min_sample, mode, previous,
                   now, errors, max_days):
    return {
        "stage": 4, "kind": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_date": now.isoformat(),
        "read_only": True, "changed_legacy_news_or_posts": False,
        "publisher_called": False, "source_input_mode": mode,
        "previous_files": previous["source_files"], "target": target,
        "required_minimum": min_sample, "minimum_sample_met": len(selected) >= min_sample,
        "sample_size": len(selected), "max_age_days": max_days,
        "selection_method": "Exclude old URL, exact source body, close-title overlap; fresh HTML, source cap",
        "independence_limitations": [
            "Title-based screening cannot detect every cross-source event paraphrase.",
            "Single-source items do not prove multi-document summarization quality.",
            "Text length alone does not establish correctness or completeness.",
        ],
        "selection_stats": stats, "collection_errors": errors, "clusters": selected,
    }


def blind_review_files(report: dict, directory: Path) -> None:
    """Output fresh sources and candidate summaries, never automatic verdicts."""
    import csv
    rows = list(report.get("clusters") or [])
    random.Random(44206).shuffle(rows)
    packet = directory / "blind_review.jsonl"
    labels = directory / "review_labels.csv"
    tmp = packet.with_name(packet.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as output, labels.open(
            "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["review_id", "faithful_facts", "natural_russian",
                         "main_event_covered", "meaningful_news_event",
                         "source_quality", "error_details"])
        for i, row in enumerate(rows, 1):
            rid = f"B{i:03d}"
            docs = row.get("source_materials") or []
            record = {
                "review_id": rid,
                "title": docs[0].get("title") if docs else "",
                "source_documents": [
                    {k: m.get(k) for k in
                     ("source", "title", "url", "text", "published_at")}
                    for m in docs],
                "gemma_summary": row.get("gemma_summary") or "",
                "evaluation_date": report.get("evaluation_date"),
                "review_instructions": (
                    "Verify every statement independently against the source. "
                    "No generated text: use NA in every rating. Mark corrupt "
                    "or insufficient source as source_quality=fail."),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow([rid, "", "", "", "", "", ""])
    os.replace(tmp, packet)
    # Keep this key private; share only JSONL and CSV with human reviewers.
    atomic_json(directory / "private_review_key.json", {
        f"B{i:03d}": {"cluster_id": row["cluster_id"],
                      "automatic_status": row["status"]}
        for i, row in enumerate(rows, 1)})


def run_holdout(*, previous_reports, out_dir, target, minimum, max_days,
                source_limit, source_cap, mode, diagnostics=None,
                allow_undated=False, from_config=False,
                collect_only=False, resume=False, model="gemma3:4b",
                recover_missing_rostec_dates=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    known = prior_signatures(previous_reports)
    manifest_path = out_dir / "holdout_manifest.json"
    if resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("kind") != VERSION or
                manifest.get("previous_files") != known["source_files"]):
            raise ValueError("Cannot resume: exclusion fingerprints changed")
    else:
        now = datetime.now(timezone.utc).date()
        groups, warnings = [], []
        if mode in ("both", "database"):
            try:
                result, errors = load_db()
                groups.extend(result)
                warnings.extend(errors)
            except Exception as exc:
                if mode == "database":
                    raise
                warnings.append("db_unavailable:" + type(exc).__name__ +
                                ":" + str(exc)[:120])
        if mode in ("both", "live"):
            result, errors = load_live(source_limit, from_config=from_config)
            groups.extend(result)
            warnings.extend(errors)
        if mode == "diagnostics":
            if diagnostics is None:
                raise ValueError("Specify --diagnostics-json")
            result, errors = load_diagnostics(diagnostics)
            groups.extend(result)
            warnings.extend(errors)
        date_recovery = None
        if recover_missing_rostec_dates:
            groups, date_recovery = recover_rostec_article_dates(
                groups, known, now, max_days)
        selected, stats = select_holdout(
            groups, known, now, target=target, max_days=max_days,
            max_per_source=source_cap, allow_undated=allow_undated)
        if date_recovery is not None:
            stats["rostec_date_recovery"] = date_recovery
        manifest = build_manifest(
            selected, stats, target=target, min_sample=minimum,
            mode=mode, previous=known, now=now, errors=warnings,
            max_days=max_days)
        if manifest_path.exists():
            raise FileExistsError("Manifest exists: use --resume or new directory")
        atomic_json(manifest_path, manifest)
    print(json.dumps({k: manifest.get(k) for k in
                      ("sample_size", "target", "required_minimum",
                       "minimum_sample_met", "selection_stats",
                       "collection_errors")}, ensure_ascii=False, indent=2),
          flush=True)
    if not manifest["minimum_sample_met"] or collect_only:
        return manifest
    from scripts.run_stage4_summaries import run
    report_path = out_dir / "gemma_holdout.json"
    report = run(
        limit=len(manifest["clusters"]), min_clusters=minimum,
        model=model, max_retries=2, output=report_path, persist=False,
        resume=resume and report_path.is_file(), input_json=manifest_path)
    report["source_input_mode"] = "frozen_new_independent_holdout_json"
    report["holdout_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()).hexdigest()
    atomic_json(report_path, report)
    # Never destroy a partially completed human review on --resume.
    packet = out_dir / "blind_review.jsonl"
    labels = out_dir / "review_labels.csv"
    key_path = out_dir / "private_review_key.json"
    if all(p.is_file() for p in (packet, labels, key_path)):
        key = json.loads(key_path.read_text(encoding="utf-8"))
        expected = {(str(row["cluster_id"]), row["status"])
                    for row in report["clusters"]}
        existing = {(str(item["cluster_id"]), item["automatic_status"])
                    for item in key.values()}
        if expected != existing:
            raise ValueError("Cannot reuse blind labels for a changed sample")
        print("Existing human review labels preserved; no overwrite.", flush=True)
    elif any(p.exists() for p in (packet, labels, key_path)):
        raise FileExistsError("Incomplete review packet exists; back it up first")
    else:
        blind_review_files(report, out_dir)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exclude", action="append", required=True, type=Path,
                   help="Prior Stage 4 JSON used ONLY for leakage screening")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "collector_test_results" / "stage4_independent_holdout")
    p.add_argument("--mode", choices=("both", "live", "database", "diagnostics"),
                   default="both")
    p.add_argument("--diagnostics-json", type=Path)
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--minimum", type=int, default=40)
    p.add_argument("--max-age-days", type=int, default=30)
    p.add_argument("--limit-per-source", type=int, default=10)
    p.add_argument("--max-per-source", type=int, default=15)
    p.add_argument("--recover-missing-rostec-dates", action="store_true",
                   help="Refetch unseen undated Rostec article pages with "
                        "verified TLS; never guess or write dates to DB")
    p.add_argument("--allow-undated", action="store_true",
                   help="Diagnostic only; sample freshness is then not verified")
    p.add_argument("--from-config", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if (args.minimum < 1 or args.target < args.minimum or args.max_age_days < 0
            or not 1 <= args.limit_per_source <= 15 or args.max_per_source < 1):
        p.error("Require target >= minimum > 0; bounded source and age limits")
    if args.mode == "diagnostics" and not args.diagnostics_json:
        p.error("--diagnostics-json required for diagnostics input")
    report = run_holdout(
        previous_reports=args.exclude, out_dir=args.out_dir, target=args.target,
        minimum=args.minimum, max_days=args.max_age_days,
        source_limit=args.limit_per_source, source_cap=args.max_per_source,
        mode=args.mode, diagnostics=args.diagnostics_json,
        allow_undated=args.allow_undated, from_config=args.from_config,
        collect_only=args.collect_only, resume=args.resume,
        recover_missing_rostec_dates=args.recover_missing_rostec_dates)
    print(f"Output: {args.out_dir.resolve()}")
    if not report.get("minimum_sample_met", True):
        print("INSUFFICIENT UNSEEN DATED ARTICLES; no accuracy claim.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
