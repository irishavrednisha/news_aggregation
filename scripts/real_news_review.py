"""Create and finalize a REAL news validation/test review set, without training or publishing.

1. Collect via check_news_collection.py (read-only).
2. python -m scripts.real_news_review prepare --news FILE.json
3. Independently inspect each source and edit review_sheet.csv.
4. python -m scripts.real_news_review finalize --review-dir training/real_review

The previous 125 evaluation URLs are excluded by default using hashes.
Never use this data for training or prompt tuning if you want a valid holdout.
The script NEVER overwrites train.jsonl or calls MAX, Ollama, or PostgreSQL.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict, deque
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from processing.local_news_llm import TAGS, source_payload
from scripts.build_news_sft import build_dataset, example_to_messages


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_HASHES = ROOT / "training" / "prior_eval_url_sha256_80bit.txt"
FIELDS = (
    "story_id", "split", "source", "published_at", "url", "title", "text_excerpt",
    "review_status", "relevant", "tags",
    "evidence_1", "evidence_2", "evidence_3", "evidence_4", "evidence_5",
    "summary", "notes",
)


def normalized_url(url: str) -> str:
    url = (url or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Некорректный URL: {url[:100]}")
    hostname = parsed.hostname.encode("idna").decode("ascii").lower()
    netloc = hostname + ((":" + str(parsed.port)) if parsed.port else "")
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not re.match(r"^(utm_|fbclid$|gclid$|yclid$)", k, re.I)]
    return urlunsplit((
        parsed.scheme.lower(), netloc, parsed.path.rstrip("/"),
        urlencode(sorted(query)), "",
    ))


def url_hash(url: str) -> str:
    return sha256(normalized_url(url).encode("utf-8")).hexdigest()[:20]


def canonical_title(title: str) -> str:
    return " ".join(re.findall(
        r"[a-zа-яё0-9]+", (title or "").lower().replace("ё", "е")
    ))


def load_previous_hashes(path: Path = PREVIOUS_HASHES) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла обязательных исключений: {path}")
    hashes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().lower()
        if not token or token.startswith("#"):
            continue
        if not re.fullmatch(r"[0-9a-f]{20}", token):
            raise ValueError("Повреждён файл исключений")
        hashes.add(token)
    if len(hashes) < 125:
        raise ValueError("Обязательный список из 125 старых URL неполный")
    return hashes


def load_extra_exclusions(paths: list[Path]) -> tuple[set[str], list[str]]:
    hashes, titles = set(), []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            rows = raw.get("articles", raw.get("results", []))
        elif isinstance(raw, list):
            rows = raw
        else:
            raise ValueError(f"Неизвестный формат exclusions: {path}")
        for row in rows:
            if isinstance(row, str):
                hashes.add(url_hash(row))
            elif isinstance(row, dict):
                if row.get("url"):
                    hashes.add(url_hash(row["url"]))
                if row.get("title"):
                    titles.append(canonical_title(row["title"]))
    return hashes, titles


def too_similar(title: str, previous_titles: list[str]) -> bool:
    clean = canonical_title(title)
    if len(clean) < 26:
        return clean in previous_titles
    return any(
        other == clean
        or (len(other) >= 26
            and SequenceMatcher(None, clean, other, autojunk=False).ratio() >= 0.90)
        for other in previous_titles
    )


def select_articles(
    files: list[Path], excluded_hashes: set[str], old_titles: list[str],
    *, max_articles: int = 120, per_source: int = 15, min_text: int = 300,
) -> tuple[list[dict], dict]:
    candidates = []
    dropped = Counter()
    known_urls = set(excluded_hashes)
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        articles = raw.get("articles") if isinstance(raw, dict) else None
        if not isinstance(articles, list):
            raise ValueError(f"Ожидался articles: {path}")
        for item in articles:
            if not isinstance(item, dict):
                dropped["not_dict"] += 1
                continue
            try:
                h = url_hash(item.get("url", ""))
            except ValueError:
                dropped["invalid_url"] += 1
                continue
            if h in known_urls:
                dropped["previous_eval_or_duplicate_url"] += 1
                continue
            known_urls.add(h)
            title = str(item.get("title") or "").strip()
            article = str(item.get("text") or "").strip()
            if not title or len(article) < min_text:
                dropped["short_or_empty"] += 1
                continue
            if item.get("unverified_tls"):
                dropped["unverified_tls"] += 1
                continue
            if too_similar(title, old_titles):
                dropped["near_old_title"] += 1
                continue
            candidates.append({
                "story_id": "real-" + h,
                "origin": "real",
                "url": item["url"],
                "title": title,
                "text": article,
                "source": str(item.get("source") or "Неизвестен"),
                "published_at": str(item.get("published_at") or ""),
            })

    # Prefer recent articles and distribute sources evenly.
    candidates.sort(
        key=lambda x: (x["published_at"], x["story_id"]),
        reverse=True,
    )
    pools = defaultdict(deque)
    local_titles = []
    for item in candidates:
        if too_similar(item["title"], local_titles):
            dropped["similar_title_in_new_batch"] += 1
            continue
        if len(pools[item["source"]]) >= per_source:
            dropped["per_source_limit"] += 1
            continue
        local_titles.append(canonical_title(item["title"]))
        pools[item["source"]].append(item)

    selected = []
    # Source round-robin helps avoid 100 articles from a single website.
    while pools and len(selected) < max_articles:
        for src in sorted(pools, key=lambda name: (-len(pools[name]), name)):
            if not pools.get(src):
                continue
            selected.append(pools[src].popleft())
            if len(selected) >= max_articles:
                break
        pools = {k: v for k, v in pools.items() if v}

    # Freeze the split BEFORE anyone sees model predictions.
    # 60% validation, 40% final test; each item remains in one fixed split.
    for item in selected:
        item["split"] = (
            "validation" if int(url_hash(item["url"]), 16) % 5 < 3 else "test"
        )
    return selected, dict(dropped)


def prepare(
    news: list[Path], exclusion_paths: list[Path], outdir: Path, *,
    max_articles: int = 120, per_source: int = 15,
) -> dict:
    excluded = load_previous_hashes()
    extra, old_titles = load_extra_exclusions(exclusion_paths)
    excluded |= extra
    chosen, dropped = select_articles(
        news, excluded, old_titles,
        max_articles=max_articles, per_source=per_source,
    )
    index_path = outdir / "source_index.json"
    csv_path = outdir / "review_sheet.csv"
    # A new collection must never erase human-reviewed labels from a prior run.
    # To combine multiple collections, pass all JSON paths to --news at once.
    if index_path.exists() or csv_path.exists():
        raise FileExistsError(
            f"В {outdir} уже есть данные разметки. "
            "Новый prepare может стереть ручные правки. "
            "Сохраните текущую разметку и укажите новый --out "
            "или передайте все новости одной командой до первой разметки."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({
        "read_only": True,
        "human_verified_count": 0,
        "excluded_prior_eval_hashes": len(excluded),
        "articles": chosen,
        "dropped": dropped,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter=";")
        w.writeheader()
        for a in chosen:
            w.writerow({
                "story_id": a["story_id"],
                "split": a["split"],
                "source": a["source"],
                "published_at": a["published_at"],
                "url": a["url"],
                "title": a["title"],
                "text_excerpt": re.sub(r"\s+", " ", a["text"])[:400],
                "review_status": "draft",
                "relevant": "",
                "tags": "",
                "summary": "",
                "notes": "",
            })
    return {
        "candidates": len(chosen),
        "by_split": dict(Counter(a["split"] for a in chosen)),
        "dropped": dropped,
        "index": str(index_path),
        "review": str(csv_path),
        "warning": "No model-generated labels are verified; manual editorial review required",
    }


def parse_relevance(value: str) -> bool:
    v = (value or "").strip().casefold()
    if v in {"1", "true", "yes", "да", "y"}:
        return True
    if v in {"0", "false", "no", "нет", "n"}:
        return False
    raise ValueError("В колонке relevant требуется yes/no (или да/нет)")


def parse_tags(raw: str) -> list[str]:
    tags = [x.strip() for x in (raw or "").split("|") if x.strip()]
    if any(t not in TAGS for t in tags):
        raise ValueError(f"Неизвестный тег: {tags}")
    return tags


def finalize(review_dir: Path, output_dir: Path, *,
             prior_hashes: set[str] | None = None) -> dict:
    index = json.loads((review_dir / "source_index.json").read_text(encoding="utf-8"))
    articles = {r["story_id"]: r for r in index["articles"]}
    if len(articles) != len(index["articles"]):
        raise ValueError("Повтор story_id в исходном индексе")
    excluded = prior_hashes if prior_hashes is not None else load_previous_hashes()
    validated = []
    rows_seen = set()
    skipped = Counter()
    with (review_dir / "review_sheet.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as fh:
        for n, row in enumerate(csv.DictReader(fh, delimiter=";"), start=2):
            sid = str(row.get("story_id") or "").strip()
            if sid not in articles or sid in rows_seen:
                raise ValueError(f"Строка {n}: незнакомый/повторный story_id")
            rows_seen.add(sid)
            a = articles[sid]
            if row.get("split") != a["split"] or row.get("url") != a["url"]:
                raise ValueError(f"Строка {n}: нельзя менять split или URL")
            if url_hash(a["url"]) in excluded:
                raise ValueError(f"Строка {n}: старый URL из прошлых экспериментов")
            status = str(row.get("review_status") or "").strip().lower()
            if status != "verified":
                skipped["not_verified"] += 1
                continue
            try:
                relevant = parse_relevance(row.get("relevant", ""))
                tags = parse_tags(row.get("tags", ""))
                evidence = [
                    row[f"evidence_{i}"].strip()
                    for i in range(1, 6) if (row.get(f"evidence_{i}") or "").strip()
                ]
                summary = str(row.get("summary") or "").strip()
                if relevant and not (100 <= len(summary) <= 550):
                    raise ValueError("Для профильной статьи нужна сводка 100–550 символов")
                example = {
                    "story_id": sid,
                    "origin": "real",
                    "split": a["split"],
                    "review_status": "verified",
                    "title": a["title"],
                    "text": a["text"],
                    "relevant": relevant,
                    "tags": tags,
                    "evidence": evidence,
                    "summary": summary,
                    "url": a["url"],
                    "source": a["source"],
                }
                example_to_messages(example)  # Same validator as SFT training
                validated.append(example)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Строка {n} ({sid}): {exc}") from exc
    if rows_seen != articles.keys():
        raise ValueError("Не хватает строк из исходного review_sheet.csv")
    if not validated:
        raise ValueError("Нет вручную подтверждённых записей: review_status=verified")
    data = build_dataset(validated)  # only verified, real and fixed split
    if data["splits"]["train"]:
        raise ValueError("Случайное попадание реальных holdout в обучение")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Only validation/test are touched; existing 6700 train messages remain.
    for split in ("validation", "test"):
        p = output_dir / (split + ".jsonl")
        with p.open("w", encoding="utf-8") as fh:
            for record in data["splits"][split]:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "real_gold_verified.json").write_text(
        json.dumps({"examples": validated}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "verified_articles": len(validated),
        "skipped_drafts": skipped["not_verified"],
        "validation_messages": len(data["splits"]["validation"]),
        "test_messages": len(data["splits"]["test"]),
        "train_touched": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Подготовка независимой ручной разметки")
    sub = ap.add_subparsers(dest="action", required=True)
    first = sub.add_parser("prepare", help="Создать CSV разметки новых новостей")
    first.add_argument("--news", required=True, nargs="+", type=Path)
    first.add_argument("--exclude", action="append", default=[], type=Path)
    first.add_argument("--out", type=Path, default=Path("training/real_review"))
    first.add_argument("--max", type=int, default=120)
    first.add_argument("--per-source", type=int, default=15)
    last = sub.add_parser("finalize", help="Собрать validated real validation/test")
    last.add_argument("--review-dir", type=Path, default=Path("training/real_review"))
    last.add_argument("--output-dir", type=Path, default=Path("training/news_sft"))
    a = ap.parse_args()
    if a.action == "prepare":
        if not 1 <= a.max <= 500 or not 1 <= a.per_source <= 200:
            ap.error("--max и --per-source должны быть положительными")
        result = prepare(a.news, a.exclude, a.out,
                         max_articles=a.max, per_source=a.per_source)
    else:
        result = finalize(a.review_dir, a.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
