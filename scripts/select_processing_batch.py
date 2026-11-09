"""Reproducible read-only diagnostic batch for Qwen and SBERT.

Draws from BOTH accepted and rejected first-stage news to expose potential
false negatives. Not gold labels, not an unbiased quality metric. Retains
the user's manually chosen TLS collection sources and their warnings.

Example:
 python -m scripts.select_processing_batch collector_test_results/news_fixed.json \
   collector_test_results/processing_audit_v2.json \
   --output collector_test_results/model_batch_v2.json --limit 40 --accepted 24
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import re


def _title_key(title: str) -> str:
    return " ".join(
        re.findall(r"[a-zа-я0-9]+", (title or "").casefold().replace("ё", "е"))
    )


def _round_robin(rows: list[dict], *, taken_urls: set[str],
                 taken_titles: set[str], limit: int) -> list[dict]:
    """Stable per-source diversity without model-generated selection."""
    grouped: dict[str, deque] = defaultdict(deque)
    # Hashes avoid privileging latest headlines of a particular source.
    for row in sorted(
        rows,
        key=lambda r: (
            str(r.get("source") or ""),
            sha256(str(r["url"]).encode("utf-8")).hexdigest(),
        ),
    ):
        grouped[str(row.get("source") or "")].append(row)
    selected = []
    while grouped and len(selected) < limit:
        for source in sorted(grouped):
            pool = grouped[source]
            while pool:
                article = pool.popleft()
                url = str(article["url"])
                title = _title_key(str(article.get("title") or ""))
                if url in taken_urls or not title or title in taken_titles:
                    continue
                taken_urls.add(url)
                taken_titles.add(title)
                selected.append(article)
                break
            if len(selected) == limit:
                break
        grouped = {src: pool for src, pool in grouped.items() if pool}
    return selected


def create_batch(
    news: dict, audit: dict, *, limit: int = 40, accepted: int = 24,
) -> dict:
    if not 1 <= limit <= 500 or not 0 <= accepted <= limit:
        raise ValueError("Проверьте --limit и --accepted")
    articles = news.get("articles")
    decisions = audit.get("decisions")
    if not isinstance(articles, list) or not isinstance(decisions, list):
        raise ValueError("Ожидались news.articles и audit.decisions")
    if len(articles) != len(decisions):
        raise ValueError("Длины сырого сбора и аудита не совпадают")
    decisions_by_url = {}
    for item in decisions:
        url = str(item.get("url") or "")
        if not url or url in decisions_by_url:
            raise ValueError("Повторный/пустой URL в аудите")
        if type(item.get("accepted_by_primary_filter")) is not bool:
            raise ValueError("Не найден результат фильтра для URL " + url)
        decisions_by_url[url] = item
    raw_urls = [str(x.get("url") or "") for x in articles]
    if len(set(raw_urls)) != len(raw_urls) or set(raw_urls) != set(decisions_by_url):
        raise ValueError("Состав URL в аудите отличается от исходного сбора")

    pools = {True: [], False: []}
    for item in articles:
        verdict = decisions_by_url[item["url"]]
        # Use only real textual articles; never turn broken collection
        # into a training/evaluation label.
        if not str(item.get("title") or "").strip() or not str(item.get("text") or "").strip():
            continue
        pools[verdict["accepted_by_primary_filter"]].append(item)

    taken_urls: set[str] = set()
    taken_titles: set[str] = set()
    chosen_accept = _round_robin(
        pools[True], taken_urls=taken_urls,
        taken_titles=taken_titles, limit=accepted,
    )
    chosen_reject = _round_robin(
        pools[False], taken_urls=taken_urls,
        taken_titles=taken_titles, limit=limit - len(chosen_accept),
    )
    # Fill underpopulated strata without duplicating articles or titles.
    if len(chosen_accept) + len(chosen_reject) < limit:
        chosen_accept.extend(_round_robin(
            pools[True], taken_urls=taken_urls, taken_titles=taken_titles,
            limit=limit - len(chosen_accept) - len(chosen_reject),
        ))
    if len(chosen_accept) + len(chosen_reject) < limit:
        chosen_reject.extend(_round_robin(
            pools[False], taken_urls=taken_urls, taken_titles=taken_titles,
            limit=limit - len(chosen_accept) - len(chosen_reject),
        ))

    # Interleave pass/fail so --limit=N gives a useful mixed prefix in
    # both check_focused_tagging and check_local_news_llm.
    selected = []
    for i in range(max(len(chosen_accept), len(chosen_reject))):
        for group in (chosen_accept, chosen_reject):
            if i < len(group):
                article = dict(group[i])
                d = decisions_by_url[article["url"]]
                article["primary_filter_accepted"] = d["accepted_by_primary_filter"]
                article["primary_filter_reason"] = d["filter_reason"]
                selected.append(article)
    return {
        "purpose": "exploratory_model_disagreement_audit",
        "read_only": True,
        "gold_labels_provided": False,
        "human_verified_count": 0,
        "training_performed": False,
        "publishing_performed": False,
        "source_collected_at": news.get("collected_at"),
        "source_audit_as_of": audit.get("as_of"),
        "selection": {
            "requested": limit,
            "selected": len(selected),
            "primary_filter_accepted": len(chosen_accept),
            "primary_filter_rejected": len(chosen_reject),
            "unverified_tls": sum(r.get("unverified_tls") is True for r in selected),
            "sources": dict(sorted(Counter(
                str(r.get("source") or "") for r in selected
            ).items())),
        },
        "articles": selected,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Смешанная выборка для проверки Qwen и SBERT")
    p.add_argument("news_json", type=Path)
    p.add_argument("audit_json", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--accepted", type=int, default=24,
                   help="Целевое число статей, прошедших первый фильтр")
    args = p.parse_args()
    if args.output.resolve() in (
        args.news_json.resolve(), args.audit_json.resolve()
    ):
        p.error("Входной файл нельзя перезаписывать")
    news = json.loads(args.news_json.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    payload = create_batch(news, audit, limit=args.limit, accepted=args.accepted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["selection"], ensure_ascii=False, indent=2))
    print(f"Диагностическая выборка: {args.output}")
    print("Это НЕ эталон и НЕ оценка точности. Проверяйте решения вручную.")


if __name__ == "__main__":
    main()
