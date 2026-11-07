"""
Read-only diagnostics for the eight re-enabled news sources.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe scripts\\check_sources.py --limit 2

Does not connect to PostgreSQL or publish anything to MAX.
"""
import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collectors.data_collector import collect_news
from processing.topic_filter import is_valid_article

TARGET_SOURCES = {
    "Газпром", "Военное.рф", "ФлотПром", "Отраслевое.рф",
    "ФлотКом", "РИА Новости", "ТАСС", "Интерфакс",
}


def get_configured_sources():
    """Read default_sources from repository.py without database access."""
    tree = ast.parse((ROOT / "database" / "repository.py").read_text(encoding="utf-8"))
    seed = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "seed_sources"
    )
    assignment = next(
        node for node in seed.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "default_sources"
                for target in node.targets)
    )
    return {
        row["name"]: SimpleNamespace(**row)
        for row in ast.literal_eval(assignment.value)
    }


def check_article(article, expected_source):
    errors = []
    if article.source != expected_source:
        errors.append("wrong source")
    if len(article.title.strip()) < 15:
        errors.append("short/empty title")
    parsed = urlparse(article.url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        errors.append("invalid URL")
    if len(article.text.strip()) < 250:
        errors.append("body shorter than 250 chars")
    if not is_valid_article(article):
        errors.append("rejected by topic_filter")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2,
                        help="Maximum full-text articles to fetch per source (default: 2)")
    parser.add_argument("--only", default="",
                        help="One source name; if omitted check all 8")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be >= 1")

    configured = get_configured_sources()
    names = [args.only] if args.only else sorted(TARGET_SOURCES)
    unknown = set(names) - TARGET_SOURCES
    if unknown:
        parser.error("Unknown target source(s): " + ", ".join(sorted(unknown)))

    failures = 0
    for name in names:
        source = configured.get(name)
        if source is None:
            print(f"[FAIL] {name}: missing from seed_sources")
            failures += 1
            continue
        if not source.is_active:
            print(f"[FAIL] {name}: disabled in seed_sources")
            failures += 1
            continue

        print(f"\\n{'=' * 88}\\nCHECK: {name} ({source.source_type})\\n{'=' * 88}")
        try:
            articles = collect_news([source], limit_per_source=args.limit, session=None)
        except Exception as exc:
            print(f"[FAIL] collector exception: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        if not articles:
            print("[FAIL] no article bodies extracted")
            failures += 1
            continue

        valid = 0
        for index, article in enumerate(articles, 1):
            errors = check_article(article, name)
            if not errors:
                valid += 1
            print(f" {index}. {'OK' if not errors else 'CHECK'} | "
                  f"{len(article.text)} chars | date={article.published_at} | "
                  f"method={article.text_source}")
            print(f"    Title: {article.title}")
            print(f"    URL: {article.url}")
            if errors:
                print(f"    Issues: {', '.join(errors)}")
            print(f"    Sample: {article.text[:180]}")

        print(f"RESULT {name}: {valid}/{len(articles)} pass quality checks")
        if valid == 0:
            failures += 1

    print(f"\\nDiagnostics completed: {len(names) - failures}/{len(names)} sources passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
