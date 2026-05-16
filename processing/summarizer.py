import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import re
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database.models import Cluster, News
from utils.text_utils import normalize_article_text, normalize_title

def split_into_sentences(text: str) -> list[str]:
    if not text:
        return []

    text = normalize_article_text(text)
    text = re.sub(r"\s+", " ", text).strip()

    abbreviations = {
        "г.": "г<dot>",
        "гг.": "гг<dot>",
        "им.": "им<dot>",
        "т.д.": "т<dot>д<dot>",
        "т.п.": "т<dot>п<dot>",
        "т.е.": "т<dot>е<dot>",
        "руб.": "руб<dot>",
        "млн.": "млн<dot>",
        "млрд.": "млрд<dot>",
    }

    for old, new in abbreviations.items():
        text = text.replace(old, new)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    result = []

    bad_phrases = [
        "читайте также",
        "материалы по теме",
        "другие материалы",
        "главное в отраслевых сми",
        "подписывайтесь",
        "печатная версия",
        "источник новости",
        "фото:",
        "реклама",
    ]

    for sentence in sentences:
        sentence = sentence.replace("<dot>", ".").strip()

        if len(sentence) < 45:
            continue

        if len(sentence) > 420:
            continue

        lowered = sentence.lower()

        if any(phrase in lowered for phrase in bad_phrases):
            continue

        result.append(sentence)

    return result


def normalize_sentence_key(sentence: str) -> str:
    sentence = sentence.lower()
    sentence = re.sub(r"[^а-яa-z0-9 ]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def remove_duplicate_sentences(sentences: list[str]) -> list[str]:
    seen = set()
    unique = []

    for sentence in sentences:
        key = normalize_sentence_key(sentence)

        if key in seen:
            continue

        seen.add(key)
        unique.append(sentence)

    return unique


def build_textrank_summary(
    text: str,
    title: str | None = None,
    sentence_count: int = 4,
    max_chars: int = 800
) -> str:
    sentences = split_into_sentences(text)
    sentences = remove_title_like_sentences(sentences, title)
    sentences = remove_duplicate_sentences(sentences)

    if not sentences:
        return ""

    if len(sentences) <= sentence_count:
        summary = " ".join(sentences)
        return limit_summary(summary, max_chars=max_chars)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_df=0.85,
        min_df=1,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(
        ((scores[i], i, sentence) for i, sentence in enumerate(sentences)),
        reverse=True
    )

    selected = []
    selected_keys = []

    for score, index, sentence in ranked_sentences:
        key = normalize_sentence_key(sentence)

        too_similar = False

        for selected_key in selected_keys:
            if key in selected_key or selected_key in key:
                too_similar = True
                break

        if too_similar:
            continue

        selected.append((score, index, sentence))
        selected_keys.append(key)

        if len(selected) >= sentence_count:
            break

    selected_sorted = sorted(selected, key=lambda item: item[1])
    selected_sentences = [item[2] for item in selected_sorted]

    summary = fit_summary_by_sentences(
        selected_sentences=selected_sentences,
        max_chars=max_chars
    )

    return summary

def limit_summary(summary: str, max_chars: int = 800) -> str:
    if len(summary) <= max_chars:
        return summary

    sentences = re.split(r"(?<=[.!?])\s+", summary)

    result = []

    for sentence in sentences:
        candidate = " ".join(result + [sentence])

        if len(candidate) > max_chars:
            break

        result.append(sentence)

    if result:
        return " ".join(result).strip()

    return summary[:max_chars].strip() + "..."


def get_news_for_cluster(session, cluster_id: int) -> list[News]:
    return (
        session.query(News)
        .filter(News.cluster_id == cluster_id)
        .order_by(News.is_primary.desc(), News.published_at.asc().nullslast(), News.collected_at.asc())
        .all()
    )


def get_cluster_text(
    session,
    cluster_id: int,
    max_news: int = 5,
    max_text_per_news: int = 2500
) -> str:
    news_items = get_news_for_cluster(session, cluster_id)

    parts = []

    for news in news_items[:max_news]:
        if news.title:
            parts.append(news.title)

        if news.text:
            parts.append(news.text[:max_text_per_news])

    return "\n".join(parts)


def generate_cluster_title(session, cluster_id: int) -> str:
    news_items = get_news_for_cluster(session, cluster_id)

    if not news_items:
        return ""

    primary_news = next(
        (news for news in news_items if news.is_primary),
        None
    )

    if primary_news and primary_news.title:
        return primary_news.title

    return news_items[0].title or ""


def summarize_cluster(
    session,
    cluster_id: int,
    sentence_count: int = 4,
    max_chars: int = 800
) -> str:
    cluster = session.query(Cluster).filter(Cluster.id == cluster_id).first()

    if not cluster:
        print(f"Кластер #{cluster_id} не найден.")
        return ""

    text = get_cluster_text(session, cluster_id)

    if not text:
        print(f"В кластере #{cluster_id} нет текста для суммаризации.")
        return ""

    cluster_title = generate_cluster_title(session, cluster_id)

    summary = build_textrank_summary(
        text=text,
        title=cluster_title,
        sentence_count=sentence_count,
        max_chars=max_chars
    )

    cluster.title = cluster_title
    cluster.summary_text = summary

    session.commit()

    return summary


def summarize_new_clusters(
    session,
    sentence_count: int = 4,
    max_chars: int = 800
):
    clusters = (
        session.query(Cluster)
        .filter(Cluster.summary_text.is_(None))
        .all()
    )

    print(f"Кластеров без summary: {len(clusters)}")

    for cluster in clusters:
        summary = summarize_cluster(
            session=session,
            cluster_id=cluster.id,
            sentence_count=sentence_count,
            max_chars=max_chars
        )

        print("-" * 100)
        print(f"Кластер #{cluster.id}")
        print(f"Заголовок: {cluster.title}")
        print(summary)

def remove_title_like_sentences(sentences: list[str], title: str | None) -> list[str]:
    if not title:
        return sentences

    title_key = normalize_sentence_key(title)

    result = []

    for sentence in sentences:
        sentence_key = normalize_sentence_key(sentence)

        if sentence_key == title_key:
            continue

        if sentence_key.startswith(title_key):
            sentence = sentence[len(title):].strip()

        if len(sentence) >= 45:
            result.append(sentence)

    return result


def resummarize_all_clusters(
    session,
    sentence_count: int = 4,
    max_chars: int = 800
):
    clusters = session.query(Cluster).all()

    print(f"Кластеров для пересуммаризации: {len(clusters)}")

    for cluster in clusters:
        summarize_cluster(
            session=session,
            cluster_id=cluster.id,
            sentence_count=sentence_count,
            max_chars=max_chars
        )

def fit_summary_by_sentences(
    selected_sentences: list[str],
    max_chars: int = 700
) -> str:
    while len(selected_sentences) > 1:
        summary = " ".join(selected_sentences)

        if len(summary) <= max_chars:
            return summary

        selected_sentences = selected_sentences[:-1]

    return selected_sentences[0] if selected_sentences else ""

def limit_summary(summary: str, max_chars: int = 800) -> str:
    if len(summary) <= max_chars:
        return summary

    sentences = re.split(r"(?<=[.!?])\s+", summary)

    result = []

    for sentence in sentences:
        candidate = " ".join(result + [sentence])

        if len(candidate) > max_chars:
            break

        result.append(sentence)

    if result:
        return " ".join(result).strip()

    return summary[:max_chars].strip() + "..."