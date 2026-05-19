import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import re
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database.models import Cluster, News
from utils.text_utils import normalize_article_text



BAD_SUMMARY_PHRASES = [
    "читайте также",
    "материалы по теме",
    "другие материалы",
    "главное в отраслевых сми",
    "подписывайтесь",
    "печатная версия",
    "источник новости",
    "фото:",
    "реклама",
    "на правах рекламы",
    "подробнее",
]

BAD_STARTS = [
    "также",
    "кроме того",
    "при этом",
    "однако",
    "в то же время",
    "между тем",

    "он",
    "она",
    "они",
    "оно",
    "это",
    "этот",
    "эта",
    "эти",
    "данный",
    "данная",
    "данные",
    "такие",
    "такой",

    "именно тогда",
    "тогда",
    "ранее",
    "позднее",
    "первоначально",
    "после этого",
    "до этого",
    "в результате этого",
]
BAD_CONTEXT_PHRASES = [
    "именно тогда",
    "как сообщалось ранее",
    "ранее сообщалось",
    "напомним",
    "стоит отметить",
    "следует отметить",
    "по словам",
    "сообщил",
    "сообщила",
    "сообщили",
    "рассказал",
    "рассказала",
    "рассказали",
]
FACT_WORDS = [
    "запустил", "запустила", "запустили",
    "открыл", "открыла", "открыли",
    "построил", "построила", "построили",
    "ввел", "ввела", "ввели", "введен", "введена",
    "начал", "начала", "начали",
    "завершил", "завершила", "завершили",
    "разработал", "разработала", "разработали",
    "представил", "представила", "представили",
    "модернизировал", "модернизировала", "модернизировали",
    "увеличил", "увеличила", "увеличили",
    "поставил", "поставила", "поставили",
    "произвел", "произвела", "произвели",
    "изготовил", "изготовила", "изготовили",
    "внедрил", "внедрила", "внедрили",
    "испытал", "испытала", "испытали",
    "передал", "передала", "передали",
]


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
        "т. д.": "т<dot> д<dot>",
        "т.п.": "т<dot>п<dot>",
        "т. п.": "т<dot> п<dot>",
        "т.е.": "т<dot>е<dot>",
        "т. е.": "т<dot> е<dot>",
        "руб.": "руб<dot>",
        "млн.": "млн<dot>",
        "млрд.": "млрд<dot>",
    }

    for old, new in abbreviations.items():
        text = text.replace(old, new)

    protected_chars = list(text)

    for i, char in enumerate(protected_chars):
        if char != ".":
            continue

        before = "".join(protected_chars[:i]).rstrip()
        word_match = re.search(r"([A-Za-zА-Яа-яЁё]+)$", before)

        if not word_match:
            continue

        last_word = word_match.group(1)

        if len(last_word) == 1:
            protected_chars[i] = "<dot>"

    text = "".join(protected_chars)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    result = []

    for sentence in sentences:
        sentence = sentence.replace("<dot>", ".").strip()

        if len(sentence) < 45:
            continue

        if len(sentence) > 420:
            continue

        lowered = sentence.lower()

        if any(phrase in lowered for phrase in BAD_SUMMARY_PHRASES):
            continue

        result.append(sentence)

    return result


def normalize_sentence_key(sentence: str) -> str:
    sentence = sentence.lower().replace("ё", "е")
    sentence = re.sub(r"[^а-яa-z0-9 ]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence

def is_bad_summary_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    key = normalize_sentence_key(sentence)

    if starts_badly(sentence):
        return True

    if any(phrase in lowered for phrase in BAD_CONTEXT_PHRASES):
        return True

    # Убираем предложения, которые начинаются с местоимений
    # и без предыдущего контекста выглядят непонятно.
    bad_pronoun_starts = [
        "он ", "она ", "они ", "оно ",
        "это ", "эти ", "этот ", "эта ",
        "данный ", "данная ", "данные ",
    ]

    if any(key.startswith(start.strip()) for start in bad_pronoun_starts):
        return True

    return False

INDUSTRIAL_FACT_WORDS = [
    "производство",
    "сборка",
    "выпуск",
    "завод",
    "предприятие",
    "площадка",
    "мощности",
    "линия",
    "цех",
    "оборудование",
    "техника",
    "экскаватор",
    "погрузчик",
    "станок",
    "двигатель",
    "компоненты",
]


def sentence_fact_bonus(sentence: str) -> float:
    lowered = sentence.lower()

    bonus = 0.0

    if any(word in lowered for word in FACT_WORDS):
        bonus += 0.08

    if any(word in lowered for word in INDUSTRIAL_FACT_WORDS):
        bonus += 0.08

    if re.search(r"\d", sentence):
        bonus += 0.04

    # Названия компаний, фондов, заводов
    if re.search(r"[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?", sentence):
        bonus += 0.03

    if is_bad_summary_sentence(sentence):
        bonus -= 0.20

    return bonus

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


def starts_badly(sentence: str) -> bool:
    key = normalize_sentence_key(sentence)

    return any(
        key.startswith(start)
        for start in BAD_STARTS
    )


def sentence_fact_bonus(sentence: str) -> float:
    lowered = sentence.lower()

    bonus = 0.0

    if any(word in lowered for word in FACT_WORDS):
        bonus += 0.08

    if re.search(r"\d", sentence):
        bonus += 0.04

    if re.search(r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?", sentence):
        bonus += 0.03

    if starts_badly(sentence):
        bonus -= 0.10

    if "сообщил" in lowered or "сообщили" in lowered:
        bonus -= 0.03

    if "по словам" in lowered:
        bonus -= 0.04

    return bonus


def get_lead_sentence(sentences: list[str]) -> tuple[int, str] | None:
    for index, sentence in enumerate(sentences[:4]):
        if starts_badly(sentence):
            continue

        return index, sentence

    return None


def are_sentences_too_similar(sentence_a: str, sentence_b: str) -> bool:
    key_a = normalize_sentence_key(sentence_a)
    key_b = normalize_sentence_key(sentence_b)

    if not key_a or not key_b:
        return False

    if key_a in key_b or key_b in key_a:
        return True

    words_a = set(key_a.split())
    words_b = set(key_b.split())

    if not words_a or not words_b:
        return False

    intersection = words_a & words_b
    union = words_a | words_b

    similarity = len(intersection) / len(union)

    return similarity >= 0.72


def build_textrank_summary(
    text: str,
    title: str | None = None,
    sentence_count: int = 3,
    max_chars: int = 600
) -> str:
    sentences = split_into_sentences(text)
    sentences = remove_title_like_sentences(sentences, title)
    sentences = remove_duplicate_sentences(sentences)

    if not sentences:
        return ""

    # Убираем предложения, которые плохо смотрятся в самостоятельном summary.
    good_sentences = [
        sentence for sentence in sentences
        if not is_bad_summary_sentence(sentence)
    ]

    # Если фильтр оказался слишком строгим, возвращаемся к исходным,
    # но всё равно не берём совсем плохие короткие куски.
    if good_sentences:
        sentences = good_sentences

    if len(sentences) == 1:
        return limit_summary(sentences[0], max_chars=max_chars)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_df=0.95,
        min_df=1,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    graph = nx.from_numpy_array(similarity_matrix)
    textrank_scores = nx.pagerank(graph)

    ranked_sentences = []

    for index, sentence in enumerate(sentences):
        score = textrank_scores[index]
        score += sentence_fact_bonus(sentence)

        # Первое нормальное предложение часто содержит главный факт.
        if index == 0:
            score += 0.10

        ranked_sentences.append((score, index, sentence))

    ranked_sentences.sort(reverse=True)

    selected = []

    lead = get_lead_sentence(sentences)

    if lead is not None:
        lead_index, lead_sentence = lead

        if not is_bad_summary_sentence(lead_sentence):
            selected.append((1.0, lead_index, lead_sentence))

    for score, index, sentence in ranked_sentences:
        if len(selected) >= sentence_count:
            break

        if is_bad_summary_sentence(sentence):
            continue

        too_similar = any(
            are_sentences_too_similar(sentence, selected_sentence)
            for _, _, selected_sentence in selected
        )

        if too_similar:
            continue

        selected.append((score, index, sentence))

    if not selected:
        selected = [ranked_sentences[0]]

    selected_sorted = sorted(selected, key=lambda item: item[1])
    selected_sentences = [item[2] for item in selected_sorted]

    summary = fit_summary_by_sentences(
        selected_sentences=selected_sentences,
        max_chars=max_chars
    )

    return summary


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


def get_news_for_cluster(session, cluster_id: int) -> list[News]:
    return (
        session.query(News)
        .filter(News.cluster_id == cluster_id)
        .order_by(
            News.is_primary.desc(),
            News.published_at.asc().nullslast(),
            News.collected_at.asc()
        )
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