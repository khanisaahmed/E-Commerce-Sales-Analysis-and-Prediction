import pandas as pd
import numpy as np
import re
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# =========================
# NLTK SETUP (SAFE FOR CLOUD)
# =========================
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sid = SentimentIntensityAnalyzer()

# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# FEATURE EXTRACTION PIPELINE
# =========================
def extract_features(df):
    """
    Generates EXACT feature set used during XGBoost training
    """

    # -------------------------
    # REQUIRED COLUMNS
    # -------------------------
    if "review_text" not in df.columns or "rating" not in df.columns:
        raise ValueError("CSV must contain 'review_text' and 'rating' columns")

    df = df.copy()

    # -------------------------
    # TEXT CLEANING
    # -------------------------
    df["clean_text"] = df["review_text"].apply(clean_text)

    # -------------------------
    # SENTIMENT SCORES
    # -------------------------
    df["sent_scores"] = df["clean_text"].apply(
        lambda x: sid.polarity_scores(x)
    )

    df["sent_pos"] = df["sent_scores"].apply(lambda x: x["pos"])
    df["sent_neg"] = df["sent_scores"].apply(lambda x: x["neg"])
    df["sent_neu"] = df["sent_scores"].apply(lambda x: x["neu"])
    df["sent_compound"] = df["sent_scores"].apply(lambda x: x["compound"])

    # -------------------------
    # REVIEW LENGTH FEATURES
    # -------------------------
    df["review_length"] = df["clean_text"].apply(lambda x: len(x))
    df["review_length_words"] = df["clean_text"].apply(lambda x: len(x.split()))

    # -------------------------
    # PRODUCT-LEVEL AGGREGATION
    # -------------------------
    features = pd.DataFrame([{
        "Score": df["rating"].mean(),
        "sent_compound": df["sent_compound"].mean(),
        "sent_pos": df["sent_pos"].mean(),
        "sent_neg": df["sent_neg"].mean(),
        "sent_neu": df["sent_neu"].mean(),
        "review_length": df["review_length"].mean(),
        "review_length_words": df["review_length_words"].mean(),
        "PositiveRatio": (df["rating"] >= 4).mean(),
        "MaxReviewAgeDays": 0   # Neutral default (user CSV has no time column)
    }])

    return features

def extract_summary_stats(df):
    """
    Returns human-readable statistics for UI display.
    This function is self-contained and does NOT assume
    precomputed sentiment columns.
    """

    df = df.copy()

    total_reviews = len(df)
    avg_rating = round(df["rating"].mean(), 2)

    # Clean text (reuse existing function)
    df["clean_text"] = df["review_text"].apply(clean_text)

    # Compute sentiment again (safe & lightweight)
    df["sent_compound"] = df["clean_text"].apply(
        lambda x: sid.polarity_scores(x)["compound"]
    )

    # Sentiment categories
    df["sentiment_category"] = df["sent_compound"].apply(
        lambda x: "Positive" if x >= 0.05 else "Negative" if x <= -0.05 else "Neutral"
    )

    sentiment_counts = df["sentiment_category"].value_counts(normalize=True) * 100

    return {
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
        "positive_pct": round(sentiment_counts.get("Positive", 0), 1),
        "neutral_pct": round(sentiment_counts.get("Neutral", 0), 1),
        "negative_pct": round(sentiment_counts.get("Negative", 0), 1),
    }


from collections import Counter
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

def extract_top_tokens(df, top_n=10):
    """
    Extracts most frequent meaningful words from reviews.
    Self-contained (creates clean_text internally).
    """

    df = df.copy()

    # Ensure clean_text exists
    df["clean_text"] = df["review_text"].apply(clean_text)

    all_tokens = []

    for text in df["clean_text"]:
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
        all_tokens.extend(tokens)

    counter = Counter(all_tokens)
    return counter.most_common(top_n)


from nltk.util import bigrams

def extract_top_bigrams(df, top_n=10):
    """
    Extracts most frequent meaningful bigrams from reviews.
    Self-contained (creates clean_text internally).
    """

    df = df.copy()

    # Ensure clean_text exists
    df["clean_text"] = df["review_text"].apply(clean_text)

    all_bigrams = []

    for text in df["clean_text"]:
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

        bigram_tokens = list(bigrams(tokens))
        all_bigrams.extend(bigram_tokens)

    bigram_counter = Counter(all_bigrams)

    formatted = [
        (" ".join(pair), count)
        for pair, count in bigram_counter.most_common(top_n)
    ]

    return formatted

def extract_review_highlights(df, top_n=1):
    """
    Returns top-N most positive and top-N most negative reviews
    based on sentiment score.
    """

    df = df.copy()

    # Clean text
    df["clean_text"] = df["review_text"].apply(clean_text)

    # Sentiment score
    df["sent_compound"] = df["clean_text"].apply(
        lambda x: sid.polarity_scores(x)["compound"]
    )

    # Sort reviews
    df_sorted = df.sort_values("sent_compound", ascending=False)

    top_positive = df_sorted.head(top_n)["review_text"].tolist()
    top_negative = df_sorted.tail(top_n)["review_text"].tolist()

    return top_positive, top_negative
