import os
import re
import math
import time
import html
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Optional NLP libs; we will import lazily
from functools import lru_cache

# ---------------------------
# Page & Theme Config
# ---------------------------
st.set_page_config(
    page_title="Sentiment Pulse Dashboard",
    page_icon="📊",
    layout="wide",
)

# Inject minimal CSS for sentiment badges and clean look
BADGE_CSS = """
<style>
.badge {padding: 3px 8px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;}
.badge.pos {background: #e6f4ea; color: #137333; border: 1px solid #c6e9cc}
.badge.neg {background: #fce8e6; color: #a50e0e; border: 1px solid #f1b0b7}
.badge.neu {background: #eef3fc; color: #1a73e8; border: 1px solid #c6dafc}
.small-muted {color: #6b7280; font-size: 0.85rem}
.header {font-weight: 700; font-size: 1.1rem}
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)

# ---------------------------
# Utilities
# ---------------------------
@lru_cache(maxsize=1)
def _vader():
    """Lazy-load VADER and its lexicon."""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"Failed to load VADER: {e}")
        return None

@lru_cache(maxsize=1)
def _hf_pipeline():
    """Lazy-load a 3-class Hugging Face sentiment model.
    We use cardiffnlp/twitter-roberta-base-sentiment-latest (labels: negative, neutral, positive).
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load Transformers model: {e}")
        return None


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def strip_html(text: str) -> str:
    # Basic HTML entities & tag removal
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", " ", text).strip()


# ---------------------------
# Data Collection
# ---------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SentimentPulse/1.0)"}


def fetch_google_news_rss(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent items from Google News RSS for the query."""
    import feedparser
    q = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:limit]:
        rows.append({
            "source": getattr(entry, 'source', {}).get('title', 'Google News'),
            "title": strip_html(entry.get('title', '')),
            "text": strip_html(entry.get('title', '')),
            "link": entry.get('link'),
            "published": entry.get('published', ''),
        })
    return rows


def fetch_newsapi(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch from NewsAPI if NEWSAPI_KEY is set. https://newsapi.org/"""
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": min(limit, 100),
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": key,
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for a in data.get("articles", [])[:limit]:
        rows.append({
            "source": (a.get("source") or {}).get("name", "NewsAPI"),
            "title": strip_html(a.get("title", "")),
            "text": strip_html(a.get("description", "") or a.get("title", "")),
            "link": a.get("url"),
            "published": a.get("publishedAt"),
        })
    return rows


def collect_texts(query: str, provider: str, limit: int) -> pd.DataFrame:
    rows = []
    try:
        if provider == "Google News RSS":
            rows = fetch_google_news_rss(query, limit)
        elif provider == "NewsAPI.org":
            rows = fetch_newsapi(query, limit)
        else:
            rows = fetch_google_news_rss(query, limit)
    except Exception as e:
        st.error(f"Data collection failed: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        # Parse dates for sorting
        def try_parse(dt):
            try:
                return pd.to_datetime(dt)
            except Exception:
                return pd.NaT
        df["published_ts"] = df["published"].apply(try_parse)
        df = df.sort_values("published_ts", ascending=False).reset_index(drop=True)
    return df


# ---------------------------
# Sentiment Analysis
# ---------------------------

LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}


def vader_sentiment(texts: List[str]) -> List[Dict[str, Any]]:
    sia = _vader()
    out = []
    if sia is None:
        return out
    for t in texts:
        s = sia.polarity_scores(t or "")
        # Determine label and confidence as max of probabilities
        label = max([("Positive", s["pos"]), ("Negative", s["neg"]), ("Neutral", s["neu"])], key=lambda x: x[1])[0]
        conf = max(s["pos"], s["neg"], s["neu"])  # 0..1
        out.append({"label": label, "confidence": conf, "scores": s})
    return out


def hf_sentiment(texts: List[str]) -> List[Dict[str, Any]]:
    tpl = _hf_pipeline()
    if tpl is None:
        return []
    tokenizer, model = tpl
    results = []
    import torch
    with torch.no_grad():
        for t in texts:
            inputs = tokenizer(t or "", return_tensors="pt", truncation=True, max_length=256)
            logits = model(**inputs).logits[0].numpy()
            probs = softmax(logits)
            idx = int(np.argmax(probs))
            # cardiffnlp labels: 0 negative, 1 neutral, 2 positive
            label = ["Negative", "Neutral", "Positive"][idx]
            conf = float(probs[idx])
            results.append({"label": label, "confidence": conf, "scores": {"neg": float(probs[0]), "neu": float(probs[1]), "pos": float(probs[2])}})
    return results


# ---------------------------
# UI - Sidebar Controls
# ---------------------------

st.title("📊 Sentiment Pulse — Real‑time Headline Sentiment Dashboard")
st.caption("Type a topic, brand, or movie. I’ll fetch recent headlines and gauge the mood.")

with st.sidebar:
    st.header("Settings")
    query = st.text_input("Search Topic", value="OpenAI", help="Enter a brand, movie, person, or any topic.")
    provider = st.selectbox("Data Source", ["Google News RSS", "NewsAPI.org"], help="NewsAPI requires a NEWSAPI_KEY env var.")
    limit = st.slider("Max Headlines", min_value=20, max_value=200, value=80, step=10)
    model_choice = st.radio("Sentiment Model", ["VADER (fast)", "RoBERTa (more accurate)"], index=0, help="RoBERTa = cardiffnlp/twitter-roberta-base-sentiment-latest")
    analyze_btn = st.button("Run Analysis", type="primary")

# ---------------------------
# Action: Fetch + Analyze
# ---------------------------

if analyze_btn and query.strip():
    with st.spinner("Collecting recent mentions..."):
        df = collect_texts(query.strip(), provider, limit)

    if df.empty:
        st.warning("No results found. Try a broader query or different source.")
        st.stop()

    texts = df["text"].fillna("").astype(str).tolist()

    with st.spinner("Scoring sentiment..."):
        if model_choice.startswith("VADER"):
            preds = vader_sentiment(texts)
        else:
            preds = hf_sentiment(texts)

    if not preds:
        st.error("Sentiment analysis failed to return results.")
        st.stop()

    # Attach predictions
    df["sentiment"] = [p["label"] for p in preds]
    df["confidence"] = [round(p["confidence"], 3) for p in preds]

    # ---------------------------
    # Metrics
    # ---------------------------
    total = len(df)
    pos = int((df["sentiment"] == "Positive").sum())
    neg = int((df["sentiment"] == "Negative").sum())
    neu = int((df["sentiment"] == "Neutral").sum())
    pos_pct = (pos / total) * 100 if total else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Mentions Analyzed", f"{total}")
    col2.metric("Overall Positive", f"{pos_pct:.1f}%")
    col3.metric("Data Source", provider)

    # ---------------------------
    # Pie / Donut Chart
    # ---------------------------
    pie_df = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Count": [pos, neu, neg]
    })
    fig = px.pie(pie_df, names="Sentiment", values="Count", hole=0.45)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Recent Mentions Table
    # ---------------------------
    st.subheader("Recent Mentions")

    def badge(label: str) -> str:
        cls = 'neu'
        if label == 'Positive':
            cls = 'pos'
        elif label == 'Negative':
            cls = 'neg'
        return f"<span class='badge {cls}'>{label}</span>"

    def row_md(row) -> str:
        ts = row.get("published_ts")
        ts_str = "" if pd.isna(ts) else ts.strftime('%Y-%m-%d %H:%M')
        link = row.get("link") or "#"
        title = (row.get("title") or row.get("text") or "").strip()
        return (
            f"{badge(row['sentiment'])} "
            f"<a href='{link}' target='_blank'><b>{html.escape(title)}</b></a><br>"
            f"<span class='small-muted'>{row.get('source', 'Unknown')} • {ts_str}</span>"
        )

    # Show top 30 for readability; allow download of full CSV
    display_df = df.head(30).copy()
    display_df["_render"] = display_df.apply(row_md, axis=1)
    st.markdown("\n".join(display_df["_render"].tolist()), unsafe_allow_html=True)

    with st.expander("View raw data / download"):
        st.dataframe(df.drop(columns=["_render"], errors='ignore'))
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), file_name=f"sentiment_{query.replace(' ', '_')}.csv", mime="text/csv")

    # ---------------------------
    # Insight bullets
    # ---------------------------
    st.subheader("Quick Insights")
    bullets = []
    # Trend-like heuristics
    if pos > neg and pos_pct >= 60:
        bullets.append("Overall sentiment skews strongly positive.")
    elif neg > pos and (neg / total) >= 0.5:
        bullets.append("Negative coverage dominates — consider drilling into top negative articles.")
    else:
        bullets.append("Mixed sentiment — both positive and negative angles present.")
    top_pos = df[df["sentiment"] == "Positive"].head(3)["title"].tolist()
    top_neg = df[df["sentiment"] == "Negative"].head(3)["title"].tolist()
    if top_pos:
        bullets.append("Representative positives: " + "; ".join(top_pos))
    if top_neg:
        bullets.append("Representative negatives: " + "; ".join(top_neg))

    for b in bullets:
        st.markdown(f"- {b}")

else:
    # Helpful empty state
    st.info("Enter a topic on the left and click **Run Analysis**. Try: *Barbie movie*, *iPhone 16*, *cricket world cup*.")

# ---------------------------
# Footer
# ---------------------------
st.write("")
st.caption("Data sources: Google News RSS and/or NewsAPI.org. Sentiment models: VADER or CardiffNLP RoBERTa (3-class).")
