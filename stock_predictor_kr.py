#!/usr/bin/env python3
"""
Korean Stock Market Predictor (multi-source baseline)

What this script does
- Downloads Korean market target data (default: KOSPI ^KS11)
- Adds U.S. market influence features (S&P500, NASDAQ, Dow, SOX, USD/KRW)
- Collects text from market media/news/forum RSS and computes daily sentiment
- Trains a classification model to predict next-day UP/DOWN direction

Notes
- This is still a research baseline, not financial advice.
- RSS and price sources may change over time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class Config:
    target_ticker: str = "^KS11"  # KOSPI
    period: str = "10y"
    test_size: float = 0.2


US_INFLUENCE_TICKERS = {
    "^GSPC": "sp500",     # S&P500
    "^IXIC": "nasdaq",    # Nasdaq
    "^DJI": "dow",        # Dow Jones
    "^SOX": "sox",        # Philadelphia Semiconductor
    "KRW=X": "usdkrw",    # USD/KRW
}


def load_price(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}")
    return df


def add_target_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["kr_ret_1"] = out["Close"].pct_change(1)
    out["kr_ret_5"] = out["Close"].pct_change(5)
    out["kr_ma_5"] = out["Close"].rolling(5).mean()
    out["kr_ma_20"] = out["Close"].rolling(20).mean()
    out["kr_vol_20"] = out["kr_ret_1"].rolling(20).std()
    out["kr_ma_ratio"] = out["kr_ma_5"] / out["kr_ma_20"]
    return out


def add_us_influence_features(target_df: pd.DataFrame, period: str) -> pd.DataFrame:
    out = target_df.copy()
    for ticker, alias in US_INFLUENCE_TICKERS.items():
        us = load_price(ticker, period)
        us_feat = pd.DataFrame(index=us.index)
        us_feat[f"{alias}_ret_1"] = us["Close"].pct_change(1)
        us_feat[f"{alias}_ret_3"] = us["Close"].pct_change(3)
        us_feat[f"{alias}_vol_10"] = us_feat[f"{alias}_ret_1"].rolling(10).std()

        # KR market reflects prior US close: shift by 1 trading day
        us_feat = us_feat.shift(1)
        out = out.join(us_feat, how="left")
    return out


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def build_rss_urls(keywords: Iterable[str]) -> list[str]:
    urls = []
    for kw in keywords:
        q = quote(kw)
        urls.append(f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko")
    return urls


def collect_daily_sentiment(period: str, max_items: int = 400) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    keywords = [
        "미국 증시", "S&P 500", "NASDAQ", "Dow Jones", "반도체 주가",
        "KOSPI", "코스피 전망", "한국 주식", "주식 포럼", "기업 실적"
    ]
    rss_urls = build_rss_urls(keywords)

    records: list[dict] = []
    seen = set()

    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[: max_items // max(1, len(rss_urls))]:
            title = normalize_text(getattr(entry, "title", ""))
            summary = normalize_text(getattr(entry, "summary", ""))
            link = getattr(entry, "link", "")
            key = (title, link)
            if not title or key in seen:
                continue
            seen.add(key)

            published = getattr(entry, "published_parsed", None)
            if not published:
                continue
            d = dt.date(published.tm_year, published.tm_mon, published.tm_mday)

            text = f"{title}. {summary}"
            score = analyzer.polarity_scores(text)["compound"]
            records.append({"date": d, "sentiment": score})

    if not records:
        return pd.DataFrame(columns=["news_sent_mean", "news_sent_std", "news_count"])

    df = pd.DataFrame(records)
    grouped = df.groupby("date").agg(
        news_sent_mean=("sentiment", "mean"),
        news_sent_std=("sentiment", "std"),
        news_count=("sentiment", "count"),
    )
    grouped["news_sent_std"] = grouped["news_sent_std"].fillna(0.0)

    # align to market reaction: prior-day news affects next KR session
    grouped.index = pd.to_datetime(grouped.index)
    grouped = grouped.shift(1)
    return grouped


def build_dataset(target_ticker: str, period: str) -> pd.DataFrame:
    kr = load_price(target_ticker, period)
    data = add_target_features(kr)
    data = add_us_influence_features(data, period=period)

    sentiment = collect_daily_sentiment(period=period)
    if not sentiment.empty:
        data = data.join(sentiment, how="left")

    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().copy()
    return data


def train_and_evaluate(data: pd.DataFrame, test_size: float):
    exclude = {"Open", "High", "Low", "Close", "Adj Close", "target"}
    feature_cols = [c for c in data.columns if c not in exclude]

    X = data[feature_cols]
    y = data["target"]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)

    latest_x = scaler.transform(X.iloc[[-1]])
    latest_prob_up = model.predict_proba(latest_x)[0][1]
    latest_pred = "UP" if latest_prob_up >= 0.5 else "DOWN"

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return {
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, digits=3),
        "latest_pred": latest_pred,
        "latest_prob_up": latest_prob_up,
        "latest_date": data.index[-1],
        "top_features": importances.head(12),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Korean stock market predictor with US+news signals")
    parser.add_argument("--ticker", default="^KS11", help="Target ticker (default: ^KS11, KOSPI)")
    parser.add_argument("--period", default="10y", help="Price history period (default: 10y)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    args = parser.parse_args()

    data = build_dataset(args.ticker, args.period)
    result = train_and_evaluate(data, test_size=args.test_size)

    print("=" * 72)
    print(f"Target: {args.ticker}")
    print(f"Samples: {len(data)} | Train: {result['n_train']} | Test: {result['n_test']}")
    print(f"Test Accuracy: {result['accuracy']:.4f}")
    print("-" * 72)
    print("Classification Report")
    print(result["report"])
    print("-" * 72)
    print("Top Feature Importances")
    print(result["top_features"].to_string())
    print("-" * 72)
    print(
        f"Latest date: {result['latest_date'].date()} | Predicted next day: {result['latest_pred']} "
        f"(P[UP]={result['latest_prob_up']:.3f})"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
