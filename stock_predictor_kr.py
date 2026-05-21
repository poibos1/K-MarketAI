#!/usr/bin/env python3
"""
Korean Stock Market Predictor (baseline)
- Downloads KOSPI (^KS11) daily data with yfinance
- Builds simple features (returns, moving averages, volatility)
- Trains a logistic regression classifier
- Predicts next-day direction (UP/DOWN)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    ticker: str = "^KS11"  # KOSPI index
    period: str = "10y"
    test_size: float = 0.2
    random_state: int = 42


def load_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    out["ma_ratio"] = out["ma_5"] / out["ma_20"]
    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out = out.dropna().copy()
    return out


def train_and_evaluate(data: pd.DataFrame, test_size: float, random_state: int):
    feature_cols = ["ret_1", "ret_5", "ma_ratio", "vol_20", "Volume"]
    X = data[feature_cols]
    y = data["target"]

    # Time-series split: preserve order (no shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    latest_x = X.iloc[[-1]]
    latest_prob_up = model.predict_proba(latest_x)[0][1]
    latest_pred = "UP" if latest_prob_up >= 0.5 else "DOWN"

    return {
        "model": model,
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, digits=3),
        "latest_pred": latest_pred,
        "latest_prob_up": latest_prob_up,
        "latest_date": data.index[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Korean stock market direction predictor")
    parser.add_argument("--ticker", default="^KS11", help="Ticker (default: ^KS11 for KOSPI)")
    parser.add_argument("--period", default="10y", help="Data period for yfinance (default: 10y)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    df = load_data(args.ticker, args.period)
    data = build_features(df)
    result = train_and_evaluate(data, test_size=args.test_size, random_state=42)

    print("=" * 60)
    print(f"Ticker: {args.ticker}")
    print(f"Samples: {len(data)}")
    print(f"Test Accuracy: {result['accuracy']:.4f}")
    print("-" * 60)
    print("Classification Report")
    print(result["report"])
    print("-" * 60)
    print(
        f"Latest date: {result['latest_date'].date()} | Predicted next day: {result['latest_pred']} "
        f"(P[UP]={result['latest_prob_up']:.3f})"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
