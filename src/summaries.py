"""Summarization helpers for ticker data."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from src.recommendations import summarize_recommendations_counts as _summarize_counts
from src.portfolio import normalize_weights


def summarize_history_stats(history: pd.DataFrame) -> Dict[str, float]:
    if history is None or history.empty or "Close" not in history.columns:
        return {}

    series = history["Close"].dropna()
    if series.empty:
        return {}

    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "current": float(series.iloc[-1]),
    }


def summarize_financials_latest(
    financials: pd.DataFrame,
    metrics: Iterable[str],
) -> Dict[str, float]:
    if financials is None or financials.empty:
        return {}

    try:
        latest_col = max(financials.columns)
    except TypeError:
        latest_col = financials.columns[-1]

    summary: Dict[str, float] = {}
    for metric in metrics:
        if metric in financials.index:
            value = financials.loc[metric, latest_col]
            if pd.notna(value):
                summary[metric] = float(value)
    return summary


def summarize_recommendations_counts(
    summary_df: pd.DataFrame,
    current_period: str,
    previous_period: str,
):
    return _summarize_counts(summary_df, current_period, previous_period)


def build_ticker_summary(
    ticker: str,
    data: Dict[str, pd.DataFrame],
    financial_metrics: Iterable[str],
    current_period: str,
    previous_period: str,
) -> str:
    parts = [ticker]

    history_stats = summarize_history_stats(data.get("history", pd.DataFrame()))
    if history_stats:
        parts.append(
            "price min {min} max {max} med {median} current {current}".format(**history_stats)
        )

    financials_summary = summarize_financials_latest(
        data.get("financials", pd.DataFrame()),
        financial_metrics,
    )
    if financials_summary:
        fin_parts = " ".join(f"{key} {value}" for key, value in financials_summary.items())
        parts.append(f"fin {fin_parts}")

    current, delta = summarize_recommendations_counts(
        data.get("recommendations_summary", pd.DataFrame()),
        current_period=current_period,
        previous_period=previous_period,
    )
    if current:
        cur_parts = " ".join(f"{key} {value}" for key, value in current.items())
        parts.append(f"recs {cur_parts}")
    if delta:
        delta_parts = " ".join(f"{key} {value}" for key, value in delta.items())
        parts.append(f"delta {delta_parts}")

    return " | ".join(parts)


def build_portfolio_summary(
    tickers: Iterable[str],
    data_by_ticker: Dict[str, Dict[str, pd.DataFrame]],
    financial_metrics: Iterable[str],
    current_period: str,
    previous_period: str,
) -> str:
    summaries = []
    for ticker in tickers:
        data = data_by_ticker.get(ticker, {})
        summaries.append(
            build_ticker_summary(
                ticker=ticker,
                data=data,
                financial_metrics=financial_metrics,
                current_period=current_period,
                previous_period=previous_period,
            )
        )
    return "\n".join(summaries)


def build_portfolio_returns_series(
    history_by_ticker: Dict[str, pd.DataFrame],
    weights: Dict[str, float] | None,
) -> pd.Series:
    closes = {}
    for ticker, data in history_by_ticker.items():
        if data is None or data.empty or "Close" not in data.columns:
            continue
        closes[ticker] = data["Close"].rename(ticker)

    if not closes:
        return pd.Series(dtype=float)

    prices = pd.concat(closes.values(), axis=1, join="inner").dropna(how="all")
    if prices.empty:
        return pd.Series(dtype=float)

    normalized = normalize_weights(weights, prices.columns)
    returns = prices.pct_change().fillna(0)
    weighted = sum(returns[col] * normalized.get(col, 0.0) for col in prices.columns)
    series = (1 + weighted).cumprod()
    series.name = "Portfolio"
    return series


def summarize_portfolio_stats(portfolio_series: pd.Series) -> Dict[str, float]:
    if portfolio_series is None or portfolio_series.empty:
        return {}

    return {
        "min": float(portfolio_series.min()),
        "max": float(portfolio_series.max()),
        "median": float(portfolio_series.median()),
        "current": float(portfolio_series.iloc[-1]),
        "return_1y": float(portfolio_series.iloc[-1] - 1.0),
    }


def summarize_portfolio_financials(
    financials_by_ticker: Dict[str, pd.DataFrame],
    weights: Dict[str, float] | None,
    metrics: Iterable[str],
) -> Dict[str, float]:
    if not financials_by_ticker:
        return {}

    normalized = normalize_weights(weights, financials_by_ticker.keys())
    totals: Dict[str, float] = {}
    for ticker, financials in financials_by_ticker.items():
        weight = normalized.get(ticker, 0.0)
        if weight <= 0:
            continue
        summary = summarize_financials_latest(financials, metrics)
        for metric, value in summary.items():
            totals[metric] = totals.get(metric, 0.0) + value * weight

    return totals
