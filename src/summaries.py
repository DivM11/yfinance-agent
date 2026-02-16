"""Summarization helpers for ticker data."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from src.recommendations import summarize_recommendations_counts as _summarize_counts


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
