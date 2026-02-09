"""Plotting utilities for the dashboard."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots


def _apply_gridlines(fig: Figure) -> None:
    fig.update_xaxes(showgrid=True, gridcolor="#E6E6E6")
    fig.update_yaxes(showgrid=True, gridcolor="#E6E6E6")


def plot_history(
    history_by_ticker: Dict[str, pd.DataFrame],
    selected_tickers: Optional[Iterable[str]] = None,
) -> Optional[Figure]:
    """Plot closing price history for each ticker on a single chart."""
    if not history_by_ticker:
        return None

    tickers = list(selected_tickers) if selected_tickers is not None else list(history_by_ticker)
    fig = go.Figure()
    for ticker in tickers:
        history = history_by_ticker.get(ticker)
        if history is None or history.empty:
            continue
        if "Close" not in history.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["Close"],
                mode="lines",
                name=ticker,
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
        title="Historical Price (Close) vs Date",
        xaxis_title="Date",
        yaxis_title="Close",
        legend_title="Ticker",
    )
    _apply_gridlines(fig)
    return fig


def _select_financials_metrics(
    financials: pd.DataFrame,
    metrics: Iterable[str],
) -> Tuple[pd.DataFrame, List[str]]:
    if financials is None or financials.empty:
        return pd.DataFrame(), []

    available = [metric for metric in metrics if metric in financials.index]
    if not available:
        return pd.DataFrame(), []

    selected = financials.loc[available]
    return selected.sort_index(axis=1), available


def plot_financials(
    financials: pd.DataFrame,
    metrics: Iterable[str],
    title: str,
) -> Optional[Figure]:
    """Plot selected financial metrics over time."""
    selected, available = _select_financials_metrics(financials, metrics)
    if selected.empty:
        return None

    fig = go.Figure()
    for metric in available:
        fig.add_trace(
            go.Scatter(
                x=selected.columns,
                y=selected.loc[metric],
                mode="lines+markers",
                name=metric,
            )
        )

    fig.update_layout(title=title, xaxis_title="Period", yaxis_title="Value")
    _apply_gridlines(fig)
    return fig


def _normalize_recommendation_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = {}
    for col in df.columns:
        normalized[col] = col.strip().lower()

    rename_map = {
        "strongbuy": "StrongBuy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strongsell": "StrongSell",
    }

    df = df.rename(columns={col: rename_map.get(norm, col) for col, norm in normalized.items()})
    return df


def _extract_period_row(df: pd.DataFrame, period: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if period in df.index:
        return df.loc[period]
    return None


def plot_recommendations(
    summary: pd.DataFrame,
    current_period: str,
    previous_period: str,
    title: str,
) -> Optional[Figure]:
    """Plot current recommendation counts and delta vs previous period."""
    if summary is None or summary.empty:
        return None

    summary = _normalize_recommendation_columns(summary)
    summary = summary.set_index(summary.index.astype(str))

    current = _extract_period_row(summary, current_period)
    previous = _extract_period_row(summary, previous_period)
    if current is None:
        return None

    categories = ["StrongBuy", "Buy", "Hold", "Sell", "StrongSell"]
    current_vals = current.reindex(categories).fillna(0)

    delta_vals = None
    if previous is not None:
        delta_vals = current_vals - previous.reindex(categories).fillna(0)

    rows = 2 if delta_vals is not None else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=False)

    fig.add_trace(
        go.Bar(x=categories, y=current_vals.values, name="Current"),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Count", row=1, col=1)

    if delta_vals is not None:
        fig.add_trace(
            go.Bar(x=categories, y=delta_vals.values, name="Delta"),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Delta", row=2, col=1)

    fig.update_layout(title=f"{title} (Current {current_period})")
    _apply_gridlines(fig)
    return fig
