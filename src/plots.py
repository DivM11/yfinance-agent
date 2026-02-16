"""Plotting utilities for the dashboard."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from src.recommendations import CATEGORIES, summarize_recommendations_counts


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
        yaxis_title="Close Price",
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

    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Value",
        legend_title="Metric",
    )
    _apply_gridlines(fig)
    return fig


def plot_recommendations(
    summary: pd.DataFrame,
    current_period: str,
    previous_period: str,
    title: str,
) -> Optional[Figure]:
    """Plot current recommendation counts and delta vs previous period."""
    if summary is None or summary.empty:
        return None

    current_vals, delta_vals = summarize_recommendations_counts(
        summary,
        current_period=current_period,
        previous_period=previous_period,
    )
    if not current_vals:
        return None

    rows = 2 if delta_vals is not None else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=False)

    fig.add_trace(
        go.Bar(x=CATEGORIES, y=list(current_vals.values()), name="Current"),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Analyst recommendations", row=1, col=1)

    if delta_vals is not None:
        fig.add_trace(
            go.Bar(x=CATEGORIES, y=list(delta_vals.values()), name="Delta"),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Change in past month", row=2, col=1)

    fig.update_layout(title=title)
    _apply_gridlines(fig)
    return fig
