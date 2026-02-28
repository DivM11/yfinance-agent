"""Unit tests for plot helpers."""

import pandas as pd

from src.plots import plot_financials, plot_history


def _history_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0, 101.5, 99.8]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )


def test_plot_history_selected_tickers():
    history = {"AAPL": _history_df(), "MSFT": _history_df()}
    fig = plot_history(history, selected_tickers=["AAPL"])

    assert fig is not None
    assert len(fig.data) == 1


def test_plot_financials_metrics():
    financials = pd.DataFrame(
        {
            pd.Timestamp("2023-12-31"): [100, 10],
            pd.Timestamp("2024-03-31"): [120, 12],
        },
        index=["Total Revenue", "EBITDA"],
    )

    fig = plot_financials(financials, metrics=["Total Revenue", "EBITDA"], title="Test")

    assert fig is not None
    assert len(fig.data) == 2
