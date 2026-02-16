"""Unit tests for data summarization helpers."""

import pandas as pd

from src.summaries import (
    summarize_financials_latest,
    summarize_history_stats,
    summarize_recommendations_counts,
    build_ticker_summary,
)


def test_summarize_history_stats():
    history = pd.DataFrame({"Close": [10.0, 12.0, 11.0, 9.0]})

    stats = summarize_history_stats(history)

    assert stats["min"] == 9.0
    assert stats["max"] == 12.0
    assert stats["median"] == 10.5
    assert stats["current"] == 9.0


def test_summarize_financials_latest():
    financials = pd.DataFrame(
        {"2024-01-01": [100, 50], "2024-04-01": [110, 60]},
        index=["Total Revenue", "EBITDA"],
    )

    summary = summarize_financials_latest(financials, ["Total Revenue", "EBITDA"])

    assert summary["Total Revenue"] == 110
    assert summary["EBITDA"] == 60


def test_summarize_recommendations_counts():
    summary_df = pd.DataFrame(
        {"period": ["0m", "-1m"], "buy": [5, 3], "hold": [2, 4]},
    )

    current, delta = summarize_recommendations_counts(summary_df, current_period="0m", previous_period="1m")

    assert current["Buy"] == 5
    assert delta["Buy"] == 2


def test_build_ticker_summary():
    data = {
        "history": pd.DataFrame({"Close": [10.0, 12.0, 11.0, 9.0]}),
        "financials": pd.DataFrame(
            {"2024-01-01": [100], "2024-04-01": [110]},
            index=["Total Revenue"],
        ),
        "recommendations_summary": pd.DataFrame(
            {"period": ["0m"], "buy": [5], "hold": [2]},
        ),
    }

    summary_text = build_ticker_summary(
        ticker="AAPL",
        data=data,
        financial_metrics=["Total Revenue"],
        current_period="0m",
        previous_period="1m",
    )

    assert "AAPL" in summary_text
    assert "price" in summary_text.lower()
