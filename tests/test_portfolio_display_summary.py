"""Unit tests for portfolio display summary formatting."""

from src.portfolio_display_summary import PortfolioDisplaySummary


def test_format_suggestions_human_readable():
    summary = PortfolioDisplaySummary()
    text = summary.format_suggestions(
        {
            "add": ["FSLR", "CRWD"],
            "remove": ["TSLA"],
            "reweight": {"MSFT": 0.15, "NVDA": 0.3},
        }
    )

    assert "Suggested portfolio changes:" in text
    assert "Add: FSLR, CRWD" in text
    assert "Remove: TSLA" in text
    assert "MSFT: 15.00%" in text


def test_format_portfolio_header_empty():
    summary = PortfolioDisplaySummary()
    assert summary.format_portfolio_header([]) == "Recommended Portfolio Tickers: (none)"
