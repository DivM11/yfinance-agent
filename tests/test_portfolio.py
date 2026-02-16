"""Unit tests for portfolio allocation helpers."""

from src.portfolio import allocate_portfolio, format_portfolio_allocation


def test_allocate_portfolio_equal_weights():
    allocation = allocate_portfolio(["NVDA", "AMD", "GOOG"], total_amount=500.0)

    assert set(allocation.keys()) == {"NVDA", "AMD", "GOOG"}
    assert round(sum(allocation.values()), 2) == 500.0


def test_format_portfolio_allocation():
    allocation = {"NVDA": 200.0, "AMD": 100.0, "GOOG": 200.0}

    text = format_portfolio_allocation(allocation)

    assert "NVDA" in text
    assert "$200.00" in text
