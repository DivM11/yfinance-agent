"""Portfolio allocation helpers."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Dict, Iterable


def allocate_portfolio(
    tickers: Iterable[str],
    total_amount: float,
    precision: int = 2,
) -> Dict[str, float]:
    """Allocate a total amount across tickers using equal weights."""
    tickers_list = list(tickers)
    if not tickers_list or total_amount <= 0:
        return {}

    count = len(tickers_list)
    quant = Decimal("1") / (Decimal(10) ** precision)
    total = Decimal(str(total_amount))
    base = (total / Decimal(count)).quantize(quant, rounding=ROUND_DOWN)

    allocation = {ticker: base for ticker in tickers_list}
    allocated = base * Decimal(count)
    remainder = (total - allocated).quantize(quant)

    step = quant
    idx = 0
    while remainder > Decimal("0"):
        ticker = tickers_list[idx % count]
        allocation[ticker] = (allocation[ticker] + step).quantize(quant)
        remainder = (remainder - step).quantize(quant)
        idx += 1

    return {ticker: float(value) for ticker, value in allocation.items()}


def format_portfolio_allocation(allocation: Dict[str, float]) -> str:
    """Format allocation output for display."""
    if not allocation:
        return "Recommended Portfolio: (none)"

    parts = [f"{ticker} ${amount:,.2f}" for ticker, amount in allocation.items()]
    return "Recommended Portfolio: " + ", ".join(parts)
