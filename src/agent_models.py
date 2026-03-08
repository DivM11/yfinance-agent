"""Context and prompt models for creator/evaluator agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreatorContext:
    user_input: str
    portfolio_size: float
    session_id: str | None = None
    run_id: str | None = None
    excluded_tickers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluatorContext:
    user_input: str
    portfolio_size: float
    tickers: tuple[str, ...]
    summary_text: str


@dataclass(frozen=True)
class CreatorPrompts:
    ticker_system: str
    ticker_template: str
    ticker_followup_system: str
    ticker_followup_template: str
    weights_system: str
    weights_template: str


@dataclass(frozen=True)
class EvaluatorPrompts:
    analysis_system: str
    analysis_template: str
    analysis_followup_system: str
    analysis_followup_template: str
