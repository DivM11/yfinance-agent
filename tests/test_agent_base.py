"""Unit tests for base agent abstractions."""

from src.agents.base import AgentResult


def test_agent_result_defaults():
    result = AgentResult()

    assert result.tickers == []
    assert result.weights == {}
    assert result.suggestions == {}
