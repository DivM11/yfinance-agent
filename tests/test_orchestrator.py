"""Unit tests for agent orchestrator."""

from src.agents.base import AgentResult
from src.agents.orchestrator import (
    AgentOrchestrator,
    STATUS_AWAITING_APPROVAL,
    STATUS_COMPLETE,
    STATUS_MAX_ITERATIONS_REACHED,
)


class DummyCreator:
    def __init__(self):
        self.followup_calls = 0

    def run_initial(self, context):
        return AgentResult(
            tickers=["AAPL"],
            weights={"AAPL": 1.0},
            allocation={"AAPL": context["portfolio_size"]},
            summary_text="summary",
            metadata={"recommended_tickers": ["AAPL"], "excluded_tickers": []},
        )

    def run_followup(self, context, feedback):
        self.followup_calls += 1
        return AgentResult(
            tickers=["AAPL", "NVDA"],
            weights={"AAPL": 0.7, "NVDA": 0.3},
            allocation={"AAPL": context["portfolio_size"] * 0.7, "NVDA": context["portfolio_size"] * 0.3},
            summary_text="summary2",
            metadata={
                "feedback": feedback,
                "recommended_tickers": ["AAPL", "TSLA"],
                "excluded_tickers": ["TSLA"],
            },
        )


class DummyEvaluator:
    def __init__(self):
        self.calls = 0

    def run_initial(self, _context):
        self.calls += 1
        return AgentResult(analysis_text="analysis", suggestions={"add": ["NVDA"], "remove": [], "reweight": {}})

    def run_followup(self, _context, _feedback):
        self.calls += 1
        return AgentResult(analysis_text="updated", suggestions={"add": [], "remove": [], "reweight": {}})


def test_orchestrator_approval_flow():
    orchestrator = AgentOrchestrator(DummyCreator(), DummyEvaluator(), max_iterations=3)

    state = orchestrator.start(user_input="growth", portfolio_size=1000.0)
    assert state.status == STATUS_AWAITING_APPROVAL
    assert state.selected_tickers == ["AAPL"]
    assert state.recommended_tickers == ["AAPL"]

    updated = orchestrator.apply_changes(state)
    assert updated.status == STATUS_COMPLETE
    assert updated.iteration == 2
    assert updated.selected_tickers == ["AAPL", "NVDA"]
    assert updated.recommended_tickers == ["AAPL", "TSLA"]
    assert updated.excluded_tickers == ["TSLA"]


def test_orchestrator_max_iterations():
    class StickyEvaluator(DummyEvaluator):
        def run_followup(self, _context, _feedback):
            return AgentResult(analysis_text="still", suggestions={"add": ["MSFT"], "remove": [], "reweight": {}})

    orchestrator = AgentOrchestrator(DummyCreator(), StickyEvaluator(), max_iterations=2)
    state = orchestrator.start(user_input="growth", portfolio_size=1000.0)
    updated = orchestrator.apply_changes(state)

    assert updated.status == STATUS_MAX_ITERATIONS_REACHED


def test_orchestrator_reject_changes():
    orchestrator = AgentOrchestrator(DummyCreator(), DummyEvaluator(), max_iterations=3)
    state = orchestrator.start(user_input="growth", portfolio_size=1000.0)

    final_state = orchestrator.reject_changes(state)
    assert final_state.status == STATUS_COMPLETE
    assert final_state.pending_suggestions == {}
