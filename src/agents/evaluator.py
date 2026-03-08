"""Portfolio evaluator agent."""

from __future__ import annotations

import json
from typing import Any, Dict

from src.agents.base import AgentResult, BaseAgent
from src.llm_validation import parse_evaluator_suggestions


class PortfolioEvaluatorAgent(BaseAgent):
    def _evaluate(
        self,
        *,
        user_input: str,
        portfolio_size: float,
        tickers: list[str],
        weights: Dict[str, float],
        allocation: Dict[str, float],
        summary_text: str,
        session_id: str | None = None,
        run_id: str | None = None,
        followup: bool = False,
        previous_analysis: str = "",
        applied_changes: Dict[str, Any] | None = None,
    ) -> AgentResult:
        openrouter_cfg = self.config["openrouter"]
        prompts_cfg = openrouter_cfg["prompts"]
        outputs_cfg = openrouter_cfg["outputs"]
        temperatures_cfg = openrouter_cfg["temperatures"]
        models_cfg = openrouter_cfg["models"]

        if followup:
            system_prompt = prompts_cfg.get("evaluator_followup_system", prompts_cfg["evaluator_system"])
            prompt = prompts_cfg.get("evaluator_followup_template", prompts_cfg["evaluator_template"]).format(
                user_input=user_input,
                portfolio_size=portfolio_size,
                tickers=", ".join(tickers),
                weights=json.dumps(weights),
                allocation=json.dumps(allocation),
                summary=summary_text,
                previous_analysis=previous_analysis,
                applied_changes=json.dumps(applied_changes or {}),
            )
            request_name = "evaluation_followup"
        else:
            system_prompt = prompts_cfg["evaluator_system"]
            prompt = prompts_cfg["evaluator_template"].format(
                user_input=user_input,
                portfolio_size=portfolio_size,
                tickers=", ".join(tickers),
                weights=json.dumps(weights),
                allocation=json.dumps(allocation),
                summary=summary_text,
            )
            request_name = "evaluation"

        response, _ = self.llm_service.complete(
            request_name=request_name,
            model=models_cfg.get("evaluator", models_cfg["analysis"]),
            max_tokens=outputs_cfg.get("evaluator_max_tokens", outputs_cfg["analysis_max_tokens"]),
            temperature=temperatures_cfg.get("evaluator", temperatures_cfg["analysis"]),
            session_id=session_id,
            run_id=run_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        analysis_text = self.llm_service.extract_message_text(response)
        suggestions = parse_evaluator_suggestions(analysis_text)
        return AgentResult(
            analysis_text=analysis_text,
            suggestions=suggestions,
            metadata={"request_name": request_name},
        )

    def run_initial(self, context: Dict[str, Any]) -> AgentResult:
        return self._evaluate(
            user_input=context["user_input"],
            portfolio_size=float(context["portfolio_size"]),
            tickers=context["tickers"],
            weights=context["weights"],
            allocation=context["allocation"],
            summary_text=context["summary_text"],
            session_id=context.get("session_id"),
            run_id=context.get("run_id"),
            followup=False,
        )

    def run_followup(self, context: Dict[str, Any], feedback: Dict[str, Any]) -> AgentResult:
        return self._evaluate(
            user_input=context["user_input"],
            portfolio_size=float(context["portfolio_size"]),
            tickers=context["tickers"],
            weights=context["weights"],
            allocation=context["allocation"],
            summary_text=context["summary_text"],
            session_id=context.get("session_id"),
            run_id=context.get("run_id"),
            followup=True,
            previous_analysis=context.get("previous_analysis", ""),
            applied_changes=feedback,
        )
