"""Portfolio creator agent."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.agents.base import AgentResult, BaseAgent
from src.data_client import create_massive_client, fetch_stock_data
from src.llm_validation import (
    extract_valid_tickers,
    has_valid_tickers,
    parse_include_exclude_payload,
    parse_weights_payload,
)
from src.portfolio import allocate_portfolio_by_weights, normalize_weights
from src.summaries import build_portfolio_summary


class PortfolioCreatorAgent(BaseAgent):
    def __init__(
        self,
        llm_service: Any,
        config: Dict[str, Any],
        *,
        massive_client_factory: Callable[[str], Any] = create_massive_client,
        stock_data_fetcher: Callable[..., Dict[str, Any]] = fetch_stock_data,
    ) -> None:
        super().__init__(llm_service, config)
        self._massive_client_factory = massive_client_factory
        self._stock_data_fetcher = stock_data_fetcher

    @staticmethod
    def apply_ticker_overrides(
        recommended: List[str],
        include: List[str],
        exclude: List[str],
        max_tickers: int,
    ) -> List[str]:
        output: List[str] = []
        for ticker in recommended:
            ticker_up = str(ticker).upper()
            if ticker_up not in output:
                output.append(ticker_up)

        exclude_set = {ticker.upper() for ticker in exclude}
        for ticker in include:
            ticker_up = str(ticker).upper()
            if ticker_up not in output and ticker_up not in exclude_set:
                output.append(ticker_up)

        output = [ticker for ticker in output if ticker not in exclude_set]
        if max_tickers <= 0:
            return []
        return output[:max_tickers]

    def _extract_explicit_tickers(
        self,
        user_query: str,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> tuple[List[str], List[str]]:
        prompts_cfg = self.config["openrouter"]["prompts"]
        outputs_cfg = self.config["openrouter"]["outputs"]
        temperatures_cfg = self.config["openrouter"]["temperatures"]
        models_cfg = self.config["openrouter"]["models"]
        dashboard_cfg = self.config["dashboard"]

        prompt = prompts_cfg["extractor_template"].format(user_input=user_query)
        response, _ = self.llm_service.complete(
            request_name="ticker_extractor",
            model=models_cfg.get("extractor", models_cfg["ticker"]),
            max_tokens=outputs_cfg.get("extractor_max_tokens", 50),
            temperature=temperatures_cfg.get("extractor", 0.0),
            session_id=session_id,
            run_id=run_id,
            messages=[
                {"role": "system", "content": prompts_cfg["extractor_system"]},
                {"role": "user", "content": prompt},
            ],
        )
        text = self.llm_service.extract_message_text(response)
        include, exclude = parse_include_exclude_payload(text)
        delimiter = dashboard_cfg.get("ticker_delimiter", ",")
        include_clean = [t for t in extract_valid_tickers(delimiter.join(include), delimiter=delimiter)]
        exclude_clean = [t for t in extract_valid_tickers(delimiter.join(exclude), delimiter=delimiter)]
        return include_clean, exclude_clean

    def _get_recommended_tickers(
        self,
        user_query: str,
        max_tickers: int,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        followup: bool = False,
    ) -> tuple[List[str], str]:
        prompts_cfg = self.config["openrouter"]["prompts"]
        outputs_cfg = self.config["openrouter"]["outputs"]
        temperatures_cfg = self.config["openrouter"]["temperatures"]
        models_cfg = self.config["openrouter"]["models"]
        dashboard_cfg = self.config["dashboard"]

        if followup:
            system_prompt = prompts_cfg.get("creator_followup_system", prompts_cfg["ticker_system"]).format(
                num_tickers=max_tickers
            )
            prompt = prompts_cfg.get("creator_followup_template", prompts_cfg["ticker_template"]).format(
                user_input=user_query,
                num_tickers=max_tickers,
            )
            request_name = "ticker_generation_followup"
        else:
            system_prompt = prompts_cfg["ticker_system"].format(num_tickers=max_tickers)
            prompt = prompts_cfg["ticker_template"].format(user_input=user_query, num_tickers=max_tickers)
            request_name = "ticker_generation"

        response, _ = self.llm_service.complete(
            request_name=request_name,
            model=models_cfg["ticker"],
            max_tokens=outputs_cfg["ticker_max_tokens"],
            temperature=temperatures_cfg["ticker"],
            session_id=session_id,
            run_id=run_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw_output = self.llm_service.extract_message_text(response)
        tickers = extract_valid_tickers(raw_output, delimiter=dashboard_cfg["ticker_delimiter"])
        return tickers, raw_output

    def _fetch_data_and_filter_tickers(
        self,
        tickers: List[str],
    ) -> tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, List[str]]]:
        stocks_cfg = self.config["stocks"]
        massive_api_key = self.config.get("massive", {}).get("api", {}).get("api_key")
        if not massive_api_key:
            raise ValueError("Missing Massive.com API key.")

        massive_client = self._massive_client_factory(massive_api_key)
        data_by_ticker: Dict[str, Dict[str, Any]] = {}
        tickers_with_history: List[str] = []
        failed_history_by_status: Dict[str, List[str]] = {
            "rate_limited": [],
            "not_found": [],
            "empty_data": [],
            "unexpected_error": [],
        }

        for ticker in tickers:
            try:
                ticker_data = self._stock_data_fetcher(
                    ticker=ticker,
                    history_period=stocks_cfg["history_period"],
                    financials_period=stocks_cfg["financials_period"],
                    massive_client=massive_client,
                )
            except Exception:
                failed_history_by_status["unexpected_error"].append(ticker)
                continue

            history = ticker_data.get("history", pd.DataFrame())
            history_status = ticker_data.get("history_status", "ok")
            if history is None or history.empty or "Close" not in history.columns:
                if history_status not in failed_history_by_status:
                    history_status = "unexpected_error"
                if history_status == "ok":
                    history_status = "empty_data"
                failed_history_by_status[history_status].append(ticker)
                continue

            data_by_ticker[ticker] = ticker_data
            tickers_with_history.append(ticker)

        return data_by_ticker, tickers_with_history, failed_history_by_status

    def _generate_weights(
        self,
        user_input: str,
        tickers: List[str],
        summary_text: str,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> tuple[Dict[str, float], Dict[str, Any]]:
        prompts_cfg = self.config["openrouter"]["prompts"]
        outputs_cfg = self.config["openrouter"]["outputs"]
        temperatures_cfg = self.config["openrouter"]["temperatures"]
        models_cfg = self.config["openrouter"]["models"]

        weights_prompt = prompts_cfg["weights_template"].format(
            user_input=user_input,
            tickers=", ".join(tickers),
            summary=summary_text,
        )
        response, _ = self.llm_service.complete(
            request_name="weights_generation",
            model=models_cfg["weights"],
            max_tokens=outputs_cfg["weights_max_tokens"],
            temperature=temperatures_cfg["weights"],
            session_id=session_id,
            run_id=run_id,
            messages=[
                {"role": "system", "content": prompts_cfg["weights_system"]},
                {"role": "user", "content": weights_prompt},
            ],
        )
        text = self.llm_service.extract_message_text(response)
        parsed = parse_weights_payload(text)
        dropped = sorted(set(tickers) - set(parsed.keys())) if parsed else []
        weights = normalize_weights(parsed, tickers) if parsed else normalize_weights({}, tickers)
        return weights, {
            "weights_raw_text": text,
            "weights_parse_failed": not bool(parsed),
            "weights_dropped": dropped,
        }

    def _run(
        self,
        context: Dict[str, Any],
        *,
        followup: bool,
        include_overrides: Optional[List[str]] = None,
        exclude_overrides: Optional[List[str]] = None,
    ) -> AgentResult:
        user_input = context["user_input"]
        portfolio_size = float(context["portfolio_size"])
        stocks_cfg = self.config["stocks"]
        max_tickers = stocks_cfg["max_tickers"]
        session_id = context.get("session_id")
        run_id = context.get("run_id")

        include, exclude = self._extract_explicit_tickers(
            user_input,
            session_id=session_id,
            run_id=run_id,
        )
        if include_overrides:
            include = list(dict.fromkeys(include + [item.upper() for item in include_overrides]))
        if exclude_overrides:
            exclude = list(dict.fromkeys(exclude + [item.upper() for item in exclude_overrides]))

        recommended_tickers, ticker_raw_output = self._get_recommended_tickers(
            user_input=user_input,
            max_tickers=max_tickers,
            session_id=session_id,
            run_id=run_id,
            followup=followup,
        )
        merged_tickers = self.apply_ticker_overrides(
            recommended_tickers,
            include,
            exclude,
            max_tickers=max_tickers,
        )

        if not has_valid_tickers(merged_tickers):
            raise ValueError("No valid ticker symbols found from LLM output.")

        data_by_ticker, tickers_with_history, failed_history = self._fetch_data_and_filter_tickers(merged_tickers)
        if not tickers_with_history:
            raise ValueError("Could not fetch historical price data for any suggested ticker.")

        summary_text = build_portfolio_summary(
            tickers=tickers_with_history,
            data_by_ticker=data_by_ticker,
            financial_metrics=stocks_cfg["financials_metrics"],
        )
        weights, weights_meta = self._generate_weights(
            user_input=user_input,
            tickers=tickers_with_history,
            summary_text=summary_text,
            session_id=session_id,
            run_id=run_id,
        )
        allocation = allocate_portfolio_by_weights(
            tickers=tickers_with_history,
            total_amount=portfolio_size,
            weights=weights,
        )

        return AgentResult(
            tickers=tickers_with_history,
            include_tickers=include,
            exclude_tickers=exclude,
            data_by_ticker=data_by_ticker,
            summary_text=summary_text,
            weights=weights,
            allocation=allocation,
            metadata={
                "recommended_tickers": recommended_tickers,
                "ticker_raw_output": ticker_raw_output,
                "failed_history_by_status": failed_history,
                **weights_meta,
            },
        )

    def run_initial(self, context: Dict[str, Any]) -> AgentResult:
        return self._run(context, followup=False)

    def run_followup(self, context: Dict[str, Any], feedback: Dict[str, Any]) -> AgentResult:
        add = [str(t).upper() for t in feedback.get("add", [])]
        remove = [str(t).upper() for t in feedback.get("remove", [])]
        return self._run(
            context,
            followup=True,
            include_overrides=add,
            exclude_overrides=remove,
        )
