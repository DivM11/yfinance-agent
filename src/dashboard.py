"""Dashboard module for the YFinance Agent application."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import streamlit as st
import yfinance as yf
import pandas as pd
from openai import OpenAI

from src.plots import plot_financials, plot_history, plot_portfolio_returns, plot_recommendations
from src.portfolio import allocate_portfolio_by_weights, format_portfolio_allocation, normalize_weights
from src.summaries import (
    build_portfolio_returns_series,
    build_portfolio_summary,
    summarize_portfolio_financials,
    summarize_portfolio_stats,
)


def create_openrouter_client(
    api_key: str,
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
) -> OpenAI:
    """Create an OpenRouter client using the OpenAI-compatible API."""
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or {})


def build_prompt(template: str, user_input: str) -> str:
    """Build the LLM prompt from a template."""
    return template.format(user_input=user_input)


def parse_tickers(text: str, delimiter: str) -> List[str]:
    """Parse a delimited ticker list into clean symbols."""
    return [item.strip().upper() for item in text.split(delimiter) if item.strip()]


def fetch_stock_data(
    ticker: str,
    history_period: str,
    financials_period: str,
) -> Dict[str, Any]:
    """Fetch stock data from YFinance."""
    stock = yf.Ticker(ticker)
    if financials_period == "quarterly":
        financials = stock.quarterly_financials
    else:
        financials = stock.financials

    try:
        recommendations_summary = stock.recommendations_summary
    except AttributeError:
        recommendations_summary = pd.DataFrame()

    return {
        "history": stock.history(period=history_period),
        "financials": financials,
        "recommendations": stock.recommendations,
        "recommendations_summary": recommendations_summary,
    }


def _extract_message_text(response: Any) -> str:
    """Extract text content from an OpenAI-compatible response."""
    try:
        return response.choices[0].message.content
    except AttributeError:
        return response["choices"][0]["message"]["content"]


def generate_tickers(
    client: OpenAI,
    prompt: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    delimiter: str,
) -> List[str]:
    """Generate ticker suggestions from the LLM."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return parse_tickers(_extract_message_text(response), delimiter=delimiter)


def limit_tickers(tickers: List[str], max_tickers: int) -> List[str]:
    """Limit the number of tickers returned."""
    if max_tickers <= 0:
        return []
    return tickers[:max_tickers]


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def _init_state(default_user_input: str, default_portfolio_size: float, chat_intro: str) -> None:
    state = st.session_state
    state.setdefault("messages", [{"role": "assistant", "content": chat_intro}])
    state.setdefault("user_input", default_user_input)
    state.setdefault("tickers", [])
    state.setdefault("data_by_ticker", {})
    state.setdefault("selected_history_tickers", [])
    state.setdefault("portfolio_size", default_portfolio_size)
    state.setdefault("weights", {})
    state.setdefault("portfolio_allocation", {})
    state.setdefault("portfolio_stats", {})
    state.setdefault("portfolio_financials", {})
    state.setdefault("portfolio_series", pd.Series(dtype=float))
    state.setdefault("analysis_text", "")


def _parse_weights(text: str, tickers: List[str]) -> Dict[str, float]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return normalize_weights({}, tickers)

    weights: Dict[str, float] = {}
    if isinstance(payload, dict):
        if "weights" in payload and isinstance(payload["weights"], dict):
            weights = payload["weights"]
        else:
            weights = payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "ticker" in item and "weight" in item:
                weights[str(item["ticker"]).upper()] = float(item["weight"])

    return normalize_weights(weights, tickers)


def run_dashboard(config: Dict[str, Any]) -> None:
    """Run the Streamlit dashboard."""
    st.title(config["app"]["title"])

    ui = config["ui"]
    dashboard = config["dashboard"]
    openrouter_cfg = config["openrouter"]
    stocks_cfg = config["stocks"]
    recommendations_cfg = config["recommendations"]

    _init_state(
        dashboard["default_user_input"],
        dashboard["default_portfolio_size"],
        ui["chat_intro"],
    )

    st.sidebar.header(ui["sidebar_header"])
    st.sidebar.number_input(
        ui["portfolio_size_label"],
        min_value=0.0,
        step=100.0,
        key="portfolio_size",
    )

    api_key = openrouter_cfg.get("api_key")
    if not api_key:
        st.sidebar.error(ui["missing_api_key"])
        return

    chat_col, plots_col = st.columns([1, 1.4], gap="large")

    with chat_col:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt_input = st.chat_input(ui["chat_placeholder"])

    if prompt_input:
        st.session_state["messages"].append({"role": "user", "content": prompt_input})
        prompt = build_prompt(openrouter_cfg["prompt_template"], prompt_input)
        client = create_openrouter_client(
            api_key=api_key,
            base_url=openrouter_cfg["base_url"],
            headers={
                "HTTP-Referer": openrouter_cfg["http_referer"],
                "X-Title": openrouter_cfg["app_title"],
            },
        )
        tickers = generate_tickers(
            client=client,
            prompt=prompt,
            system_prompt=openrouter_cfg["system_prompt"],
            model=openrouter_cfg["model"],
            max_tokens=openrouter_cfg["max_tokens"],
            temperature=openrouter_cfg["temperature"],
            delimiter=dashboard["ticker_delimiter"],
        )

        limited_tickers = limit_tickers(tickers, stocks_cfg["max_tickers"])
        if len(limited_tickers) < len(tickers):
            st.sidebar.warning(
                ui["max_tickers_warning"].format(max_tickers=stocks_cfg["max_tickers"])
            )
        tickers = limited_tickers

        data_by_ticker: Dict[str, Dict[str, Any]] = {}
        for ticker in tickers:
            data_by_ticker[ticker] = fetch_stock_data(
                ticker,
                history_period=stocks_cfg["history_period"],
                financials_period=stocks_cfg["financials_period"],
            )

        summary_text = build_portfolio_summary(
            tickers=tickers,
            data_by_ticker=data_by_ticker,
            financial_metrics=stocks_cfg["financials_metrics"],
            current_period=recommendations_cfg["current_period"],
            previous_period=recommendations_cfg["previous_period"],
        )

        weights_prompt = openrouter_cfg["weights_prompt_template"].format(
            user_input=prompt_input,
            tickers=", ".join(tickers),
            summary=summary_text,
        )
        weights_response = client.chat.completions.create(
            model=openrouter_cfg["weights_model"],
            max_tokens=openrouter_cfg["weights_max_tokens"],
            temperature=openrouter_cfg["weights_temperature"],
            messages=[
                {"role": "system", "content": openrouter_cfg["weights_system_prompt"]},
                {"role": "user", "content": weights_prompt},
            ],
        )
        weights_text = _extract_message_text(weights_response)
        weights = _parse_weights(weights_text, tickers)

        allocation = allocate_portfolio_by_weights(
            tickers=tickers,
            total_amount=st.session_state["portfolio_size"],
            weights=weights,
        )
        allocation_text = format_portfolio_allocation(allocation)

        portfolio_series = build_portfolio_returns_series(
            {ticker: data["history"] for ticker, data in data_by_ticker.items()},
            weights,
        )
        portfolio_stats = summarize_portfolio_stats(portfolio_series)
        portfolio_financials = summarize_portfolio_financials(
            {ticker: data["financials"] for ticker, data in data_by_ticker.items()},
            weights,
            stocks_cfg["financials_metrics"],
        )

        analysis_prompt = openrouter_cfg["analysis_prompt_template"].format(
            user_input=prompt_input,
            tickers=", ".join(tickers),
            portfolio_size=st.session_state["portfolio_size"],
            weights=json.dumps(weights),
            summary=summary_text,
            allocation=allocation_text,
        )
        analysis_response = client.chat.completions.create(
            model=openrouter_cfg["analysis_model"],
            max_tokens=openrouter_cfg["analysis_max_tokens"],
            temperature=openrouter_cfg["analysis_temperature"],
            messages=[
                {"role": "system", "content": openrouter_cfg["analysis_system_prompt"]},
                {"role": "user", "content": analysis_prompt},
            ],
        )
        analysis_text = _extract_message_text(analysis_response)

        st.session_state["messages"].append({"role": "assistant", "content": analysis_text})

        st.session_state["tickers"] = tickers
        st.session_state["data_by_ticker"] = data_by_ticker
        st.session_state["selected_history_tickers"] = tickers
        st.session_state["weights"] = weights
        st.session_state["portfolio_allocation"] = allocation
        st.session_state["portfolio_stats"] = portfolio_stats
        st.session_state["portfolio_financials"] = portfolio_financials
        st.session_state["portfolio_series"] = portfolio_series
        st.session_state["analysis_text"] = analysis_text

    tickers = st.session_state.get("tickers", [])
    if not tickers:
        return

    st.sidebar.write(ui["suggested_label"], tickers)
    selected_history_tickers = st.sidebar.multiselect(
        ui["history_ticker_label"],
        options=tickers,
        default=st.session_state.get("selected_history_tickers", tickers),
        key="selected_history_tickers",
    )

    data_by_ticker = st.session_state.get("data_by_ticker", {})

    with plots_col:
        tabs = st.tabs(
            [
                ui["history_tab_label"],
                ui["financials_tab_label"],
                ui["recommendations_tab_label"],
                ui["portfolio_tab_label"],
            ]
        )

        with tabs[0]:
            history_fig = plot_history(
                {ticker: data["history"] for ticker, data in data_by_ticker.items()},
                selected_tickers=selected_history_tickers,
            )
            if history_fig is not None:
                st.plotly_chart(history_fig, use_container_width=True)

            st.caption(ui["download_prompt"])
            for ticker in tickers:
                history = data_by_ticker.get(ticker, {}).get("history", pd.DataFrame())
                st.download_button(
                    label=f"{ui['download_history_label']} ({ticker})",
                    data=_df_to_csv_bytes(history),
                    file_name=f"{ticker}_history.csv",
                    mime="text/csv",
                )

        with tabs[1]:
            for ticker in tickers:
                st.subheader(ui["ticker_header_template"].format(ticker=ticker))
                data = data_by_ticker.get(ticker, {})
                financials_fig = plot_financials(
                    data.get("financials", pd.DataFrame()),
                    metrics=stocks_cfg["financials_metrics"],
                    title=f"{ticker} Financials",
                )
                if financials_fig is not None:
                    st.plotly_chart(financials_fig, use_container_width=True)

                st.caption(ui["download_prompt"])
                financials = data.get("financials", pd.DataFrame())
                st.download_button(
                    label=f"{ui['download_financials_label']} ({ticker})",
                    data=_df_to_csv_bytes(financials),
                    file_name=f"{ticker}_financials.csv",
                    mime="text/csv",
                )

        with tabs[2]:
            for ticker in tickers:
                st.subheader(ui["ticker_header_template"].format(ticker=ticker))
                data = data_by_ticker.get(ticker, {})
                recommendations_fig = plot_recommendations(
                    data.get("recommendations_summary", pd.DataFrame()),
                    current_period=recommendations_cfg["current_period"],
                    previous_period=recommendations_cfg["previous_period"],
                    title=f"{ticker} Recommendations",
                )
                if recommendations_fig is not None:
                    st.plotly_chart(recommendations_fig, use_container_width=True)
                else:
                    st.info(ui["recommendations_missing"].format(ticker=ticker))

                st.caption(ui["download_prompt"])
                recommendations = data.get("recommendations", pd.DataFrame())
                st.download_button(
                    label=f"{ui['download_recommendations_label']} ({ticker})",
                    data=_df_to_csv_bytes(recommendations),
                    file_name=f"{ticker}_recommendations.csv",
                    mime="text/csv",
                )

        with tabs[3]:
            allocation = st.session_state.get("portfolio_allocation", {})
            if allocation:
                st.subheader(ui["portfolio_output_label"])
                st.write(format_portfolio_allocation(allocation))

            stats = st.session_state.get("portfolio_stats", {})
            if stats:
                st.subheader(ui["portfolio_stats_label"])
                st.write(
                    ui["portfolio_stats_template"].format(
                        min=stats.get("min"),
                        max=stats.get("max"),
                        median=stats.get("median"),
                        current=stats.get("current"),
                        return_1y=stats.get("return_1y"),
                    )
                )

            portfolio_series = st.session_state.get("portfolio_series", pd.Series(dtype=float))
            returns_fig = plot_portfolio_returns(portfolio_series, ui["portfolio_returns_label"])
            if returns_fig is not None:
                st.plotly_chart(returns_fig, use_container_width=True)

            portfolio_financials = st.session_state.get("portfolio_financials", {})
            if portfolio_financials:
                st.subheader(ui["portfolio_financials_label"])
                st.dataframe(pd.DataFrame([portfolio_financials]))