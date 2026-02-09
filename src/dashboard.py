"""Dashboard module for the YFinance Agent application."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
import yfinance as yf
import pandas as pd
from openai import OpenAI

from src.plots import plot_financials, plot_history, plot_recommendations


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


def run_dashboard(config: Dict[str, Any]) -> None:
    """Run the Streamlit dashboard."""
    st.title(config["app"]["title"])

    ui = config["ui"]
    dashboard = config["dashboard"]
    openrouter_cfg = config["openrouter"]
    stocks_cfg = config["stocks"]
    recommendations_cfg = config["recommendations"]

    st.sidebar.header(ui["sidebar_header"])
    user_input = st.sidebar.text_area(ui["input_label"], dashboard["default_user_input"])

    api_key = openrouter_cfg.get("api_key")
    if not api_key:
        st.sidebar.error(ui["missing_api_key"])
        return

    if st.sidebar.button(ui["button_label"]):
        prompt = build_prompt(openrouter_cfg["prompt_template"], user_input)
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

        st.sidebar.write(ui["suggested_label"], tickers)
        selected_history_tickers = st.sidebar.multiselect(
            ui["history_ticker_label"],
            options=tickers,
            default=tickers,
        )

        data_by_ticker: Dict[str, Dict[str, Any]] = {}
        for ticker in tickers:
            data_by_ticker[ticker] = fetch_stock_data(
                ticker,
                history_period=stocks_cfg["history_period"],
                financials_period=stocks_cfg["financials_period"],
            )

        history_fig = plot_history(
            {ticker: data["history"] for ticker, data in data_by_ticker.items()},
            selected_tickers=selected_history_tickers,
        )
        if history_fig is not None:
            st.plotly_chart(history_fig, use_container_width=True)

        for ticker in tickers:
            st.subheader(ui["ticker_header_template"].format(ticker=ticker))
            data = data_by_ticker[ticker]
            st.write(ui["section_history"])
            st.dataframe(data["history"])

            financials_fig = plot_financials(
                data["financials"],
                metrics=stocks_cfg["financials_metrics"],
                title=f"{ticker} Financials",
            )
            if financials_fig is not None:
                st.plotly_chart(financials_fig, use_container_width=True)

            st.write(ui["section_financials"])
            st.dataframe(data["financials"])

            recommendations_fig = plot_recommendations(
                data["recommendations_summary"],
                current_period=recommendations_cfg["current_period"],
                previous_period=recommendations_cfg["previous_period"],
                title=f"{ticker} Recommendations",
            )
            if recommendations_fig is not None:
                st.plotly_chart(recommendations_fig, use_container_width=True)
            else:
                st.info(ui["recommendations_missing"].format(ticker=ticker))

            st.write(ui["section_recommendations"])
            st.dataframe(data["recommendations"])