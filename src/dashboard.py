"""Dashboard module for the YFinance Agent application.

Data is fetched from Massive.com (formerly Polygon.io) via the ``massive``
Python SDK.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
from openai import OpenAI

from src.data_client import (
    create_massive_client,
    fetch_stock_data as _massive_fetch_stock_data,
)
from src.llm_validation import (
    extract_valid_tickers,
    has_valid_tickers,
    parse_weights_payload,
    validate_weight_sum,
)
from src.plots import (
    plot_financials,
    plot_history,
    plot_portfolio_allocation,
    plot_portfolio_returns,
)
from src.portfolio import allocate_portfolio_by_weights, format_portfolio_allocation, normalize_weights
from src.summaries import (
    build_portfolio_returns_series,
    build_portfolio_summary,
    summarize_portfolio_financials,
    summarize_portfolio_stats,
)


logger = logging.getLogger(__name__)
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+(?::[a-zA-Z0-9_.-]+)?$")


def _log_backend(message: str, *args: object) -> None:
    logger.warning(message, *args)
    try:
        rendered = message % args if args else message
    except Exception:
        rendered = f"{message} | args={args}"
    print(rendered, flush=True)


def _is_model_name_valid(model_name: str) -> bool:
    return bool(MODEL_NAME_PATTERN.match(model_name))


def _create_openrouter_completion(
    *,
    client: OpenAI,
    request_name: str,
    model: str,
    max_tokens: int,
    temperature: float,
    messages: List[Dict[str, str]],
) -> tuple[Any, Optional[int]]:
    model_valid = _is_model_name_valid(model)
    _log_backend(
        "[%s] OpenRouter request start model=%s valid_model_format=%s max_tokens=%s temperature=%s messages=%s",
        request_name,
        model,
        model_valid,
        max_tokens,
        temperature,
        len(messages),
    )
    if not model_valid:
        _log_backend("[%s] Model name may be malformed: %s", request_name, model)

    try:
        raw_client = client.chat.completions.with_raw_response
        raw_response = raw_client.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        status_code = getattr(raw_response, "status_code", None)
        _log_backend(
            "[%s] OpenRouter response received status_code=%s",
            request_name,
            status_code,
        )
        return raw_response.parse(), status_code
    except AttributeError:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        _log_backend(
            "[%s] OpenRouter response received status_code=unavailable",
            request_name,
        )
        return response, None
    except Exception:
        logger.exception("[%s] OpenRouter request failed", request_name)
        raise


def create_openrouter_client(
    api_key: str,
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
) -> OpenAI:
    """Create an OpenRouter client using the OpenAI-compatible API."""
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or {})


def build_prompt(template: str, user_input: str, **kwargs: object) -> str:
    """Build the LLM prompt from a template."""
    return template.format(user_input=user_input, **kwargs)


def parse_tickers(text: str, delimiter: str) -> List[str]:
    """Parse a delimited ticker list into clean symbols."""
    return extract_valid_tickers(text, delimiter=delimiter)


def fetch_stock_data(
    ticker: str,
    history_period: str,
    financials_period: str,
    massive_client: Any = None,
) -> Dict[str, Any]:
    """Fetch stock data from Massive.com (formerly Polygon.io).

    Parameters
    ----------
    ticker:
        US equity symbol.
    history_period:
        Lookback period string (e.g. ``"1y"``).
    financials_period:
        ``"annual"`` or ``"quarterly"``.
    massive_client:
        An authenticated ``massive.RESTClient``.  If *None*, the caller must
        have already ensured a client is available (used only in tests).
    """
    if massive_client is None:
        raise ValueError("A Massive.com RESTClient must be provided.")
    return _massive_fetch_stock_data(
        client=massive_client,
        ticker=ticker,
        history_period=history_period,
        financials_period=financials_period,
    )


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
) -> tuple[List[str], str]:
    """Generate ticker suggestions from the LLM and return parsed + raw output."""
    response, _status_code = _create_openrouter_completion(
        client=client,
        request_name="ticker_generation",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    raw_output = _extract_message_text(response)
    return parse_tickers(raw_output, delimiter=delimiter), raw_output


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


def _push_chat_message(role: str, content: str, container) -> None:
    st.session_state["messages"].append({"role": role, "content": content})
    with container:
        with st.chat_message(role):
            st.markdown(content)


def run_dashboard(config: Dict[str, Any]) -> None:
    """Run the Streamlit dashboard."""
    st.title(config["app"]["title"])

    ui = config["ui"]
    dashboard = config["dashboard"]
    openrouter_cfg = config["openrouter"]
    stocks_cfg = config["stocks"]

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

    api_cfg = openrouter_cfg["api"]
    models_cfg = openrouter_cfg["models"]
    outputs_cfg = openrouter_cfg["outputs"]
    temperatures_cfg = openrouter_cfg["temperatures"]
    prompts_cfg = openrouter_cfg["prompts"]

    api_key = api_cfg.get("api_key")
    if not api_key:
        st.sidebar.error(ui["missing_api_key"])
        return

    tabs = st.tabs(
        [
            ui["chat_tab_label"],
            ui["history_tab_label"],
            ui["financials_tab_label"],
            ui["portfolio_tab_label"],
        ]
    )
    chat_tab, history_tab, financials_tab, portfolio_tab = tabs

    with chat_tab:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt_input = st.chat_input(ui["chat_placeholder"])

    if prompt_input:
        _push_chat_message("user", prompt_input, chat_tab)
        client = create_openrouter_client(
            api_key=api_key,
            base_url=api_cfg["base_url"],
            headers={
                "HTTP-Referer": api_cfg["http_referer"],
                "X-Title": api_cfg["app_title"],
            },
        )

        num_tickers = stocks_cfg["max_tickers"]
        ticker_system = prompts_cfg["ticker_system"].format(
            num_tickers=num_tickers,
        )
        prompt = build_prompt(
            prompts_cfg["ticker_template"], prompt_input,
            num_tickers=num_tickers,
        )
        tickers, ticker_raw_output = generate_tickers(
            client=client,
            prompt=prompt,
            system_prompt=ticker_system,
            model=models_cfg["ticker"],
            max_tokens=outputs_cfg["ticker_max_tokens"],
            temperature=temperatures_cfg["ticker"],
            delimiter=dashboard["ticker_delimiter"],
        )

        if not has_valid_tickers(tickers):
            _log_backend(
                "Ticker validation failed. Raw model output from OpenRouter: %s",
                ticker_raw_output,
            )
            _log_backend(
                "Ticker validation failed details: output_len=%s output_repr=%r",
                len(ticker_raw_output or ""),
                ticker_raw_output,
            )
            _push_chat_message("assistant", ui["ticker_validation_error"], chat_tab)
            return

        limited_tickers = limit_tickers(tickers, stocks_cfg["max_tickers"])
        if len(limited_tickers) < len(tickers):
            st.sidebar.warning(
                ui["max_tickers_warning"].format(max_tickers=stocks_cfg["max_tickers"])
            )
        tickers = limited_tickers
        _push_chat_message(
            "assistant",
            ui["ticker_reply_template"].format(tickers=", ".join(tickers)),
            chat_tab,
        )

        massive_api_key = config.get("massive", {}).get("api", {}).get("api_key")
        if not massive_api_key:
            st.sidebar.error(ui.get("missing_massive_key", "Missing Massive.com API key. Set MASSIVE_API_KEY in .env."))
            return
        massive_client = create_massive_client(api_key=massive_api_key)

        data_by_ticker: Dict[str, Dict[str, Any]] = {}
        for ticker in tickers:
            data_by_ticker[ticker] = fetch_stock_data(
                ticker,
                history_period=stocks_cfg["history_period"],
                financials_period=stocks_cfg["financials_period"],
                massive_client=massive_client,
            )

        summary_text = build_portfolio_summary(
            tickers=tickers,
            data_by_ticker=data_by_ticker,
            financial_metrics=stocks_cfg["financials_metrics"],
        )

        weights_prompt = prompts_cfg["weights_template"].format(
            user_input=prompt_input,
            tickers=", ".join(tickers),
            summary=summary_text,
        )
        weights_response, _weights_status_code = _create_openrouter_completion(
            client=client,
            request_name="weights_generation",
            model=models_cfg["weights"],
            max_tokens=outputs_cfg["weights_max_tokens"],
            temperature=temperatures_cfg["weights"],
            messages=[
                {"role": "system", "content": prompts_cfg["weights_system"]},
                {"role": "user", "content": weights_prompt},
            ],
        )
        weights_text = _extract_message_text(weights_response)
        parsed_weights = parse_weights_payload(weights_text)
        is_valid_weight_sum, raw_weight_sum = validate_weight_sum(parsed_weights)
        if not parsed_weights or not is_valid_weight_sum:
            st.sidebar.warning(ui["weights_validation_warning"].format(total=raw_weight_sum))
            _push_chat_message(
                "assistant",
                ui["weights_fallback_message"].format(total=raw_weight_sum),
                chat_tab,
            )
            weights = normalize_weights({}, tickers)
        else:
            weights = normalize_weights(parsed_weights, tickers)

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

        analysis_prompt = prompts_cfg["analysis_template"].format(
            user_input=prompt_input,
            tickers=", ".join(tickers),
            portfolio_size=st.session_state["portfolio_size"],
            weights=json.dumps(weights),
            summary=summary_text,
            allocation=allocation_text,
        )
        analysis_response, _analysis_status_code = _create_openrouter_completion(
            client=client,
            request_name="analysis_generation",
            model=models_cfg["analysis"],
            max_tokens=outputs_cfg["analysis_max_tokens"],
            temperature=temperatures_cfg["analysis"],
            messages=[
                {"role": "system", "content": prompts_cfg["analysis_system"]},
                {"role": "user", "content": analysis_prompt},
            ],
        )
        analysis_text = _extract_message_text(analysis_response)
        _push_chat_message("assistant", analysis_text, chat_tab)
        _push_chat_message("assistant", ui["post_analysis_nudge"], chat_tab)

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
    st.sidebar.write(ui["suggested_label"], tickers)
    if tickers:
        selected_history_tickers = st.sidebar.multiselect(
            ui["history_ticker_label"],
            options=tickers,
            default=st.session_state.get("selected_history_tickers", tickers),
            key="selected_history_tickers",
        )
    else:
        selected_history_tickers = []

    data_by_ticker = st.session_state.get("data_by_ticker", {})

    with history_tab:
        if not tickers:
            st.info(ui["history_empty_message"])
        else:
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

    with financials_tab:
        if not tickers:
            st.info(ui["financials_empty_message"])
        else:
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

    with portfolio_tab:
        if not tickers:
            st.info(ui["portfolio_empty_message"])
        else:
            allocation = st.session_state.get("portfolio_allocation", {})
            if allocation:
                st.subheader(ui["portfolio_output_label"])
                alloc_fig = plot_portfolio_allocation(
                    allocation, title=ui["portfolio_output_label"],
                )
                if alloc_fig is not None:
                    st.plotly_chart(alloc_fig, use_container_width=True)

            stats = st.session_state.get("portfolio_stats", {})
            if stats:
                st.subheader(ui["portfolio_stats_label"])
                stats_df = pd.DataFrame(
                    [{
                        "Min": f"{stats.get('min', 0):.2f}",
                        "Max": f"{stats.get('max', 0):.2f}",
                        "Median": f"{stats.get('median', 0):.2f}",
                        "Current": f"{stats.get('current', 0):.2f}",
                        "1Y Return": f"{stats.get('return_1y', 0):.2%}",
                    }]
                )
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            portfolio_series = st.session_state.get("portfolio_series", pd.Series(dtype=float))
            returns_fig = plot_portfolio_returns(portfolio_series, ui["portfolio_returns_label"])
            if returns_fig is not None:
                st.plotly_chart(returns_fig, use_container_width=True)

            portfolio_financials = st.session_state.get("portfolio_financials", {})
            if portfolio_financials:
                st.subheader(ui["portfolio_financials_label"])
                st.dataframe(
                    pd.DataFrame([portfolio_financials]),
                    use_container_width=True,
                    hide_index=True,
                )