"""Unit tests for the dashboard module."""

import pandas as pd

from src.dashboard import (
    _extract_message_text,
    build_prompt,
    create_openrouter_client,
    fetch_stock_data,
    generate_tickers,
    limit_tickers,
    parse_tickers,
    run_dashboard,
)


class DummyTicker:
    """Minimal stub for yfinance.Ticker."""

    def history(self, period: str) -> pd.DataFrame:
        return pd.DataFrame({"price": [1, 2, 3]})

    @property
    def financials(self) -> pd.DataFrame:
        return pd.DataFrame({"metric": ["revenue"]})

    @property
    def recommendations(self) -> pd.DataFrame:
        return pd.DataFrame({"grade": ["buy"]})

    @property
    def recommendations_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "strongBuy": [1, 2],
                "buy": [3, 4],
                "hold": [5, 6],
                "sell": [1, 1],
                "strongSell": [0, 1],
            },
            index=["0m", "1m"],
        )


class DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = DummyMessage(content)


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [DummyChoice(content)]


class DummyChatCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **_kwargs):
        return DummyResponse(self._content)


class DummyChat:
    def __init__(self, content: str) -> None:
        self.completions = DummyChatCompletions(content)


class DummyClient:
    """Minimal stub for OpenAI-compatible client."""

    def __init__(self, text: str) -> None:
        self.chat = DummyChat(text)


class DummySidebar:
    def __init__(self, button_value: bool, user_input: str = "input") -> None:
        self._button_value = button_value
        self._user_input = user_input
        self.session_state = None
        self.header_text = None
        self.text_area_args = None
        self.multiselect_args = None
        self.error_msg = None
        self.write_args = None
        self.warning_msg = None
        self.button_called = False

    def header(self, text: str) -> None:
        self.header_text = text

    def text_area(self, label: str, value: str | None = None, key: str | None = None) -> str:
        self.text_area_args = (label, value, key)
        if key and self.session_state is not None:
            if key not in self.session_state:
                self.session_state[key] = value if value is not None else self._user_input
            return self.session_state[key]
        return self._user_input

    def button(self, label: str) -> bool:
        self.button_called = True
        return self._button_value

    def multiselect(self, label: str, options, default=None, key: str | None = None):
        self.multiselect_args = (label, options, default, key)
        selection = default or []
        if key and self.session_state is not None:
            if key not in self.session_state:
                self.session_state[key] = selection
            return self.session_state[key]
        return selection

    def write(self, label: str, value) -> None:
        self.write_args = (label, value)

    def error(self, msg: str) -> None:
        self.error_msg = msg

    def warning(self, msg: str) -> None:
        self.warning_msg = msg


class DummyStreamlit:
    def __init__(self, sidebar: DummySidebar) -> None:
        self.sidebar = sidebar
        self.session_state = {}
        self.sidebar.session_state = self.session_state
        self.title_text = None
        self.subheaders = []
        self.writes = []
        self.dataframes = []
        self.plots = []
        self.infos = []
        self.download_buttons = []
        self.captions = []

    def title(self, text: str) -> None:
        self.title_text = text

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def write(self, text: str) -> None:
        self.writes.append(text)

    def dataframe(self, df: pd.DataFrame) -> None:
        self.dataframes.append(df)

    def pyplot(self, fig) -> None:
        self.plots.append(fig)

    def plotly_chart(self, fig, **_kwargs) -> None:
        self.plots.append(fig)

    def info(self, text: str) -> None:
        self.infos.append(text)

    def download_button(self, label: str, data, file_name: str, mime: str) -> None:
        self.download_buttons.append((label, file_name, mime, data))

    def caption(self, text: str) -> None:
        self.captions.append(text)


def test_build_prompt():
    prompt = build_prompt("Hello {user_input}", "world")
    assert prompt == "Hello world"


def test_parse_tickers():
    tickers = parse_tickers("aapl, msft,  goog", delimiter=",")
    assert tickers == ["AAPL", "MSFT", "GOOG"]


def test_generate_tickers():
    client = DummyClient("AAPL, MSFT")
    tickers = generate_tickers(
        client=client,
        prompt="test",
        system_prompt="system",
        model="model",
        max_tokens=10,
        temperature=0.1,
        delimiter=",")
    assert tickers == ["AAPL", "MSFT"]


def test_fetch_stock_data(monkeypatch):
    def fake_ticker(_symbol: str):
        return DummyTicker()

    monkeypatch.setattr("src.dashboard.yf.Ticker", fake_ticker)

    data = fetch_stock_data("AAPL", history_period="1y", financials_period="quarterly")

    assert "history" in data
    assert "financials" in data
    assert "recommendations" in data
    assert "recommendations_summary" in data
    assert not data["history"].empty
    assert not data["financials"].empty
    assert not data["recommendations"].empty


def test_limit_tickers():
    tickers = ["A", "B", "C"]
    assert limit_tickers(tickers, 2) == ["A", "B"]
    assert limit_tickers(tickers, 0) == []


def test_extract_message_text_dict_response():
    response = {"choices": [{"message": {"content": "AAPL, MSFT"}}]}
    assert _extract_message_text(response) == "AAPL, MSFT"


def test_create_openrouter_client(monkeypatch):
    class DummyOpenAI:
        def __init__(self, api_key: str, base_url: str, default_headers: dict):
            self.api_key = api_key
            self.base_url = base_url
            self.default_headers = default_headers

    monkeypatch.setattr("src.dashboard.OpenAI", DummyOpenAI)

    client = create_openrouter_client(
        api_key="key",
        base_url="https://example.com",
        headers={"X-Test": "1"},
    )

    assert client.api_key == "key"
    assert client.base_url == "https://example.com"
    assert client.default_headers == {"X-Test": "1"}


def _base_config(api_key: str | None) -> dict:
    return {
        "app": {"title": "Title", "layout": "wide"},
        "ui": {
            "sidebar_header": "Header",
            "input_label": "Input",
            "button_label": "Run",
            "suggested_label": "Suggested",
            "ticker_header_template": "Data for {ticker}",
            "section_history": "History",
            "section_financials": "Financials",
            "section_recommendations": "Recommendations",
            "download_prompt": "Download",
            "download_history_label": "History CSV",
            "download_financials_label": "Financials CSV",
            "download_recommendations_label": "Recommendations CSV",
            "missing_api_key": "Missing",
            "max_tickers_warning": "Limiting to {max_tickers}",
            "history_ticker_label": "History tickers",
            "recommendations_missing": "Missing recs for {ticker}",
        },
        "dashboard": {"default_user_input": "default", "ticker_delimiter": ","},
        "openrouter": {
            "api_key": api_key,
            "base_url": "https://example.com",
            "model": "model",
            "max_tokens": 5,
            "temperature": 0.1,
            "system_prompt": "system",
            "prompt_template": "Prompt {user_input}",
            "http_referer": "http://localhost",
            "app_title": "App",
        },
        "stocks": {
            "history_period": "1y",
            "financials_period": "quarterly",
            "max_tickers": 10,
            "financials_metrics": ["Total Revenue"],
        },
        "recommendations": {"current_period": "0m", "previous_period": "1m"},
    }


def test_run_dashboard_missing_api_key(monkeypatch):
    sidebar = DummySidebar(button_value=False)
    st = DummyStreamlit(sidebar)
    monkeypatch.setattr("src.dashboard.st", st)

    run_dashboard(_base_config(api_key=None))

    assert sidebar.error_msg == "Missing"
    assert sidebar.button_called is False


def test_run_dashboard_button_not_clicked(monkeypatch):
    sidebar = DummySidebar(button_value=False)
    st = DummyStreamlit(sidebar)
    monkeypatch.setattr("src.dashboard.st", st)

    def should_not_be_called(*_args, **_kwargs):
        raise AssertionError("unexpected call")

    monkeypatch.setattr("src.dashboard.create_openrouter_client", should_not_be_called)

    run_dashboard(_base_config(api_key="key"))

    assert sidebar.error_msg is None
    assert sidebar.button_called is True


def test_run_dashboard_button_clicked(monkeypatch):
    sidebar = DummySidebar(button_value=True)
    st = DummyStreamlit(sidebar)
    monkeypatch.setattr("src.dashboard.st", st)

    monkeypatch.setattr("src.dashboard.create_openrouter_client", lambda **_kwargs: DummyClient(""))
    monkeypatch.setattr("src.dashboard.generate_tickers", lambda **_kwargs: ["AAPL", "MSFT"])
    monkeypatch.setattr("src.dashboard.plot_history", lambda *_args, **_kwargs: "history")
    monkeypatch.setattr("src.dashboard.plot_financials", lambda *_args, **_kwargs: "financials")
    monkeypatch.setattr("src.dashboard.plot_recommendations", lambda *_args, **_kwargs: "recommendations")
    monkeypatch.setattr(
        "src.dashboard.fetch_stock_data",
        lambda *_args, **_kwargs: {
            "history": pd.DataFrame({"price": [1]}),
            "financials": pd.DataFrame({"metric": ["rev"]}),
            "recommendations": pd.DataFrame({"grade": ["buy"]}),
            "recommendations_summary": pd.DataFrame({"buy": [1]}, index=["0m"]),
        },
    )

    run_dashboard(_base_config(api_key="key"))

    assert sidebar.write_args == ("Suggested", ["AAPL", "MSFT"])
    assert len(st.subheaders) == 2
    assert len(st.dataframes) == 0
    assert len(st.download_buttons) == 6
    assert len(st.plots) == 5