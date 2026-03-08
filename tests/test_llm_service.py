"""Unit tests for shared LLM service."""

from src.llm_service import LLMService, build_prompt, extract_message_text


class DummyRawResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def parse(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class DummyWithRaw:
    def create(self, **_kwargs):
        return DummyRawResponse()


class DummyCompletions:
    def __init__(self):
        self.with_raw_response = DummyWithRaw()

    def create(self, **_kwargs):
        return {"choices": [{"message": {"content": "fallback"}}]}


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyClient:
    def __init__(self):
        self.chat = DummyChat()


def test_build_prompt():
    assert build_prompt("Hello {user_input}", "world") == "Hello world"


def test_extract_message_text_dict():
    assert extract_message_text({"choices": [{"message": {"content": "x"}}]}) == "x"


def test_llm_service_complete_with_raw_response():
    service = LLMService(DummyClient())

    response, status = service.complete(
        request_name="test",
        model="anthropic/claude-3.5-haiku",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert status == 200
    assert response["choices"][0]["message"]["content"] == "ok"
