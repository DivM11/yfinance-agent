"""Unit tests for LLM validation helpers."""

from src.llm_validation import (
    extract_valid_tickers,
    has_valid_tickers,
    parse_weights_payload,
    validate_weight_sum,
)


def test_extract_valid_tickers_filters_noise():
    tickers = extract_valid_tickers("AAPL, ???, MSFT, AAPL", delimiter=",")

    assert tickers == ["AAPL", "MSFT"]


def test_extract_valid_tickers_space_separated():
    tickers = extract_valid_tickers("AAPL GOOGLE\nMSFT", delimiter=",")

    assert tickers == ["AAPL", "GOOGLE", "MSFT"]


def test_has_valid_tickers():
    assert has_valid_tickers(["AAPL", "MSFT"]) is True
    assert has_valid_tickers(["", "???"]) is False


def test_parse_weights_payload_object_and_nested():
    nested = parse_weights_payload('{"weights": {"AAPL": 0.6, "MSFT": 0.4}}')
    plain = parse_weights_payload('{"AAPL": 0.7, "MSFT": 0.3}')

    assert nested["AAPL"] == 0.6
    assert plain["MSFT"] == 0.3


def test_validate_weight_sum():
    ok, total_ok = validate_weight_sum({"AAPL": 0.6, "MSFT": 0.4})
    bad, total_bad = validate_weight_sum({"AAPL": 0.6, "MSFT": 0.2})

    assert ok is True
    assert round(total_ok, 2) == 1.0
    assert bad is False
    assert round(total_bad, 2) == 0.8
