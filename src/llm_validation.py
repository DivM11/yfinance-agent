"""Validation helpers for LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Tuple

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def extract_valid_tickers(text: str, delimiter: str) -> List[str]:
    split_pattern = r"[\s,;|]+"
    if delimiter and delimiter not in [",", ";", "|"]:
        split_pattern = rf"(?:\s+|{re.escape(delimiter)}|,|;|\|)+"

    candidates = [item.strip().upper() for item in re.split(split_pattern, text) if item.strip()]
    deduped: List[str] = []
    for ticker in candidates:
        if ticker not in deduped and TICKER_PATTERN.match(ticker):
            deduped.append(ticker)
    return deduped


def has_valid_tickers(tickers: Iterable[str]) -> bool:
    for ticker in tickers:
        if TICKER_PATTERN.match(str(ticker).upper()):
            return True
    return False


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json_string(text: str) -> str:
    """Extract a JSON object from LLM output that may contain markdown
    code fences, ``<think>`` blocks, or surrounding prose."""
    cleaned = _THINK_TAG_RE.sub("", text).strip()

    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        return fence.group(1).strip()

    for open_ch, close_ch in ["{", "}"], ["[", "]"]:
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            return cleaned[start:end + 1]

    return cleaned


def parse_weights_payload(text: str) -> Dict[str, float]:
    cleaned = _extract_json_string(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    weights: Dict[str, float] = {}
    if isinstance(payload, dict):
        source = payload.get("weights", payload)
        if isinstance(source, dict):
            for ticker, value in source.items():
                try:
                    weights[str(ticker).upper()] = float(value)
                except (TypeError, ValueError):
                    continue
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "ticker" in item and "weight" in item:
                try:
                    weights[str(item["ticker"]).upper()] = float(item["weight"])
                except (TypeError, ValueError):
                    continue

    return weights


def validate_weight_sum(weights: Dict[str, float], tolerance: float = 0.02) -> Tuple[bool, float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    return abs(total - 1.0) <= tolerance, total
