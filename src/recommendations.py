"""Recommendation summary helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

CATEGORIES: List[str] = ["StrongBuy", "Buy", "Hold", "Sell", "StrongSell"]


def normalize_recommendation_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = {col: col.strip().lower() for col in df.columns}
    rename_map = {
        "strongbuy": "StrongBuy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strongsell": "StrongSell",
    }

    return df.rename(columns={col: rename_map.get(norm, col) for col, norm in normalized.items()})


def normalize_period_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    if "period" in normalized.columns:
        normalized["period"] = normalized["period"].astype(str)
        normalized = normalized.set_index("period")

    normalized.index = normalized.index.astype(str).str.strip().str.lower()
    return normalized


def _period_candidates(period: str) -> List[str]:
    normalized = str(period).strip().lower()
    candidates = [normalized]
    if normalized.startswith("-"):
        candidates.append(normalized[1:])
    elif normalized and normalized != "0m":
        candidates.append(f"-{normalized}")
    return list(dict.fromkeys(candidates))


def extract_period_row(df: pd.DataFrame, period: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for candidate in _period_candidates(period):
        if candidate in df.index:
            return df.loc[candidate]
    return None


def summarize_recommendations_counts(
    summary: pd.DataFrame,
    current_period: str,
    previous_period: str,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    if summary is None or summary.empty:
        return {}, None

    summary = normalize_recommendation_columns(summary)
    summary = normalize_period_index(summary)

    current = extract_period_row(summary, current_period)
    previous = extract_period_row(summary, previous_period)
    if current is None:
        return {}, None

    current_vals = current.reindex(CATEGORIES).fillna(0)

    delta_vals = None
    if previous is not None:
        delta_vals = current_vals - previous.reindex(CATEGORIES).fillna(0)

    current_dict = {cat: float(current_vals.get(cat, 0)) for cat in CATEGORIES}
    delta_dict = None
    if delta_vals is not None:
        delta_dict = {cat: float(delta_vals.get(cat, 0)) for cat in CATEGORIES}

    return current_dict, delta_dict
