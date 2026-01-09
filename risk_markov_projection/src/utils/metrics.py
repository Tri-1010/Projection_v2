from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, List, Sequence


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_arr = np.asarray(list(actual), dtype=float)
    pred_arr = np.asarray(list(predicted), dtype=float)
    return float(np.mean(np.abs(actual_arr - pred_arr)))


def wape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_arr = np.asarray(list(actual), dtype=float)
    pred_arr = np.asarray(list(predicted), dtype=float)
    denom = np.sum(np.abs(actual_arr))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(actual_arr - pred_arr)) / denom)


def delinquency_indicator(dist: Sequence[float], state_order: List[str], buckets: List[str]) -> float:
    state_to_idx = {s: i for i, s in enumerate(state_order)}
    return float(sum(dist[state_to_idx[s]] for s in buckets if s in state_to_idx))


def aggregate_indicator_by_mob(
    df: pd.DataFrame,
    indicator_col: str,
    weight_col: str,
    mob_col: str = "MOB",
) -> pd.Series:
    grouped = df.groupby(mob_col)
    weighted = grouped.apply(
        lambda g: np.average(g[indicator_col], weights=g[weight_col]) if g[weight_col].sum() > 0 else g[indicator_col].mean(),
        include_groups=False,
    )
    return weighted
