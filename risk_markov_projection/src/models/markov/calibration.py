from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

import config
from src.utils.metrics import delinquency_indicator


def fit_scalar_by_mob(
    actual: Iterable[float],
    predicted: Iterable[float],
    lower: float,
    upper: float,
) -> pd.Series:
    actual_series = pd.Series(actual)
    predicted_series = pd.Series(predicted)
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = actual_series / predicted_series
    factors = factors.replace(np.inf, upper).replace(-np.inf, lower).fillna(1.0)
    return factors.clip(lower, upper)


def apply_scalar_to_distributions(
    df: pd.DataFrame,
    factors: Dict[int, float],
    target_buckets: Sequence[str],
    state_order: List[str],
    mob_col: str = "MOB",
) -> pd.DataFrame:
    if df.empty or not factors:
        return df

    dist_cols = [f"DIST_{s}" for s in state_order]
    delinquent_idx = [state_order.index(s) for s in target_buckets if s in state_order]
    result = df.copy()
    result["calibration_factor"] = 1.0

    for mob, factor in factors.items():
        mask = result[mob_col] == mob
        if not mask.any():
            continue
        dist_values = result.loc[mask, dist_cols].to_numpy(dtype=float)
        ead0 = result.loc[mask, "EAD0"].to_numpy(dtype=float).reshape(-1, 1)
        before_totals = dist_values.sum(axis=1).reshape(-1, 1)

        if delinquent_idx:
            dist_values[:, delinquent_idx] = np.clip(dist_values[:, delinquent_idx] * factor, a_min=0.0, a_max=None)

        after_totals = dist_values.sum(axis=1).reshape(-1, 1)
        scale = np.divide(before_totals, after_totals, out=np.ones_like(before_totals), where=after_totals > 0)
        dist_values = dist_values * scale

        # Renormalize any zero rows to maintain structure
        zero_rows = after_totals.flatten() == 0
        if zero_rows.any():
            dist_values[zero_rows, :] = 0.0

        result.loc[mask, dist_cols] = dist_values
        for idx, state in enumerate(state_order):
            ead_values = dist_values[:, idx].reshape(-1, 1) * ead0
            result.loc[mask, f"EAD_{state}"] = ead_values.ravel()
        result.loc[mask, "calibration_factor"] = factor

    # Recompute indicators after calibration
    for mob, factor in factors.items():
        mask = result[mob_col] == mob
        if not mask.any():
            continue
        dist_matrix = result.loc[mask, dist_cols].to_numpy(dtype=float)
        indicators = []
        for row in dist_matrix:
            indicators.append(
                {
                    "DEL_30P_ON_EAD0": delinquency_indicator(row, state_order, config.BUCKETS_30P),
                    "DEL_60P_ON_EAD0": delinquency_indicator(row, state_order, config.BUCKETS_60P),
                    "DEL_90P_ON_EAD0": delinquency_indicator(row, state_order, config.BUCKETS_90P),
                }
            )
        ind_df = pd.DataFrame(indicators, index=result.index[mask])
        for col in ind_df.columns:
            result.loc[mask, col] = ind_df[col].values

    return result
