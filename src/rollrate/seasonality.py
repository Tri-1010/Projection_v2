"""
seasonality.py
----------------
Tính seasonality theo month(vintage) và điều chỉnh lifecycle / sale plan.
"""

import pandas as pd
import numpy as np
from typing import Dict


# ============================================================
# 1️⃣ Lấy DEL90 actual per cohort
# ============================================================

def compute_actual_del90_pct(df_lifecycle: pd.DataFrame) -> pd.DataFrame:

    if "DEL90_PCT" not in df_lifecycle.columns:
        raise KeyError("df_lifecycle chưa có DEL90_PCT — hãy chạy add_del_metrics trước.")

    df = (
        df_lifecycle
        .groupby(["PRODUCT_TYPE", "VINTAGE_DATE"])["DEL90_PCT"]
        .max()
        .reset_index()
    )

    df["MONTH"] = df["VINTAGE_DATE"].dt.month

    return df


# ============================================================
# 2️⃣ Raw seasonality = avg(month) / avg(overall)
# ============================================================

def compute_seasonality_raw(df_del90: pd.DataFrame) -> Dict[str, pd.Series]:

    season_raw = {}

    for product, grp in df_del90.groupby("PRODUCT_TYPE"):

        overall = grp["DEL90_PCT"].mean()
        month_avg = grp.groupby("MONTH")["DEL90_PCT"].mean()

        season = (month_avg / overall).reindex(range(1, 13), fill_value=1.0)
        season_raw[product] = season

    return season_raw


# ============================================================
# 3️⃣ Smooth
# ============================================================

def smooth_seasonality(season: pd.Series, window: int = 3) -> pd.Series:
    return season.rolling(window=window, center=True, min_periods=1).mean()


# ============================================================
# 4️⃣ Normalize to mean = 1
# ============================================================

def normalize_seasonality(season: pd.Series) -> pd.Series:
    m = season.mean()
    return season / m if m != 0 else season


# ============================================================
# 5️⃣ Full seasonality builder
# ============================================================

def build_seasonality(df_lifecycle: pd.DataFrame, smooth_window: int = 3) -> Dict[str, pd.Series]:

    df_del90 = compute_actual_del90_pct(df_lifecycle)
    raw = compute_seasonality_raw(df_del90)

    seasonality = {}

    for product, s in raw.items():
        ss = smooth_seasonality(s, window=smooth_window)
        ss = normalize_seasonality(ss)
        seasonality[product] = ss

    return seasonality


# ============================================================
# 6️⃣ Apply seasonality vào lifecycle
# ============================================================

def apply_seasonality_to_lifecycle(df_lifecycle: pd.DataFrame,
                                   seasonality: Dict[str, pd.Series]) -> pd.DataFrame:

    df = df_lifecycle.copy()
    df["MONTH"] = df["VINTAGE_DATE"].dt.month

    df["SEASON_FACTOR"] = df.apply(
        lambda r: seasonality.get(r["PRODUCT_TYPE"], {}).get(r["MONTH"], 1.0),
        axis=1
    )

    bucket_cols = [c for c in df.columns if c.startswith("DPD") or c in ["WRITEOFF", "PREPAY"]]

    for c in bucket_cols:
        df[c] *= df["SEASON_FACTOR"]

    return df


# ============================================================
# 7️⃣ Apply seasonality vào Sale Plan FC
# ============================================================

def apply_seasonality_to_sale_plan(df_plan_fc: pd.DataFrame,
                                   seasonality: Dict[str, pd.Series]) -> pd.DataFrame:

    df = df_plan_fc.copy()
    df["MONTH"] = df["VINTAGE_DATE"].dt.month

    df["SEASON_FACTOR"] = df.apply(
        lambda r: seasonality.get(r["PRODUCT_TYPE"], {}).get(r["MONTH"], 1.0),
        axis=1
    )

    bucket_cols = [c for c in df.columns if c.startswith("DPD") or c in ["WRITEOFF", "PREPAY"]]

    for c in bucket_cols:
        df[c] *= df["SEASON_FACTOR"]

    return df
