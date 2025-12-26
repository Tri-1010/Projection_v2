"""
calibration.py – Anchor-based calibration (IFRS-style)

Ý tưởng:
    - Dùng DEL90_PCT tại 1 MOB anchor (per product) để tính k:
          k_product = trimmed_mean( DEL90_actual / DEL90_forecast_full )
    - CHỈ áp k:
          + cho FORECAST (IS_FORECAST == 1)
          + trên lifecycle lịch sử: từ MOB >= H_APPLY(product)
          + cho SALE PLAN: tất cả MOB (vì không có actual)

Yêu cầu input:
    df_actual_lifecycle:
        - PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, MOB
        - DEL90_PCT
        - IS_FORECAST (0 = actual, 1 = forecast)

    df_full_forecast:
        - PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, MOB
        - DEL90_PCT (full-history forecast từ MOB0)

Ghi chú:
    - Hàm compute_k_per_product_ifrs chỉ tính k.
    - apply_k_to_lifecycle chỉ scale phần forecast trong lịch sử.
    - apply_k_to_sale_plan scale toàn bộ forecast của sale plan (forecast 100%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ============================================================
# 1️⃣ CẤU HÌNH ANCHOR MOB
# ============================================================

# MOB dùng để TÍNH k (calibration point)
H_MAP_CALIB: Dict[str, int] = {
    "CDLPIL": 12,
    "TWLPIL": 12,
    "SPLPIL": 12,
    # Các sản phẩm khác không khai báo → dùng default_anchor_mob
}

# MOB dùng để ÁP k trên lịch sử (áp từ đây trở đi, chỉ cho forecast)
H_MAP_APPLY: Dict[str, int] = {
    "CDLPIL": 4,
    "TWLPIL": 4,
    "SPLPIL": 4,
    "SALPIL": 5,
    "TOPUP": 6,
    "XSELL": 6,
}


def _get_anchor_calib(product: str, default_anchor_mob: int) -> int:
    return H_MAP_CALIB.get(product, default_anchor_mob)


def _get_anchor_apply(product: str, default_apply_mob: int) -> int:
    return H_MAP_APPLY.get(product, default_apply_mob)


# ============================================================
# 2️⃣ TRIMMED MEAN HELPER
# ============================================================

def trimmed_mean(x: np.ndarray, trim_prop: float = 0.2) -> float:
    """
    Tính trimmed mean:
        - loại bỏ trim_prop/2 ở mỗi tail
        - vd: trim_prop=0.2 → bỏ 10% nhỏ nhất + 10% lớn nhất

    Nếu số lượng quan sát ít → tự động fallback về median.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan
    if x.size < 5:
        # quá ít quan sát → median ổn định hơn
        return float(np.median(x))

    x_sorted = np.sort(x)
    n = x_sorted.size
    k = int(n * trim_prop / 2)

    if k == 0:
        return float(np.mean(x_sorted))

    trimmed = x_sorted[k : n - k]
    if trimmed.size == 0:
        return float(np.mean(x_sorted))

    return float(np.mean(trimmed))


# ============================================================
# 3️⃣ EXTRACT DEL90_PCT (ACTUAL & FULL-FORECAST)
# ============================================================

def extract_actual_del90(
    df_lifecycle: pd.DataFrame,
    require_actual_flag: bool = True,
) -> pd.DataFrame:
    """
    df_lifecycle:
        - phải có cột DEL90_PCT
        - nếu require_actual_flag=True → cần cột IS_FORECAST để lọc actual

    Trả về:
        PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, MOB, DEL90_PCT_ACT
    """
    if "DEL90_PCT" not in df_lifecycle.columns:
        raise KeyError("df_lifecycle chưa có DEL90_PCT – hãy chạy add_del_metrics trước.")

    df = df_lifecycle.copy()

    if require_actual_flag:
        if "IS_FORECAST" not in df.columns:
            raise KeyError("df_lifecycle cần cột IS_FORECAST để tách actual vs forecast.")
        df = df[df["IS_FORECAST"] == 0].copy()

    df_act = (
        df.groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"])["DEL90_PCT"]
        .max()
        .rename("DEL90_ACT")
        .reset_index()
    )

    return df_act


def extract_forecast_del90(
    df_full_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    df_full_forecast:
        - long-format full-history forecast
        - đã có DEL90_PCT

    Trả về:
        PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, MOB, DEL90_PCT_FC
    """
    if "DEL90_PCT" not in df_full_forecast.columns:
        raise KeyError("df_full_forecast chưa có DEL90_PCT – cần tính trước khi calibration.")

    df_fc = (
        df_full_forecast
        .groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"])["DEL90_PCT"]
        .max()
        .rename("DEL90_FC")
        .reset_index()
    )

    return df_fc


# ============================================================
# 4️⃣ TÍNH k PER PRODUCT (IFRS STYLE, ANCHOR MOB)
# ============================================================

def compute_k_per_product_ifrs(
    df_actual_lifecycle: pd.DataFrame,
    df_full_forecast: pd.DataFrame,
    default_anchor_mob: int = 24,
    min_obs: int = 3,
    method: str = "trimmed_mean",
    trim_prop: float = 0.2,
) -> Dict[str, float]:
    """
    Tính k per product bằng cách so sánh DEL90_PCT_actual vs DEL90_PCT_full-forecast
    tại MOB anchor.

    df_actual_lifecycle:
        - lifecycle đã có DEL90_PCT, IS_FORECAST
        - có cả actual + forecast, nhưng ta chỉ dùng actual

    df_full_forecast:
        - full-history forecast từ MOB0
        - đã có DEL90_PCT

    Output:
        k_dict = {product: k}

    method:
        - "median"
        - "trimmed_mean" (mặc định, trim_prop=0.2)
    """

    actual_anchor = extract_actual_del90(df_actual_lifecycle, require_actual_flag=True)
    forecast_anchor = extract_forecast_del90(df_full_forecast)

    results: Dict[str, float] = {}

    products = sorted(actual_anchor["PRODUCT_TYPE"].unique())

    for product in products:
        H = _get_anchor_calib(product, default_anchor_mob=default_anchor_mob)

        act_p = actual_anchor[
            (actual_anchor["PRODUCT_TYPE"] == product) &
            (actual_anchor["MOB"] == H)
        ]
        fc_p = forecast_anchor[
            (forecast_anchor["PRODUCT_TYPE"] == product) &
            (forecast_anchor["MOB"] == H)
        ]

        merged = act_p.merge(
            fc_p,
            on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"],
            how="inner",
        )

        if merged.shape[0] < min_obs:
            results[product] = 1.0
            continue

        ratios = merged["DEL90_ACT"] / merged["DEL90_FC"].replace(0, np.nan)
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

        if ratios.empty:
            results[product] = 1.0
            continue

        if method == "median":
            k_val = float(np.median(ratios.values))
        else:  # trimmed_mean (mặc định)
            k_val = trimmed_mean(ratios.values, trim_prop=trim_prop)

        # Limit để tránh hệ số cực đoan
        k_val = float(np.clip(k_val, 0.3, 3.0))

        results[product] = k_val

    return results


# ============================================================
# 5️⃣ ÁP k VÀO LIFECYCLE (HISTORICAL)
# ============================================================

def apply_k_to_lifecycle(
    df_lifecycle: pd.DataFrame,
    k_dict: Dict[str, float],
    default_apply_mob: int = 24,
) -> pd.DataFrame:
    """
    Áp k cho lifecycle lịch sử:
        - KHÔNG đụng vào actual (IS_FORECAST == 0)
        - Chỉ scale forecast (IS_FORECAST == 1)
        - Chỉ scale từ MOB >= H_APPLY(product)

    df_lifecycle:
        - phải có:
            PRODUCT_TYPE, MOB, IS_FORECAST
            các bucket DPD*, WRITEOFF, PREPAY
    """

    df = df_lifecycle.copy()

    if "IS_FORECAST" not in df.columns:
        raise KeyError("apply_k_to_lifecycle: df_lifecycle cần IS_FORECAST (0/1).")

    df["CALIB_K"] = df["PRODUCT_TYPE"].map(lambda p: k_dict.get(p, 1.0))
    df["H_APPLY"] = df["PRODUCT_TYPE"].map(
        lambda p: _get_anchor_apply(p, default_apply_mob=default_apply_mob)
    )

    bucket_cols = [
        c for c in df.columns
        if c.startswith("DPD") or c in ["WRITEOFF", "PREPAY"]
    ]

    # Chỉ scale:
    #  - forecast
    #  - MOB >= H_APPLY(product)
    mask = (df["IS_FORECAST"] == 1) & (df["MOB"] >= df["H_APPLY"])

    for c in bucket_cols:
        df.loc[mask, c] = df.loc[mask, c] * df.loc[mask, "CALIB_K"]

    return df


# ============================================================
# 6️⃣ ÁP k VÀO SALE PLAN (TẤT CẢ MOB)
# ============================================================

def apply_k_to_sale_plan(
    df_plan_fc: pd.DataFrame,
    k_dict: Dict[str, float],
) -> pd.DataFrame:
    """
    Áp k cho SALE PLAN:

    - Sale plan không có actual, mọi dòng đều là forecast.
    - Vì vậy ÁP k cho TẤT CẢ MOB (MOB0..target).

    df_plan_fc:
        - PRODUCT_TYPE
        - MOB
        - các bucket EAD (DPD*, WRITEOFF, PREPAY)
    """

    df = df_plan_fc.copy()

    df["CALIB_K"] = df["PRODUCT_TYPE"].map(lambda p: k_dict.get(p, 1.0))

    bucket_cols = [
        c for c in df.columns
        if c.startswith("DPD") or c in ["WRITEOFF", "PREPAY"]
    ]

    for c in bucket_cols:
        df[c] = df[c] * df["CALIB_K"]

    return df
