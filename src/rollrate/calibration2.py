# src/rollrate/calibration.py

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from src.config import CFG, BUCKETS_CANON
from src.rollrate.forecast import forecast_all_vintages
from src.rollrate.lifecycle import (
    get_actual_all_vintages_amount,
    combine_all_lifecycle_amount,
    lifecycle_to_long_df_amount,
    add_del_metrics,
)

# ============================================================
# 1) Build lifecycle ACTUAL ONLY (EAD + DEL metrics)
# ============================================================

def build_actual_lifecycle_amount_only(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Lifecycle chỉ dùng dữ liệu thực tế (không forecast).
    Trả về long-format + DEL30/60/90 + DISB_TOTAL.
    """

    # dict: {(product, score, vintage): {mob: Series(EAD per state)}}
    actual_results = get_actual_all_vintages_amount(df_raw)

    # Long format theo state
    df_actual_states = lifecycle_to_long_df_amount(actual_results)

    # DEL metrics + DISB_TOTAL
    df_actual = add_del_metrics(df_actual_states, df_raw)

    # Gắn cờ actual
    df_actual["IS_FORECAST"] = 0

    return df_actual


# ============================================================
# 2) Build lifecycle MODEL ONLY (FORECAST) cho backtest
# ============================================================

def build_model_lifecycle_amount_only(
    df_raw: pd.DataFrame,
    matrices_by_mob: Dict,
    max_mob: int = 29,
    enable_macro: bool = False,
    macro_params: dict | None = None,
) -> pd.DataFrame:
    """
    Lifecycle thuần FORECAST (từ Markov) cho chính các vintage lịch sử,
    dùng để backtest & calibration.

    forecast_results:
        {(product, score, vintage): {mob: Series(EAD per state)}}
    """

    forecast_results = forecast_all_vintages(
        df_raw=df_raw,
        matrices_by_mob=matrices_by_mob,
        max_mob=max_mob,
        enable_macro=enable_macro,
        macro_params=macro_params,
    )

    # Long format theo state
    df_model_states = lifecycle_to_long_df_amount(forecast_results)

    # DEL metrics + DISB_TOTAL
    df_model = add_del_metrics(df_model_states, df_raw)

    # Gắn cờ forecast
    df_model["IS_FORECAST"] = 1

    return df_model


# ============================================================
# 3) Calibration A — k per PRODUCT
# ============================================================

def compute_k_per_product_auto(
    df_actual: pd.DataFrame,
    df_model: pd.DataFrame,
    horizon_mob: int = 12,
    metric_col: str = "DEL90_PCT",
    min_cohort: int = 5,
    clip_range: Tuple[float, float] = (0.5, 1.5),
) -> Dict[str, float]:
    """
    Option A:
        k_product = mean(Actual metric @ MOB=horizon)
                    / mean(Model metric @ MOB=horizon)

    df_actual: lifecycle ACTUAL ONLY (build_actual_lifecycle_amount_only + add_del_metrics)
    df_model : lifecycle MODEL ONLY   (build_model_lifecycle_amount_only + add_del_metrics)
    """

    k_dict: Dict[str, float] = {}

    # Lọc horizon
    ac_h = df_actual[df_actual["MOB"] == horizon_mob].copy()
    md_h = df_model[df_model["MOB"] == horizon_mob].copy()

    products = sorted(set(ac_h["PRODUCT_TYPE"]).intersection(md_h["PRODUCT_TYPE"]))

    for product in products:
        ac_p = ac_h[ac_h["PRODUCT_TYPE"] == product]
        md_p = md_h[md_h["PRODUCT_TYPE"] == product]

        if ac_p.empty or md_p.empty:
            print(f"⚠️ {product}: không đủ dữ liệu tại MOB={horizon_mob}.")
            continue

        # Join theo cohort nhỏ: Product × Score × Vintage
        merge_cols = ["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"]

        ac_s = ac_p[merge_cols + [metric_col]].rename(columns={metric_col: metric_col + "_AC"})
        md_s = md_p[merge_cols + [metric_col]].rename(columns={metric_col: metric_col + "_MD"})

        df = ac_s.merge(md_s, on=merge_cols, how="inner")

        if len(df) < min_cohort:
            print(f"⚠️ {product}: số cohort < {min_cohort}, skip.")
            continue

        L_ac = df[metric_col + "_AC"].mean()
        L_md = df[metric_col + "_MD"].mean()

        if L_md <= 0:
            print(f"⚠️ {product}: model {metric_col} = 0, skip.")
            continue

        k = L_ac / L_md
        k = float(np.clip(k, clip_range[0], clip_range[1]))

        k_dict[product] = k
        print(
            f"✔ Calib A – {product}: k={k:.3f} "
            f"(actual={L_ac:.4f}, model={L_md:.4f}, n={len(df)})"
        )

    return k_dict


def apply_product_calibration(
    df_lifecycle: pd.DataFrame,
    k_dict: Dict[str, float],
    metric_cols=("DEL30_PCT", "DEL60_PCT", "DEL90_PCT"),
    only_forecast: bool = True,
) -> pd.DataFrame:
    """
    Áp hệ số k per product cho toàn bộ forecast:
        metric_adj = metric_raw * k_product

    Thường dùng cho các cột DELxx_PCT.
    """

    df = df_lifecycle.copy()

    # Mask forecast
    if only_forecast and ("IS_FORECAST" in df.columns):
        base_mask = df["IS_FORECAST"] == 1
    else:
        base_mask = pd.Series(True, index=df.index)

    for product, k in k_dict.items():
        mask = base_mask & (df["PRODUCT_TYPE"] == product)

        for col in metric_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] * k

    return df


# ============================================================
# 4) Calibration B — Seasonality per Month of Origination
# ============================================================

def compute_month_seasonality(
    df_actual: pd.DataFrame,
    horizon_mob: int = 12,
    metric_col: str = "DEL90_PCT",
    min_cohort: int = 5,
    clip_range: Tuple[float, float] = (0.7, 1.3),
) -> Dict[int, float]:
    """
    Tính seasonality factor theo tháng giải ngân (VINTAGE_DATE.month) từ dữ liệu thực.

    F_month = mean(metric @ MOB=horizon của các vintage tháng đó)
              / mean(metric @ MOB=horizon toàn sample)
    """

    df = df_actual.copy()
    df = df[df["MOB"] == horizon_mob].copy()

    if df.empty:
        print("⚠️ compute_month_seasonality: không có dữ liệu tại horizon.")
        return {}

    df["VINTAGE_MONTH"] = df["VINTAGE_DATE"].dt.month

    # Loss metric per cohort nhỏ
    cohort = (
        df.groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "VINTAGE_MONTH"])[metric_col]
        .mean()
        .reset_index()
    )

    # Trung bình theo month
    month_loss = (
        cohort.groupby("VINTAGE_MONTH")[metric_col]
        .mean()
        .rename("LOSS_MONTH")
        .reset_index()
    )

    # Số cohort theo month
    month_counts = (
        cohort.groupby("VINTAGE_MONTH")[metric_col]
        .count()
        .rename("N_COHORT")
        .reset_index()
    )

    month_loss = month_loss.merge(month_counts, on="VINTAGE_MONTH", how="left")

    # Chỉ giữ month có đủ cohort
    month_loss = month_loss[month_loss["N_COHORT"] >= min_cohort].copy()
    if month_loss.empty:
        print("⚠️ compute_month_seasonality: không có month nào đủ cohort.")
        return {}

    L_avg = month_loss["LOSS_MONTH"].mean()
    if L_avg <= 0:
        print("⚠️ compute_month_seasonality: L_avg <= 0.")
        return {}

    month_loss["FACTOR"] = month_loss["LOSS_MONTH"] / L_avg
    month_loss["FACTOR"] = month_loss["FACTOR"].clip(lower=clip_range[0], upper=clip_range[1])

    month_factor = month_loss.set_index("VINTAGE_MONTH")["FACTOR"].to_dict()

    print("✔ Calib B – Month seasonality factors:")
    for m, f in sorted(month_factor.items()):
        print(f"  Month {m:02d}: factor={f:.3f} (n={int(month_loss[month_loss['VINTAGE_MONTH']==m]['N_COHORT'].iloc[0])})")

    return month_factor


def apply_month_seasonality(
    df_lifecycle: pd.DataFrame,
    month_factor: Dict[int, float],
    metric_cols=("DEL30_PCT", "DEL60_PCT", "DEL90_PCT"),
    only_forecast: bool = True,
    col_factor_name: str = "SEASON_FACTOR",
) -> pd.DataFrame:
    """
    Áp seasonality factor theo tháng giải ngân cho forecast rows.

      metric_adj = metric * F_month

    month_factor: {1..12: factor}
    """

    df = df_lifecycle.copy()

    if df.empty:
        return df

    df["VINTAGE_MONTH"] = df["VINTAGE_DATE"].dt.month
    df[col_factor_name] = df["VINTAGE_MONTH"].map(month_factor).fillna(1.0)

    if only_forecast and ("IS_FORECAST" in df.columns):
        base_mask = df["IS_FORECAST"] == 1
    else:
        base_mask = pd.Series(True, index=df.index)

    for col in metric_cols:
        if col in df.columns:
            df.loc[base_mask, col] = df.loc[base_mask, col] * df.loc[base_mask, col_factor_name]

    return df
def compute_k_per_product(df_actual: pd.DataFrame, df_forecast: pd.DataFrame):
    """
    Tính hệ số calibration k per product dựa trên DEL90_PCT:
        k_p = mean(actual DEL90) / mean(forecast DEL90)

    df_actual  : lifecycle (actual-only rows → IS_FORECAST=0)
    df_forecast: lifecycle (forecast-only rows → IS_FORECAST=1)
    """
    k_dict = {}
    products = df_actual["PRODUCT_TYPE"].unique()

    for prod in products:
        ac = df_actual[df_actual["PRODUCT_TYPE"] == prod]["DEL90_PCT"].mean()
        fc = df_forecast[df_forecast["PRODUCT_TYPE"] == prod]["DEL90_PCT"].mean()

        if pd.isna(ac) or pd.isna(fc) or fc == 0:
            k_dict[prod] = 1.0
        else:
            k_dict[prod] = ac / fc

    print("✔ k per product:")
    for p, k in k_dict.items():
        print(f"  {p}: k = {k:.4f}")

    return k_dict
def apply_k_calibration(df_lifecycle: pd.DataFrame, k_dict: dict):
    """
    Áp calibration k per product vào forecast rows:
        DELxx_PCT_calibrated = DELxx_PCT_raw * k_product
    """
    df = df_lifecycle.copy()

    # Chỉ calibrate forecast rows
    df_fc = df[df["IS_FORECAST"] == 1].copy()

    for prod, k in k_dict.items():
        mask = df_fc["PRODUCT_TYPE"] == prod
        df_fc.loc[mask, "DEL30_PCT"] *= k
        df_fc.loc[mask, "DEL60_PCT"] *= k
        df_fc.loc[mask, "DEL90_PCT"] *= k

    # Combine actual + calibrated forecast
    df_out = pd.concat([df[df["IS_FORECAST"] == 0], df_fc], ignore_index=True)

    df_out = df_out.sort_values(
        ["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"]
    ).reset_index(drop=True)

    return df_out
