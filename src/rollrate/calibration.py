import pandas as pd
import numpy as np
from typing import Dict

# ------------------------------
# Cấu hình anchor
# ------------------------------

H_MAP_CALIB = {
    "CDLPIL": 12,
    "TWLPIL": 12,
    "SPLPIL": 12,
}

# MOB bắt đầu áp k
M_APPLY_MAP = {
    "CDLPIL": 4,
    "SALPIL": 4,
    "SPLPIL": 4,
    "TOPUP": 4,
    "TWLPIL": 4,
    "XSELL": 4,
}
DEFAULT_M_APPLY = 4


def get_anchor_calib(product: str, default: int = 24):
    return H_MAP_CALIB.get(product, default)


# ------------------------------
# Helper: trimmed mean
# ------------------------------

def trimmed_mean(x, trim=0.2):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    if x.size < 5:
        return float(np.median(x))

    x_sorted = np.sort(x)
    n = x_sorted.size
    k = int(n * trim / 2)
    if k == 0:
        return float(np.mean(x_sorted))
    trimmed = x_sorted[k:n-k]
    if trimmed.size == 0:
        return float(np.mean(x_sorted))
    return float(np.mean(trimmed))


# ------------------------------
# Extract actual & forecast at MOB = H
# ------------------------------

def extract_actual(df):
    return (
        df.groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"])["DEL90_PCT"]
        .max().reset_index().rename(columns={"DEL90_PCT": "DEL90_ACT"})
    )

def extract_forecast(df):
    return (
        df.groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"])["DEL90_PCT"]
        .max().reset_index().rename(columns={"DEL90_PCT": "DEL90_FC"})
    )


# ------------------------------
# MAIN: compute k per product
# ------------------------------

def compute_k_per_product_ifrs_fullhistory(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    H_map: dict = None,
    min_obs: int = 3,
    default_anchor_mob: int = 24,
    method: str = "trimmed_mean",
    clip_min: float = 0.3,
    clip_max: float = 3.0,
):
    """
    Compute k per product using full-history forecast vs actual.
    """

    if H_map is None:
        H_map = {}

    # --- Extract actual ---
    actual_full = (
        df_actual.groupby(["PRODUCT_TYPE","RISK_SCORE","VINTAGE_DATE","MOB"])["DEL90_PCT"]
        .max().reset_index().rename(columns={"DEL90_PCT": "DEL90_ACT"})
    )

    # --- Extract forecast FULL HISTORY ---
    fc_full = (
        df_forecast.groupby(["PRODUCT_TYPE","RISK_SCORE","VINTAGE_DATE","MOB"])["DEL90_PCT"]
        .max().reset_index().rename(columns={"DEL90_PCT": "DEL90_FC"})
    )

    products = sorted(actual_full["PRODUCT_TYPE"].unique())
    results = {}

    for p in products:

        # 🟦 Lấy anchor MOB theo product
        H = H_map.get(p, default_anchor_mob)

        act_p = actual_full[
            (actual_full["PRODUCT_TYPE"] == p) &
            (actual_full["MOB"] == H)
        ]
        fc_p  = fc_full[
            (fc_full["PRODUCT_TYPE"] == p) &
            (fc_full["MOB"] == H)
        ]

        merged = act_p.merge(
            fc_p,
            on=["PRODUCT_TYPE","RISK_SCORE","VINTAGE_DATE","MOB"],
            how="inner"
        )

        if merged.shape[0] < min_obs:
            print(f"[WARN] Product {p}: sample quá ít, k = 1.0")
            results[p] = 1.0
            continue

        ratios = merged["DEL90_ACT"] / merged["DEL90_FC"].replace(0, np.nan)
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

        if ratios.empty:
            print(f"[WARN] Product {p}: ratio rỗng → k = 1.0")
            results[p] = 1.0
            continue

        # 🟦 trimmed_mean / median
        if method == "median":
            k = float(np.median(ratios.values))
        else:
            k = trimmed_mean(ratios.values, trim=0.2)

        # 🟦 Clip theo khoảng quy định
        k = float(np.clip(k, clip_min, clip_max))
        results[p] = k

    return results


def compute_k_per_product_by_mob_ifrs_fullhistory(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    H_map: dict = None,
    min_obs: int = 5,
    default_anchor_mob: int = 24,
    metric_col: str = "DEL90_PCT",
    method: str = "trimmed_mean",
    clip_min: float = 0.5,
    clip_max: float = 1.5,
    tail_decay_n: int | None = None,
) -> Dict[str, Dict[int, float]]:
    """
    Compute k per product per MOB using full-history actual vs forecast.
    - Only compute k up to anchor MOB (H). After H, k = 1.0.
    - H is taken from H_map or get_anchor_calib(product, default_anchor_mob).
    - metric_col default is DEL90_PCT.
    - If tail_decay_n is set and a valid k is found at/before H, blend k -> 1.0
      linearly over tail_decay_n MOBs for mob > H to avoid a hard step.
    """

    if H_map is None:
        H_map = {}

    actual_full = (
        df_actual.groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"])[metric_col]
        .max().reset_index().rename(columns={metric_col: "METRIC_ACT"})
    )

    fc_full = (
        df_forecast.groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"])[metric_col]
        .max().reset_index().rename(columns={metric_col: "METRIC_FC"})
    )

    merged = actual_full.merge(
        fc_full,
        on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"],
        how="inner",
    )

    results: Dict[str, Dict[int, float]] = {}

    for product in sorted(merged["PRODUCT_TYPE"].unique()):
        H = H_map.get(product, get_anchor_calib(product, default_anchor_mob))

        df_p = merged[merged["PRODUCT_TYPE"] == product].copy()
        if df_p.empty:
            continue

        k_by_mob: Dict[int, float] = {}
        last_k: float | None = None

        # Compute k for each MOB <= H. After H -> k = 1.
        for mob in sorted(df_p["MOB"].unique()):
            mob_int = int(mob)
            if mob_int > H:
                if tail_decay_n is None or last_k is None:
                    k_by_mob[mob_int] = 1.0
                else:
                    t = min((mob_int - H) / float(tail_decay_n), 1.0)
                    k_tail = last_k + (1.0 - last_k) * t
                    k_by_mob[mob_int] = float(np.clip(k_tail, clip_min, clip_max))
                continue

            df_m = df_p[df_p["MOB"] == mob_int]
            if len(df_m) < min_obs:
                k_by_mob[mob_int] = 1.0
                continue

            ratios = df_m["METRIC_ACT"] / df_m["METRIC_FC"].replace(0, np.nan)
            ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

            if ratios.empty:
                k_by_mob[mob_int] = 1.0
                continue

            if method == "median":
                k = float(np.median(ratios.values))
            else:
                k = trimmed_mean(ratios.values, trim=0.2)

            k = float(np.clip(k, clip_min, clip_max))
            k_by_mob[mob_int] = k
            last_k = k

        results[product] = k_by_mob

    return results




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

def compute_scale(k, mob, m_apply, blend_n=2):
    """
    k       : hệ số calibration của product
    mob     : MOB hiện tại
    m_apply : MOB bắt đầu áp dụng k
    blend_n : số kỳ blend từ m_apply (mặc định = 2)
    
    Logic:
        mob < m_apply      → scale = 1
        mob = m_apply      → scale = blend 1
        mob = m_apply+1    → scale = blend 2
        mob >= m_apply+2   → scale = k
    """
    if mob < m_apply:
        return 1.0

    # blend 2 kỳ đầu: 
    # mob = m_apply     → weight = 0.5
    # mob = m_apply + 1 → weight = 0.75
    # sau đó = k
    for i in range(blend_n):
        if mob == m_apply + i:
            # i=0 → weight=0.5 ; i=1 → weight=0.75
            w = 0.5 + 0.25 * i
            return (1 - w) + w * k   # blend 1 và k
    
    return k  # từ kỳ thứ 3 trở đi
def apply_k_to_lifecycle(df_lifecycle, k_dict, 
                         m_apply_map=M_APPLY_MAP, 
                         default_m_apply=DEFAULT_M_APPLY,
                         blend_n=2):
    df = df_lifecycle.copy()

    risk_cols = ["DPD30+", "DPD60+", "DPD90+", 
                 "DPD120+", "DPD180+", "WRITEOFF"]

    for product, k in k_dict.items():
        m_apply = m_apply_map.get(product, default_m_apply)

        # Lấy subset để xử lý tiết kiệm tài nguyên
        idx = df.index[df["PRODUCT_TYPE"] == product]

        for i in idx:
            mob = df.at[i, "MOB"]
            is_fc = df.at[i, "IS_FORECAST"]

            if not is_fc:
                continue

            scale = compute_scale(k, mob, m_apply, blend_n)

            if scale != 1:
                df.loc[i, risk_cols] = df.loc[i, risk_cols] * scale

    return df
def apply_k_to_sale_plan(df_plan_fc, k_dict, 
                         m_apply_map=M_APPLY_MAP, 
                         default_m_apply=DEFAULT_M_APPLY,
                         blend_n=2):
    df = df_plan_fc.copy()

    risk_cols = ["DPD30+", "DPD60+", "DPD90+", 
                 "DPD120+", "DPD180+", "WRITEOFF"]

    for product, k in k_dict.items():
        m_apply = m_apply_map.get(product, default_m_apply)

        idx = df.index[df["PRODUCT_TYPE"] == product]

        for i in idx:
            mob = df.at[i, "MOB"]
            scale = compute_scale(k, mob, m_apply, blend_n)

            if scale != 1:
                df.loc[i, risk_cols] = df.loc[i, risk_cols] * scale

    return df


def apply_k_to_lifecycle_by_mob(
    df_lifecycle: pd.DataFrame,
    k_by_product_mob: Dict[str, Dict[int, float]],
    m_apply_map: Dict[str, int] = M_APPLY_MAP,
    default_m_apply: int = DEFAULT_M_APPLY,
):
    df = df_lifecycle.copy()

    risk_cols = ["DPD30+", "DPD60+", "DPD90+",
                 "DPD120+", "DPD180+", "WRITEOFF"]

    for product, k_by_mob in k_by_product_mob.items():
        m_apply = m_apply_map.get(product, default_m_apply)

        idx = df.index[(df["PRODUCT_TYPE"] == product) & (df["IS_FORECAST"] == 1)]

        for i in idx:
            mob = int(df.at[i, "MOB"])
            if mob < m_apply:
                continue

            k = k_by_mob.get(mob, 1.0)
            if not np.isfinite(k):
                k = 1.0

            if k != 1:
                df.loc[i, risk_cols] = df.loc[i, risk_cols] * k

    return df


def apply_k_to_sale_plan_by_mob(
    df_plan_fc: pd.DataFrame,
    k_by_product_mob: Dict[str, Dict[int, float]],
    m_apply_map: Dict[str, int] = M_APPLY_MAP,
    default_m_apply: int = DEFAULT_M_APPLY,
):
    df = df_plan_fc.copy()

    risk_cols = ["DPD30+", "DPD60+", "DPD90+",
                 "DPD120+", "DPD180+", "WRITEOFF"]

    for product, k_by_mob in k_by_product_mob.items():
        m_apply = m_apply_map.get(product, default_m_apply)

        idx = df.index[df["PRODUCT_TYPE"] == product]

        for i in idx:
            mob = int(df.at[i, "MOB"])
            if mob < m_apply:
                continue

            k = k_by_mob.get(mob, 1.0)
            if not np.isfinite(k):
                k = 1.0

            if k != 1:
                df.loc[i, risk_cols] = df.loc[i, risk_cols] * k

    return df
