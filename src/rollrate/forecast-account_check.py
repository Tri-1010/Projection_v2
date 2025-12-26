# ============================================================
#  forecast.py – Full Forecast Engine (BUCKETS_CANON VERSION)
# ============================================================

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Tuple

import matplotlib.pyplot as plt

# QUAN TRỌNG: dùng BUCKETS_CANON thay vì STATE_SPACE
from src.config import CFG, BUCKETS_CANON


# ============================================================
# 0️⃣ Macro Adjustment (optional)
# ============================================================

def apply_macro_adjustment(P, macro_params=None, enable_macro=False):
    if not enable_macro:
        return P
    # TODO: implement macro adjustments later
    return P


# ============================================================
# 1️⃣ FORECAST SEGMENT (MOB-level Markov Simulation)
# ============================================================

def forecast_segment(
    matrices_by_mob: Dict,
    product: str,
    score: str,
    start_mob: int,
    initial_dist: pd.Series,
    max_mob: int,
    enable_macro=False,
    macro_params=None,
):
    """
    Forecast từ start_mob → max_mob sử dụng transition matrices.
    Return: {mob: distribution}
    """

    result = {}
    cur_dist = initial_dist.copy()
    result[start_mob] = cur_dist.copy()

    for mob in range(start_mob, max_mob):

        # Lấy matrix đúng MOB
        if (
            product in matrices_by_mob and
            mob in matrices_by_mob[product] and
            score in matrices_by_mob[product][mob]
        ):
            P = matrices_by_mob[product][mob][score]["P"]
        else:
            # fallback: MOB cuối cùng
            last_mob = max(matrices_by_mob[product].keys())
            P = matrices_by_mob[product][last_mob][score]["P"]

        P_adj = apply_macro_adjustment(P, macro_params, enable_macro)

        # vector × matrix
        cur_dist = cur_dist @ P_adj.values

        # ALWAYS map back to BUCKETS_CANON
        result[mob + 1] = pd.Series(cur_dist, index=BUCKETS_CANON)

    return result


# ============================================================
# 2️⃣ FORECAST CHO 1 VINTAGE (DISBURSAL_DATE)
# ============================================================

def forecast_vintage(
    df_raw: pd.DataFrame,
    matrices_by_mob: Dict,
    product: str,
    score: str,
    vintage_date,
    max_mob=29,
    enable_macro=False,
    macro_params=None,
):
    """
    Forecast cho 1 vintage (cohort):
      - Vintage xác định bằng DISBURSAL_DATE
      - Tự lấy MOB hiện tại từ cutoff mới nhất
    """

    orig_col   = CFG["orig_date"]   # DISBURSAL_DATE
    cutoff_col = CFG["cutoff"]
    state_col  = CFG["state"]
    mob_col    = CFG["mob"]

    df_vintage = df_raw[
        (df_raw["PRODUCT_TYPE"] == product) &
        (df_raw["RISK_SCORE"] == score) &
        (df_raw[orig_col] == vintage_date)
    ]

    if df_vintage.empty:
        raise ValueError(f"No loans found for vintage {vintage_date}")

    # Snapshot gần nhất
    latest_cutoff = df_vintage[cutoff_col].max()
    df_latest = df_vintage[df_vintage[cutoff_col] == latest_cutoff]

    # MOB hiện tại của vintage
    start_mob = int(df_latest[mob_col].max())

    # Initial distribution
    init_dist = (
        df_latest[state_col]
        .value_counts(normalize=True)
        .reindex(BUCKETS_CANON, fill_value=0.0)
    )

    return forecast_segment(
        matrices_by_mob,
        product,
        score,
        start_mob,
        init_dist,
        max_mob,
        enable_macro,
        macro_params,
    )


# ============================================================
# 3️⃣ FORECAST FULL PORTFOLIO
# ============================================================

def forecast_all_vintages(
    df_raw: pd.DataFrame,
    matrices_by_mob: Dict,
    max_mob=29,
    enable_macro=False,
    macro_params=None,
):
    """
    Tự động forecast cho tất cả:
        - mọi Product trong matrices_by_mob
        - mọi Score trong matrices_by_mob
        - mọi Vintage thực tế (DISBURSAL_DATE) trong df_raw
    """

    orig_col = CFG["orig_date"]
    results = {}

    for product in matrices_by_mob.keys():

        # Lấy toàn bộ score
        sample_mob = next(iter(matrices_by_mob[product].keys()))
        scores = list(matrices_by_mob[product][sample_mob].keys())

        for score in scores:

            # Lấy toàn bộ vintages có trong data
            vintages = (
                df_raw[df_raw["PRODUCT_TYPE"] == product][orig_col]
                .dropna()
                .unique()
            )

            for v in vintages:
                try:
                    fc = forecast_vintage(
                        df_raw=df_raw,
                        matrices_by_mob=matrices_by_mob,
                        product=product,
                        score=score,
                        vintage_date=v,
                        max_mob=max_mob,
                        enable_macro=enable_macro,
                        macro_params=macro_params,
                    )
                    results[(product, score, v)] = fc

                except Exception as e:
                    print(f"⚠️ Skip ({product}, {score}, v={v}): {e}")

    return results


# ============================================================
# 4️⃣ Helper: Convert forecast → DataFrame
# ============================================================

def forecast_to_dataframe(forecast_dict):
    df = pd.DataFrame(forecast_dict).T
    df.index.name = "MOB"
    return df


# ============================================================
# 5️⃣ PD Helpers
# ============================================================

def marginal_pd(forecast_dict, default_state="WRITEOFF"):
    return pd.Series({mob: dist[default_state] for mob, dist in forecast_dict.items()})

def cumulative_pd(forecast_dict, default_state="WRITEOFF"):
    mpd = marginal_pd(forecast_dict, default_state)
    return mpd.cumsum().clip(upper=1.0)


# ============================================================
# 6️⃣ Vintage Table
# ============================================================

def vintage_table(forecast_dict):
    df = forecast_to_dataframe(forecast_dict)
    dpd_cols = [c for c in df.columns if c in ["DPD30+", "DPD60+", "DPD90+", "DPD120+", "DPD180+"]]
    if dpd_cols:
        df["DPD30+_AGG"] = df[dpd_cols].sum(axis=1)
    return df


# ============================================================
# 7️⃣ Plot Helper
# ============================================================

def plot_curve(forecast_df, columns, title=None):
    plt.figure(figsize=(10, 5))
    for col in columns:
        plt.plot(forecast_df.index, forecast_df[col], label=col)

    plt.xlabel("MOB")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.title(title or "Forecast Curve")
    plt.show()


# ============================================================
# 8️⃣ Validate Matrices
# ============================================================

def validate_matrices(matrices_by_mob, atol=1e-6):
    issues = []
    for product, mob_dict in matrices_by_mob.items():
        for mob, score_dict in mob_dict.items():
            for score, entry in score_dict.items():

                P = entry["P"]

                # Check state order
                if list(P.index) != BUCKETS_CANON or list(P.columns) != BUCKETS_CANON:
                    issues.append((product, mob, score, "State mismatch"))
                    continue

                # Check normalization
                row_sums = P.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=atol):
                    issues.append((product, mob, score,
                                   f"Row sum error (min={row_sums.min()}, max={row_sums.max()})"))

    return issues
# def get_actual_all_vintages(df_raw):
#     """
#     Lấy dữ liệu thực tế cho TẤT CẢ vintage trong df_raw:
#         product × score × vintage_date → {mob: Series}

#     Output:
#         actual_results[(product, score, vintage_date)] = {mob: Series}
#     """

#     orig_col  = CFG["orig_date"]
#     state_col = CFG["state"]
#     mob_col   = CFG["mob"]

#     results = {}

#     # Lấy toàn bộ nhóm
#     for (product, score, vintage_date), df_vintage in df_raw.groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col]):

#         # Bỏ các group rác
#         if df_vintage.empty:
#             continue

#         actual_dict = {}

#         # Group theo MOB để lấy state distribution thực tế
#         for mob, df_m in df_vintage.groupby(mob_col):

#             dist = (
#                 df_m[state_col]
#                 .value_counts(normalize=True)
#                 .reindex(BUCKETS_CANON, fill_value=0.0)
#             )

#             actual_dict[mob] = dist

#         results[(product, score, vintage_date)] = actual_dict

#     return results

# def combine_all_lifecycle(actual_results, forecast_results):
#     """
#     Ghép actual + forecast thành lifecycle đầy đủ
#     cho mọi product × score × vintage.

#     Output:
#         lifecycle[(product, score, vintage)] = {mob: Series}
#     """

#     lifecycle = {}

#     for key in forecast_results.keys():

#         forecast_dict = forecast_results[key]
#         actual_dict   = actual_results.get(key, {})

#         merged = {}

#         # Actual trước
#         for mob in sorted(actual_dict.keys()):
#             merged[mob] = actual_dict[mob]

#         # Forecast sau (overwrite nếu MOB trùng)
#         for mob in sorted(forecast_dict.keys()):
#             merged[mob] = forecast_dict[mob]

#         lifecycle[key] = merged

#     return lifecycle
# def lifecycle_to_long_df(lifecycle):
#     """
#     Convert lifecycle dict → long format:
    
#     Columns:
#         PRODUCT_TYPE
#         RISK_SCORE
#         VINTAGE_DATE
#         MOB
#         DPD0 ... SOLDOUT
#     """

#     rows = []

#     for (product, score, vintage_date), mob_dict in lifecycle.items():
#         for mob, dist in mob_dict.items():

#             row = {
#                 "PRODUCT_TYPE": product,
#                 "RISK_SCORE": score,
#                 "VINTAGE_DATE": vintage_date,
#                 "MOB": mob
#             }

#             row.update(dist.to_dict())
#             rows.append(row)

#     return pd.DataFrame(rows)
# def build_full_lifecycle(df_raw, matrices_by_mob, max_mob=29):
#     """
#     Pipeline đầy đủ:
#         1. Forecast toàn bộ vintage
#         2. Lấy actual toàn bộ vintage
#         3. Merge actual + forecast → lifecycle
#         4. Convert lifecycle → long format
    
#     Output:
#         df_lifecycle_long
#     """

#     # 1. FORECAST
#     forecast_results = forecast_all_vintages(
#         df_raw=df_raw,
#         matrices_by_mob=matrices_by_mob,
#         max_mob=max_mob
#     )

#     # 2. ACTUAL
#     actual_results = get_actual_all_vintages(df_raw)

#     # 3. MERGE
#     lifecycle = combine_all_lifecycle(actual_results, forecast_results)

#     # 4. LONG FORMAT
#     df_long = lifecycle_to_long_df(lifecycle)

#     return df_long


