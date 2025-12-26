"""
pipeline_lifecycle.py
----------------------
Full pipeline từ raw data → transition → forecast →
calibration → seasonality → lifecycle → export report.
"""

import pandas as pd
import numpy as np

# --- Transition & Forecast Engines ---
from src.rollrate.transition import compute_transition_by_mob
from src.rollrate.forecast import forecast_all_vintages
from src.rollrate.forecast_plan import forecast_sale_plan_by_mob

# --- Lifecycle ---
from src.rollrate.lifecycle import (
    get_actual_all_vintages_amount,
    combine_all_lifecycle_amount,
    lifecycle_to_long_df_amount,
    tag_forecast_rows_amount,
    add_del_metrics,
    aggregate_to_product,
    aggregate_products_to_portfolio,
    export_lifecycle_all_products_one_file,
    extend_actual_info_with_portfolio,
)

# --- Calibration ---
from src.rollrate.calibration import (
    compute_k_per_product_anchor,
    apply_k_to_lifecycle,
    apply_k_to_sale_plan
)

# --- Seasonality ---
from src.rollrate.seasonality import (
    build_seasonality,
    apply_seasonality_to_lifecycle,
    apply_seasonality_to_sale_plan
)

from src.config import CFG, BUCKETS_CANON


# ============================================================
#  MAIN PIPELINE
# ============================================================

def run_full_pipeline(
    df_raw: pd.DataFrame,
    sale_plan_df: pd.DataFrame,
    max_mob: int = 29,
    sale_plan_mob: int = 24,
    export_filename: str = "Lifecycle_Report.xlsx",
    portfolio_name: str = "PORTFOLIO_ALL",
):

    print("\n======================")
    print("1) COMPUTE TRANSITION")
    print("======================")
    matrices_by_mob, parent_fallback = compute_transition_by_mob(df_raw)


    print("\n======================")
    print("2) FORECAST ACTUAL LOANS")
    print("======================")
    forecast_results = forecast_all_vintages(
        df_raw=df_raw,
        matrices_by_mob=matrices_by_mob,
        max_mob=max_mob,
        enable_macro=False,
    )


    print("\n======================")
    print("3) FORECAST SALE PLAN")
    print("======================")
    sale_plan_fc = forecast_sale_plan_by_mob(
        sale_plan_df=sale_plan_df,
        matrices_by_mob=matrices_by_mob,
        parent_fallback=parent_fallback,
        mob_target=sale_plan_mob,
        start_state=BUCKETS_CANON[0],
        states=list(dict.fromkeys(list(BUCKETS_CANON) + ["PREPAY", "WRITEOFF"]))
    )


    print("\n======================")
    print("4) BUILD LIFECYCLE ACTUAL + FORECAST")
    print("======================")
    actual_results = get_actual_all_vintages_amount(df_raw)
    lifecycle_dict = combine_all_lifecycle_amount(actual_results, forecast_results)
    df_lifecycle = lifecycle_to_long_df_amount(lifecycle_dict)


    print("\n======================")
    print("5) ADD DEL30/60/90 METRICS")
    print("======================")
    df_lifecycle = add_del_metrics(df_lifecycle, df_raw)

    # Sale plan forecast also needs DEL metrics
    sale_plan_fc = add_del_metrics(sale_plan_fc, sale_plan_df)


    print("\n======================")
    print("6) CALIBRATION (k per product)")
    print("======================")
    # Dùng lifecycle gốc cho cập nhật k
    k_dict = compute_k_per_product_anchor(
        df_lifecycle=df_lifecycle,
        forecast_long_df=df_lifecycle,
    )
    print("Calibration K:", k_dict)

    df_lifecycle = apply_k_to_lifecycle(df_lifecycle, k_dict)
    sale_plan_fc = apply_k_to_sale_plan(sale_plan_fc, k_dict)


    print("\n======================")
    print("7) SEASONALITY")
    print("======================")
    seasonality = build_seasonality(df_lifecycle)
    df_lifecycle = apply_seasonality_to_lifecycle(df_lifecycle, seasonality)
    sale_plan_fc = apply_seasonality_to_sale_plan(sale_plan_fc, seasonality)


    print("\n======================")
    print("8) AGG PRODUCT → PORTFOLIO")
    print("======================")
    # Gộp DEL metrics theo product
    df_del_prod = aggregate_to_product(df_lifecycle)

    # Gộp product → portfolio
    df_port = aggregate_products_to_portfolio(df_del_prod, portfolio_name=portfolio_name)

    df_final = pd.concat([df_del_prod, df_port], ignore_index=True)


    print("\n======================")
    print("9) EXPORT REPORT")
    print("======================")

    # Actual max MOB để highlight actual vs forecast
    actual_info = (
        df_raw.groupby(["PRODUCT_TYPE", CFG["orig_date"]])[CFG["mob"]]
        .max()
        .rename("max_mob")
    )
    actual_info = { (p, d): m for (p, d), m in actual_info.items() }

    # extend thêm portfolio-level actual_info
    actual_info = extend_actual_info_with_portfolio(actual_info, portfolio_name)

    export_lifecycle_all_products_one_file(
        df_del_prod=df_final,
        actual_info=actual_info,
        filename=export_filename
    )

    print("\n======================")
    print("DONE!")
    print("======================")

    return {
        "df_lifecycle": df_lifecycle,
        "df_plan_fc": sale_plan_fc,
        "df_final": df_final,
        "k_dict": k_dict,
        "seasonality": seasonality
    }
