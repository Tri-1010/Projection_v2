"""
Reporting utilities for lifecycle analysis:
- Merge Actual + Forecast (sale plan)
- Export Lifecycle Excel (multi-product, multi-metric)
"""

import pandas as pd
import numpy as np


# ============================================================
# 1️⃣ BUILD LIFECYCLE REPORT DATA (ACTUAL + FORECAST)
# ============================================================

def build_lifecycle_for_report(df_actual, df_plan_fc, buckets):
    """
    Kết hợp Actual lifecycle + Forecast sale plan lifecycle.
    Chuẩn hóa format để export Excel.

    Output columns:
        PRODUCT_TYPE | RISK_SCORE | VINTAGE_DATE | MOB | bucket%... | is_forecast
    """
    dfA = df_actual.copy()
    dfF = df_plan_fc.copy()

    # Thêm cột flag
    dfA["is_forecast"] = 0
    dfF["is_forecast"] = 1

    # Align columns
    keep_cols = ["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"]
    bucket_cols = buckets  # BUCKETS_CANON

    dfA = dfA[keep_cols + bucket_cols + ["is_forecast"]]
    dfF = dfF[keep_cols + bucket_cols + ["is_forecast"]]

    df_all = pd.concat([dfA, dfF], ignore_index=True)

    # Sort đẹp
    df_all = df_all.sort_values(
        ["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"]
    ).reset_index(drop=True)

    return df_all



# ============================================================
# 2️⃣ EXPORT LIFECYCLE (ACTUAL + FORECAST)
# ============================================================

def export_lifecycle_all_products_one_file_extended(
    df_lifecycle,
    actual_info,
    filename="Lifecycle_Actual_Forecast.xlsx"
):
    """
    Xuất Actual + Forecast lifecycle theo Product × Metric.
    Bản mở rộng từ export_lifecycle_all_products_one_file cũ.
    """

    import xlsxwriter

    metric_map = {
        "DEL30": "DPD30+",
        "DEL60": "DPD60+",
        "DEL90": "DPD90+",
    }

    products = df_lifecycle["PRODUCT_TYPE"].unique()

    with pd.ExcelWriter(filename, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        workbook = writer.book

        # Format
        fmt_header = workbook.add_format({
            "bold": True,
            "bg_color": "#D9D9D9",
            "border": 2,
            "align": "center",
        })
        fmt_data = workbook.add_format({
            "border": 1,
            "num_format": "0.00%",
        })
        fmt_forecast = workbook.add_format({
            "bg_color": "#FFF3B0",
            "border": 1,
            "num_format": "0.00%",
        })
        fmt_boundary = workbook.add_format({
            "border": 1,
            "bottom": 5,
            "bottom_color": "red",
            "right": 3,
            "right_color": "gray",
            "num_format": "0.00%",
        })
        fmt_cohort = workbook.add_format({
            "border": 1,
            "num_format": "yyyy-mm-dd",
        })

        # ============================================================
        # LOOP TỪNG PRODUCT
        # ============================================================
        for product in products:

            df_prod = df_lifecycle[df_lifecycle["PRODUCT_TYPE"] == product]

            # Loop metrics
            for metric_name, bucket_col in metric_map.items():

                sheet_name = f"{product}_{metric_name}"[:31]

                pivot = df_prod.pivot_table(
                    index="VINTAGE_DATE",
                    columns="MOB",
                    values=bucket_col
                ).fillna(0)

                pivot.index.name = "Cohort"
                n_rows, n_cols = pivot.shape

                # write raw data
                pivot.to_excel(writer, sheet_name=sheet_name,
                               startrow=1, startcol=1,
                               index=False, header=False)

                ws = writer.sheets[sheet_name]
                ws.hide_gridlines(2)

                # header
                ws.write(0, 0, "Cohort", fmt_header)
                for col_idx, mob in enumerate(pivot.columns, start=1):
                    ws.write(0, col_idx, mob, fmt_header)

                # ============================================================
                # CELL FORMATTING (ACTUAL vs FORECAST)
                # ============================================================
                for i, cohort in enumerate(pivot.index):
                    r = 1 + i
                    ws.write(r, 0, cohort, fmt_cohort)

                    max_actual = actual_info.get((product, cohort), None)

                    for j, mob in enumerate(pivot.columns):
                        c = 1 + j
                        val = round(float(pivot.iat[i, j]), 4)

                        # detect actual/forecast row
                        df_row = df_prod[
                            (df_prod["VINTAGE_DATE"] == cohort) &
                            (df_prod["MOB"] == mob)
                        ]

                        if df_row.empty:
                            fmt = fmt_data
                        else:
                            is_fc = int(df_row["is_forecast"].iloc[0])
                            fmt = fmt_forecast if is_fc == 1 else fmt_data

                        # boundary red nếu đây là MOB actual cuối cùng
                        if (max_actual is not None) and (mob == max_actual):
                            fmt = fmt_boundary

                        ws.write(r, c, val, fmt)

                # column width
                ws.set_column(0, 0, 12)
                for j, mob in enumerate(pivot.columns):
                    ws.set_column(1 + j, 1 + j, max(8, len(str(mob))) + 2)

    print(f"✔ Export Actual + Plan lifecycle → {filename}")
