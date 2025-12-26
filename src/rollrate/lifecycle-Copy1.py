import pandas as pd
from typing import Dict

from src.config import CFG, BUCKETS_CANON

# Forecast engine đã amount-based
from src.rollrate.forecast import forecast_all_vintages


# ============================================================
# 0️⃣ Bucket groups
# ============================================================

BUCKETS_30P = ["DPD30+", "DPD60+", "DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]
BUCKETS_60P = ["DPD60+", "DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]
BUCKETS_90P = ["DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]


# ============================================================
# 1️⃣ ACTUAL LIFECYCLE — AMOUNT-BASED (EAD)
# ============================================================

def get_actual_all_vintages_amount(df_raw: pd.DataFrame):
    """
    Trả về:
        actual_results[(product, score, vintage)] = {mob: Series(EAD theo state)}
    """
    state_col = CFG["state"]
    mob_col   = CFG["mob"]
    orig_col  = CFG["orig_date"]
    ead_col   = CFG["ead"]

    results = {}

    for (product, score, vintage_date), df_vintage in \
        df_raw.groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col]):

        mob_dict = {}

        for mob, df_m in df_vintage.groupby(mob_col):

            ead_vec = (
                df_m.groupby(state_col)[ead_col].sum()
                .reindex(BUCKETS_CANON, fill_value=0.0)
            )

            mob_dict[mob] = ead_vec

        results[(product, score, vintage_date)] = mob_dict

    return results


# ============================================================
# 2️⃣ MERGE ACTUAL + FORECAST (EAD-based)
# ============================================================

def combine_all_lifecycle_amount(actual, forecast):
    lifecycle = {}

    for key in forecast.keys():

        ac = actual.get(key, {})
        fc = forecast[key]

        merged = {}

        # Actual trước
        for mob in sorted(ac.keys()):
            merged[mob] = ac[mob]

        # Forecast sau
        for mob in sorted(fc.keys()):
            merged[mob] = fc[mob]

        lifecycle[key] = merged

    return lifecycle


# ============================================================
# 3️⃣ Convert lifecycle → Long Format (EAD columns)
# ============================================================

def lifecycle_to_long_df_amount(lifecycle: Dict):
    rows = []

    for (product, score, vintage_date), mob_dict in lifecycle.items():
        for mob, ead_vec in mob_dict.items():

            row = {
                "PRODUCT_TYPE": product,
                "RISK_SCORE": score,
                "VINTAGE_DATE": vintage_date,
                "MOB": mob
            }

            row.update(ead_vec.to_dict())
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 4️⃣ FULL PIPELINE
# ============================================================

def build_full_lifecycle_amount(df_raw, matrices_by_mob, max_mob=29):

    # Forecast
    forecast_results = forecast_all_vintages(
        df_raw=df_raw,
        matrices_by_mob=matrices_by_mob,
        max_mob=max_mob,
        enable_macro=False
    )

    # Actual
    actual_results = get_actual_all_vintages_amount(df_raw)

    # Merge
    lifecycle = combine_all_lifecycle_amount(actual_results, forecast_results)

    # Long format
    df_long = lifecycle_to_long_df_amount(lifecycle)

    return df_long


# ============================================================
# 5️⃣ Tag forecast rows
# ============================================================

def tag_forecast_rows_amount(df_lifecycle, df_raw):
    mob_col = CFG["mob"]
    orig = CFG["orig_date"]

    actual_max = (
        df_raw.groupby(["PRODUCT_TYPE", "RISK_SCORE", orig])[mob_col]
        .max()
        .rename("ACTUAL_MAX_MOB")
        .reset_index()
        .rename(columns={orig: "VINTAGE_DATE"})
    )

    df = df_lifecycle.merge(actual_max,
                            on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"],
                            how="left")

    df["IS_FORECAST"] = (df["MOB"] > df["ACTUAL_MAX_MOB"]).astype(int)

    return df


# ============================================================
# 6️⃣ DEL30+, DEL60+, DEL90+ amount-based
# ============================================================

def add_del_metrics(df_lifecycle, df_raw):
    """
    Tính DEL30/60/90 amount + pct trên DISB_TOTAL.

    DISB_TOTAL được tính đúng theo:
        - mỗi loan chỉ đóng góp đúng 1 lần DISBURSAL_AMOUNT
        - sau đó sum lên theo Product × Score × Vintage
    """

    disp_col  = CFG["disb"]
    orig_col  = CFG["orig_date"]
    loan_col  = CFG["loan"]

    # 1️⃣ DISB per loan (mỗi khoản vay 1 lần)
    loan_disb = (
        df_raw
        .groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col, loan_col])[disp_col]
        .first()  # hoặc .max(), vì disb không đổi
        .reset_index()
    )

    # 2️⃣ DISB_TOTAL per cohort nhỏ (Product × Score × Vintage)
    cohort_disb = (
        loan_disb
        .groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col])[disp_col]
        .sum()
        .rename("DISB_TOTAL")
        .reset_index()
        .rename(columns={orig_col: "VINTAGE_DATE"})
    )

    # 3️⃣ Merge vào lifecycle
    df = df_lifecycle.merge(
        cohort_disb,
        on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"],
        how="left"
    )

    # 4️⃣ Tính DEL30/60/90 amount
    df["DEL30_AMT"] = df[BUCKETS_30P].sum(axis=1)
    df["DEL60_AMT"] = df[BUCKETS_60P].sum(axis=1)
    df["DEL90_AMT"] = df[BUCKETS_90P].sum(axis=1)

    # 5️⃣ Tính % trên DISB_TOTAL (chuẩn)
    df["DEL30_PCT"] = df["DEL30_AMT"] / df["DISB_TOTAL"]
    df["DEL60_PCT"] = df["DEL60_AMT"] / df["DISB_TOTAL"]
    df["DEL90_PCT"] = df["DEL90_AMT"] / df["DISB_TOTAL"]

    return df



# ============================================================
# 7️⃣ Aggregate Product_Score → Product
# ============================================================
def aggregate_loss_to_product(
    forecast_results: dict,
    df_raw: pd.DataFrame,
    product_filter: str | None = None,
    default_state: str = "WRITEOFF"
):
    """
    Tính LOSS RATE (WRITEOFF) theo:
        PRODUCT_TYPE × VINTAGE_DATE × MOB  (amount-based)

    forecast_results: output từ forecast_all_vintages
        {(product, score, vintage): {mob: Series(EAD)}}

    df_raw: để tính DISB weighting:
        weight = DISB_SCORE / DISB_PRODUCT

    product_filter: 
        - None → chạy tất cả product
        - "SALPIL" → chỉ tính product SALPIL
    """

    disb_col = CFG["disb"]
    orig_col = CFG["orig_date"]

    # 1️⃣ DISB theo PRODUCT × SCORE × VINTAGE
    disb_psv = (
        df_raw.groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col])[disb_col]
        .sum()
        .rename("DISB_SCORE")
        .reset_index()
    )

    # 2️⃣ DISB theo PRODUCT × VINTAGE
    disb_pv = (
        df_raw.groupby(["PRODUCT_TYPE", orig_col])[disb_col]
        .sum()
        .rename("DISB_PRODUCT")
        .reset_index()
    )

    # Merge weight
    weight_df = disb_psv.merge(
        disb_pv,
        on=["PRODUCT_TYPE", orig_col],
        how="left"
    )
    weight_df["WEIGHT"] = weight_df["DISB_SCORE"] / weight_df["DISB_PRODUCT"]

    # 3️⃣ Aggregate lên PRODUCT LEVEL
    product_loss = {}

    for (product, score, vintage), fc_seg in forecast_results.items():

        if product_filter is not None:
            if product != product_filter:
                continue

        # Lấy weight score-level
        w = weight_df[
            (weight_df["PRODUCT_TYPE"] == product) &
            (weight_df["RISK_SCORE"] == score) &
            (weight_df[orig_col] == vintage)
        ]

        if w.empty:
            continue

        w = w["WEIGHT"].iloc[0]

        # Tính LOSS RATE score-level
        first_mob = min(fc_seg.keys())
        original_ead = fc_seg[first_mob].sum()

        loss_series = pd.Series({
            mob: fc_seg[mob][default_state] / original_ead
            for mob in sorted(fc_seg.keys())
        })

        # Cộng dồn vào product-level
        key = (product, vintage)
        if key not in product_loss:
            product_loss[key] = loss_series * w
        else:
            product_loss[key] += loss_series * w

    return product_loss

def aggregate_to_product(df_del):
    """
    Gộp DEL30+, DEL60+, DEL90+ từ level Product × Score × Vintage × MOB
    lên level Product × Vintage × MOB bằng cách:

        DEL30_PCT_product
            = sum( DEL30_AMT_score ) / sum( DISB_TOTAL_score )

        = sum( (DEL30_PCT_score × DISB_TOTAL_score) ) / sum(DISB_TOTAL_score)

    Đây là Cách 2 (chuẩn nhất):
        - Bước 1: xác định DISB_TOTAL duy nhất cho từng cohort nhỏ 
                  (PRODUCT_TYPE × RISK_SCORE × VINTAGE_DATE)
        - Bước 2: tính tổng DISB_TOTAL theo cấp Product × Vintage
        - Bước 3: tính WEIGHT = DISB_TOTAL_score / PRODUCT_DISB
        - Bước 4: Weighted-average các pct theo weight
    """

    df = df_del.copy()

    # ===================================================
    # 1️⃣ Lấy DISB_TOTAL duy nhất per Product × Score × Vintage
    #    (KHÔNG bị trùng theo MOB)
    # ===================================================
    cohort_disb = (
        df.groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"])["DISB_TOTAL"]
        .first()   # hoặc .max(), vì DISB_TOTAL không đổi theo MOB
        .reset_index()
    )


    # ===================================================
    # 2️⃣ Lấy tổng DISB theo Product × Vintage
    # ===================================================
    product_disb = (
        cohort_disb.groupby(["PRODUCT_TYPE", "VINTAGE_DATE"])["DISB_TOTAL"]
        .sum()
        .rename("PRODUCT_DISB")
        .reset_index()
    )


    # ===================================================
    # 3️⃣ Merge DISB_TOTAL và PRODUCT_DISB vào từng dòng lifecycle
    # ===================================================
    df = df.merge(
        cohort_disb,
        on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"],
        how="left",
        suffixes=("", "_COHORT")
    )

    df = df.merge(
        product_disb,
        on=["PRODUCT_TYPE", "VINTAGE_DATE"],
        how="left"
    )


    # ===================================================
    # 4️⃣ Weight theo DISB_TOTAL_score / PRODUCT_DISB
    # ===================================================
    df["WEIGHT"] = df["DISB_TOTAL_COHORT"] / df["PRODUCT_DISB"]


    # ===================================================
    # 5️⃣ Aggregate theo Product × Vintage × MOB
    # ===================================================
    agg = (
        df.groupby(["PRODUCT_TYPE", "VINTAGE_DATE", "MOB"])
        .apply(lambda g: pd.Series({
            "DEL30_PCT": (g["DEL30_PCT"] * g["WEIGHT"]).sum(),
            "DEL60_PCT": (g["DEL60_PCT"] * g["WEIGHT"]).sum(),
            "DEL90_PCT": (g["DEL90_PCT"] * g["WEIGHT"]).sum(),
            "PRODUCT_DISB": g["PRODUCT_DISB"].iloc[0]   # giữ lại để dễ check
        }))
        .reset_index()
    )

    return agg


# ============================================================
# 8️⃣ Pivot tables
# ============================================================

def make_metric_pivot(df, metric):
    return df.pivot_table(
        index="VINTAGE_DATE",
        columns="MOB",
        values=metric
    ).fillna(0)

# def export_heatmap(df_pivot, df_del_prod, actual_info, filename="DEL30P_LIFECYCLE.xlsx"):
#     """
#     Xuất pivot lifecycle DEL30+ theo dạng:
#         Row    = Cohort (VintageDate)
#         Column = MOB
#         Value  = % DEL30+ trên DISB
    
#     THÊM:
#         ✔ Format % (value * 100)
#         ✔ Format cohort = YYYY-MM-DD
#         ✔ Heatmap xanh→vàng→đỏ
#         ✔ Highlight forecast (vàng nhạt)
#         ✔ Đóng khung toàn bảng
#         ✔ Kẻ đường đỏ tại vị trí actual max MOB cho từng cohort
#     """

#     with pd.ExcelWriter(filename, engine="xlsxwriter", datetime_format='yyyy-mm-dd') as writer:

#         df_pivot.to_excel(writer, sheet_name="DEL30P", startrow=1)
#         workbook  = writer.book
#         worksheet = writer.sheets["DEL30P"]

#         # =========================================
#         # 1. Format chuẩn
#         # =========================================
#         fmt_header = workbook.add_format({
#             "bold": True,
#             "bg_color": "#D9D9D9",
#             "border": 2,
#             "align": "center"
#         })

#         fmt_data = workbook.add_format({
#             "border": 1,
#             "num_format": "0.00%"   # %-format
#         })

#         fmt_forecast = workbook.add_format({
#             "bg_color": "#FFF3B0",  # vàng
#             "border": 1,
#             "num_format": "0.00%"
#         })

#         fmt_red_bottom = workbook.add_format({
#             "num_format": "0.00%",
            
#             # Thin border quanh cell (tuỳ bạn có muốn hay không)
#             "border": 1,
        
#             # Bottom border: thick red
#             "bottom": 6,
#             "bottom_color": "red",
        
#             # Right border: thick red
#             "right": 6,
#             "right_color": "red",
#         })


#         # =========================================
#         # 2. Ghi header thủ công
#         # =========================================
#         worksheet.write(0, 0, "Cohort", fmt_header)
#         for col_idx, mob in enumerate(df_pivot.columns, start=1):
#             worksheet.write(0, col_idx, mob, fmt_header)

#         # =========================================
#         # 3. Apply heatmap color scale
#         # =========================================
#         n_rows, n_cols = df_pivot.shape

#         worksheet.conditional_format(
#             1, 1, n_rows, n_cols,
#             {
#                 "type": "3_color_scale",
#                 "min_color": "#63BE7B",
#                 "mid_color": "#FFEB84",
#                 "max_color": "#F8696B",
#             }
#         )

#         # =========================================
#         # 4. Format theo từng cell
#         #    - % format
#         #    - highlight forecast
#         #    - line red cho actual max mob
#         # =========================================

#         for row_idx, cohort in enumerate(df_pivot.index, start=1):

#             cohort_str = cohort.strftime("%Y-%m-%d")
#             worksheet.write(row_idx, 0, cohort_str, fmt_data)

#             # Lấy actual max MOB
#             max_actual = actual_info.get((cohort,), None)

#             # Có forecast cho cohort này chưa?
#             df_row = df_del_prod[df_del_prod["VINTAGE_DATE"] == cohort]

#             forecast_mobs = df_row[df_row["DEL30_IS_FORECAST"] == 1]["MOB"].tolist()

#             for col_idx, mob in enumerate(df_pivot.columns, start=1):
#                 value = df_pivot.loc[cohort, mob]

#                 # Forecast cell
#                 if mob in forecast_mobs:
#                     fmt = fmt_forecast

#                 # Border đỏ tại boundary actual/predict
#                 elif max_actual is not None and mob == max_actual:
#                     fmt = fmt_red_bottom

#                 else:
#                     fmt = fmt_data

#                 worksheet.write(row_idx, col_idx, value, fmt)

#     print(f"✔ Exported DEL30+ lifecycle heatmap → {filename}")

# # ============================
# # FORMAT TỪNG CELL (ROUND + BORDER)
# # ============================

# for i, cohort in enumerate(df_pivot.index):
#     r_idx = 1 + i   # row of Excel data

#     worksheet.write(r_idx, 0, cohort, fmt_cohort)

#     max_actual = actual_info.get((product, cohort), None)

#     for j, mob in enumerate(df_pivot.columns):
#         c_idx = 1 + j

#         raw_value = df_pivot.iat[i, j]
#         value = round(float(raw_value), 4)   # ROUND HERE

#         if (max_actual is not None) and (mob > max_actual):
#             fmt = fmt_forecast
#         elif (max_actual is not None) and (mob == max_actual):
#             fmt = fmt_red_bottom
#         else:
#             fmt = fmt_data

#         worksheet.write(r_idx, c_idx, value, fmt)

# # ============================
# # AUTO COLUMN WIDTH
# # ============================
# col_widths = {}

# # Cohort column
# # col_widths[0] = max(15, max(len(str(i)) for i in df_pivot.index))

# worksheet.set_column(0, 0, 13)

# # MOB columns
# for j, mob in enumerate(df_pivot.columns):
#     col_widths[1 + j] = max(8, len(str(mob)))

# # Apply
# for col_idx, width in col_widths.items():
#     worksheet.set_column(col_idx, col_idx, width + 2)

def export_lifecycle_all_products_one_file(
    df_del_prod,
    actual_info,
    filename="Lifecycle_All_Products.xlsx"
):
    """
    Xuất tất cả Product × (DEL30, DEL60, DEL90) vào nhiều sheet trong 1 file Excel.
    Gồm:
        ✔ Heatmap
        ✔ Format %
        ✔ Forecast highlight vàng
        ✔ Boundary đỏ đậm (bottom thick red + right medium grey)
        ✔ Autosize column
        ✔ Cột A (Cohort) width = 12
        ✔ ROUND 4 decimals
        ✔ Không gridlines
    """

    import numpy as np
    import pandas as pd

    metric_map = {
        "DEL30": "DEL30_PCT",
        "DEL60": "DEL60_PCT",
        "DEL90": "DEL90_PCT",
    }

    products = df_del_prod["PRODUCT_TYPE"].unique()

    with pd.ExcelWriter(filename, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:

        workbook = writer.book

        # ============================
        # FORMAT DEFINITIONS
        # ============================

        fmt_header = workbook.add_format({
            "bold": True,
            "bg_color": "#D9D9D9",
            "border": 2,
            "align": "center",
        })

        fmt_cohort = workbook.add_format({
            "border": 1,
            "num_format": "yyyy-mm-dd",
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

        fmt_red_bottom = workbook.add_format({
            "border": 1,
            "bottom": 5,
            "bottom_color": "red",
            "right": 5,
            "right_color": "red",
            "num_format": "0.00%",
        })

        # ============================
        # LOOP ALL PRODUCTS
        # ============================

        for product in products:

            df_prod = df_del_prod[df_del_prod["PRODUCT_TYPE"] == product]

            # Loop metrics (DEL30 / DEL60 / DEL90)
            for metric_name, colname in metric_map.items():

                sheet_name = f"{product}_{metric_name}"[:31]

                # Pivot table
                df_pivot = df_prod.pivot_table(
                    index="VINTAGE_DATE",
                    columns="MOB",
                    values=colname,
                ).fillna(0.0)
                df_pivot.index.name = "Cohort"

                n_rows, n_cols = df_pivot.shape

                # Ghi data raw (no header/index) vào vị trí (1,1)
                df_pivot.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=1,
                    startcol=1,
                    header=False,
                    index=False,
                )

                worksheet = writer.sheets[sheet_name]
                worksheet.hide_gridlines(2)

                # ============================
                # HEADER hàng 0
                # ============================
                worksheet.write(0, 0, "Cohort", fmt_header)
                for col_idx, mob in enumerate(df_pivot.columns, start=1):
                    worksheet.write(0, col_idx, mob, fmt_header)

                # ============================
                # HEATMAP RANGE
                # ============================
                if n_rows > 0 and n_cols > 0:
                    worksheet.conditional_format(
                        1, 1,                # start row/col
                        n_rows, n_cols,      # end row/col
                        {
                            "type": "3_color_scale",
                            "min_color": "#63BE7B",
                            "mid_color": "#FFEB84",
                            "max_color": "#F8696B",
                        }
                    )

                # ============================
                # FORMAT TỪNG CELL (ROUND + FORMAT)
                # ============================
                for i, cohort in enumerate(df_pivot.index):
                    r_idx = 1 + i   # data row

                    # Cột A: Cohort
                    worksheet.write(r_idx, 0, cohort, fmt_cohort)

                    max_actual = actual_info.get((product, cohort), None)

                    for j, mob in enumerate(df_pivot.columns):
                        c_idx = 1 + j

                        raw_value = df_pivot.iat[i, j]

                        # Round 4 digits
                        try:
                            value = round(float(raw_value), 4)
                        except:
                            value = raw_value

                        # Chọn format
                        if (max_actual is not None) and (mob > max_actual):
                            fmt = fmt_forecast
                        elif (max_actual is not None) and (mob == max_actual):
                            fmt = fmt_red_bottom
                        else:
                            fmt = fmt_data

                        worksheet.write(r_idx, c_idx, value, fmt)

                # ============================
                # AUTO COLUMN WIDTH
                # ============================

                # Cột A width fixed = 12
                worksheet.set_column(0, 0, 12)

                # MOB columns auto
                for j, mob in enumerate(df_pivot.columns):
                    width = max(8, len(str(mob)))
                    worksheet.set_column(1 + j, 1 + j, width + 2)

    print(f"✔ Export lifecycle multi-product thành công → {filename}")


