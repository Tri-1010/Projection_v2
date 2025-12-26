from __future__ import annotations
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

    df = df_lifecycle.copy()
    # Normalize key dtypes to avoid merge mismatches that zero out IS_FORECAST.
    for col in ("PRODUCT_TYPE", "RISK_SCORE"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["VINTAGE_DATE"] = pd.to_datetime(df["VINTAGE_DATE"], errors="coerce")

    df_raw_norm = df_raw.copy()
    for col in ("PRODUCT_TYPE", "RISK_SCORE"):
        if col in df_raw_norm.columns:
            df_raw_norm[col] = df_raw_norm[col].astype(str)
    df_raw_norm[orig] = pd.to_datetime(df_raw_norm[orig], errors="coerce")

    actual_max = (
        df_raw_norm.groupby(["PRODUCT_TYPE", "RISK_SCORE", orig])[mob_col]
        .max()
        .rename("ACTUAL_MAX_MOB")
        .reset_index()
        .rename(columns={orig: "VINTAGE_DATE"})
    )

    df = df.merge(actual_max,
                            on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"],
                            how="left")

    df["IS_FORECAST"] = (df["MOB"] > df["ACTUAL_MAX_MOB"]).astype(int)

    return df


# ============================================================
# 6️⃣ DEL30+, DEL60+, DEL90+ amount-based
# ============================================================

def add_del_metrics(df_lifecycle: pd.DataFrame, df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Tính DEL30/60/90 amount + % trên DISB_TOTAL.

    Logic chuẩn:
        - Nếu df_lifecycle CHƯA có DISB_TOTAL:
            + Bắt buộc truyền df_raw (loan-level) để tính DISB_TOTAL đúng:
                * Mỗi loan chỉ đóng góp 1 lần DISBURSAL_AMOUNT
                * Sum theo PRODUCT_TYPE × RISK_SCORE × VINTAGE_DATE
        - Nếu df_lifecycle ĐÃ có DISB_TOTAL:
            + Không cần df_raw, chỉ dùng DISB_TOTAL sẵn có.

    Parameters
    ----------
    df_lifecycle : DataFrame
        Long lifecycle (PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, MOB, các bucket DPD*, WRITEOFF,...)
    df_raw : DataFrame, optional
        Loan-level raw data, dùng để tính DISB_TOTAL nếu df_lifecycle chưa có cột này.

    Returns
    -------
    DataFrame
        df_lifecycle kèm thêm:
            - DEL30_AMT, DEL60_AMT, DEL90_AMT
            - DEL30_PCT, DEL60_PCT, DEL90_PCT
            - DISB_TOTAL (nếu chưa có)
    """

    import numpy as np

    disp_col  = CFG["disb"]
    orig_col  = CFG["orig_date"]
    loan_col  = CFG["loan"]

    df = df_lifecycle.copy()

    # ===================================================
    # 1️⃣ Nếu chưa có DISB_TOTAL thì tính từ df_raw
    # ===================================================
    if "DISB_TOTAL" not in df.columns:
        if df_raw is None:
            raise KeyError(
                "add_del_metrics: df_lifecycle chưa có DISB_TOTAL và df_raw=None "
                "→ không biết tính DISB_TOTAL từ đâu."
            )

        # DISB per loan (mỗi loan đúng 1 lần)
        loan_disb = (
            df_raw
            .groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col, loan_col])[disp_col]
            .first()          # hoặc .max(), vì disb không đổi theo loan
            .reset_index()
        )

        # DISB_TOTAL per cohort nhỏ (Product × Score × Vintage)
        cohort_disb = (
            loan_disb
            .groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col])[disp_col]
            .sum()
            .rename("DISB_TOTAL")
            .reset_index()
            .rename(columns={orig_col: "VINTAGE_DATE"})
        )

        # Merge DISB_TOTAL vào lifecycle (theo PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE)
        df = df.merge(
            cohort_disb,
            on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"],
            how="left"
        )

    # Đảm bảo DISB_TOTAL tồn tại, dù có thể NaN cho một số dòng (vd: sale plan)
    if "DISB_TOTAL" not in df.columns:
        df["DISB_TOTAL"] = np.nan

    # ===================================================
    # 2️⃣ Tính DEL30/60/90 AMOUNT
    # ===================================================
    df["DEL30_AMT"] = df[BUCKETS_30P].sum(axis=1)
    df["DEL60_AMT"] = df[BUCKETS_60P].sum(axis=1)
    df["DEL90_AMT"] = df[BUCKETS_90P].sum(axis=1)

    # ===================================================
    # 3️⃣ Tính % trên DISB_TOTAL
    #     - Nếu DISB_TOTAL = 0 hoặc NaN → PCT = NaN
    # ===================================================
    denom = df["DISB_TOTAL"].replace(0, np.nan)

    df["DEL30_PCT"] = df["DEL30_AMT"] / denom
    df["DEL60_PCT"] = df["DEL60_AMT"] / denom
    df["DEL90_PCT"] = df["DEL90_AMT"] / denom

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

def aggregate_products_to_portfolio(
    df_del_prod: pd.DataFrame,
    portfolio_name: str = "PORTFOLIO_ALL",
    product_filter: list[str] | None = None,
):
    """
    Gộp nhiều PRODUCT_TYPE thành 1 portfolio chung.
    
    Cách tính:
        - Weight theo PRODUCT_DISB_product / PORTFOLIO_DISB_vintage
        - Portfolio DEL30/60/90 = weighted-average các product theo weight
    """

    df = df_del_prod.copy()

    # Nếu muốn chỉ lấy một nhóm product con (ví dụ: ["SALPIL", "CARD"])
    if product_filter is not None:
        df = df[df["PRODUCT_TYPE"].isin(product_filter)]

    # 1️⃣ Lấy PRODUCT_DISB duy nhất theo Product × Vintage
    prod_disb = (
        df.groupby(["PRODUCT_TYPE", "VINTAGE_DATE"])["PRODUCT_DISB"]
        .first()               # vì PRODUCT_DISB không đổi theo MOB
        .reset_index()
    )

    # 2️⃣ Tổng disb toàn portfolio theo Vintage
    port_disb = (
        prod_disb.groupby("VINTAGE_DATE")["PRODUCT_DISB"]
        .sum()
        .rename("PORTFOLIO_DISB")
        .reset_index()
    )

    # 3️⃣ Merge PORTFOLIO_DISB vào từng dòng (Product × Vintage × MOB)
    df = df.merge(port_disb, on="VINTAGE_DATE", how="left")

    # 4️⃣ Weight = PRODUCT_DISB_product / PORTFOLIO_DISB
    df["WEIGHT_PORT"] = df["PRODUCT_DISB"] / df["PORTFOLIO_DISB"]

    # 5️⃣ Aggregate theo Vintage × MOB → portfolio
    agg = (
        df.groupby(["VINTAGE_DATE", "MOB"])
        .apply(
            lambda g: pd.Series({
                "DEL30_PCT": (g["DEL30_PCT"] * g["WEIGHT_PORT"]).sum(),
                "DEL60_PCT": (g["DEL60_PCT"] * g["WEIGHT_PORT"]).sum(),
                "DEL90_PCT": (g["DEL90_PCT"] * g["WEIGHT_PORT"]).sum(),
                # tổng disb portfolio tại cohort (VINTAGE_DATE)
                "PRODUCT_DISB": g["PORTFOLIO_DISB"].iloc[0],
            })
        )
        .reset_index()
    )

    # 6️⃣ Thêm cột PRODUCT_TYPE = tên portfolio
    agg["PRODUCT_TYPE"] = portfolio_name

    # Reorder cột cho giống df_del_prod
    cols = [
        "PRODUCT_TYPE",
        "VINTAGE_DATE",
        "MOB",
        "DEL30_PCT",
        "DEL60_PCT",
        "DEL90_PCT",
        "PRODUCT_DISB",
    ]
    agg = agg[cols]

    return agg


def export_lifecycle_all_products_one_file(
    df_del_prod,
    actual_info,
    filename="Lifecycle_All_Products.xlsx"
):
    """
    Xuất tất cả Product × (DEL30, DEL60, DEL90) vào nhiều sheet trong 1 file Excel.
    Gồm:
        ✔ Title lớn A1:J1
        ✔ Heatmap
        ✔ Format %
        ✔ Forecast highlight vàng
        ✔ Boundary đỏ đậm (actual cuối)
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

        # Title format
        fmt_title = workbook.add_format({
            "bold": True,
            "font_size": 20,
            "font_color": "#00008B",  # dark blue
            "align": "center",
            "valign": "vcenter",
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

                # Write raw data starting at row 5 (startrow=4)
                df_pivot.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=4,
                    startcol=1,
                    header=False,
                    index=False,
                )

                worksheet = writer.sheets[sheet_name]
                worksheet.hide_gridlines(2)

                # ============================
                # TITLE ROW (A1:J1)
                # ============================
                title_text = f"{product}_{metric_name} Actual & Forecast"
                worksheet.merge_range("A1:J1", title_text, fmt_title)

                # ============================
                # HEADER ROW (row index 3 => dòng 4)
                # ============================
                header_row = 3
                worksheet.write(header_row, 0, "Cohort", fmt_header)

                for col_idx, mob in enumerate(df_pivot.columns, start=1):
                    worksheet.write(header_row, col_idx, mob, fmt_header)

                # ============================
                # HEATMAP RANGE
                # ============================
                if n_rows > 0 and n_cols > 0:
                    worksheet.conditional_format(
                        4, 1,                # start row/col (data)
                        4 + n_rows - 1, 1 + n_cols - 1,
                        {
                            "type": "3_color_scale",
                            "min_color": "#63BE7B",
                            "mid_color": "#FFEB84",
                            "max_color": "#F8696B",
                        }
                    )

                # ============================
                # FORMAT TỪNG CELL
                # ============================
                for i, cohort in enumerate(df_pivot.index):
                    r_idx = 4 + i   # data row starts from row 5

                    # Cột A: Cohort
                    worksheet.write(r_idx, 0, cohort, fmt_cohort)

                    max_actual = actual_info.get((product, cohort), None)

                    for j, mob in enumerate(df_pivot.columns):
                        c_idx = 1 + j
                        raw_value = df_pivot.iat[i, j]

                        try:
                            value = round(float(raw_value), 4)
                        except:
                            value = raw_value

                        # Chọn format actual vs forecast
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

                worksheet.set_column(0, 0, 12)  # Cohort column

                for j, mob in enumerate(df_pivot.columns):
                    width = max(8, len(str(mob)))
                    worksheet.set_column(1 + j, 1 + j, width + 2)
    # ============================
    # MOVE PORTFOLIO SHEETS TO TOP
    # ============================
    # Lấy danh sách sheet hiện có
    sheetnames = writer.book.worksheets()
    
    # Sheet nào có chữ "Portfolio" sẽ được đưa lên trước
    portfolio_sheets = [s for s in sheetnames if "Portfolio" in s.name]
    other_sheets = [s for s in sheetnames if "Portfolio" not in s.name]
    
    # Xếp lại thứ tự: Portfolio trước, rồi các sheet còn lại
    writer.book.worksheets_objs = portfolio_sheets + other_sheets

    print(f"✔ Export lifecycle multi-product thành công → {filename}")



def extend_actual_info_with_portfolio(
    actual_info_prod: Dict[tuple, int],
    portfolio_name: str = "PORTFOLIO_ALL",
):
    """
    actual_info_prod: {(product, cohort) -> max_actual_mob}
    Trả về:
        actual_info_all: gồm cả product và portfolio
        trong đó:
            (portfolio_name, cohort) = max_mob của tất cả product tại cohort đó
    """

    actual_info_port = {}

    for (product, cohort), max_mob in actual_info_prod.items():
        key_port = (portfolio_name, cohort)
        if key_port not in actual_info_port:
            actual_info_port[key_port] = max_mob
        else:
            actual_info_port[key_port] = max(
                actual_info_port[key_port],
                max_mob
            )

    # Gộp lại: product-level + portfolio-level
    actual_info_all = {**actual_info_prod, **actual_info_port}
    return actual_info_all
def add_del_metrics_for_sale_plan(
    df_plan_fc: pd.DataFrame,
    sale_plan_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tính DEL30/60/90 cho SALE PLAN forecast.

    Giả định:
        - df_plan_fc: output từ forecast_sale_plan_by_mob
            cột bắt buộc:
                PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, MOB
                các bucket EAD: DPD0, DPD1+, DPD30+, ..., WRITEOFF, PREPAY
        - sale_plan_df:
            cohort-level:
                PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, EAD_PLAN (tổng disb / EAD gốc)

    Logic:
        - DISB_TOTAL (per cohort) = EAD_PLAN
        - DEL30_AMT = sum(EAD ở BUCKETS_30P)
        - DELxx_PCT = DELxx_AMT / DISB_TOTAL
        - IS_FORECAST = 1, SOURCE = "SALE_PLAN"
    """

    df = df_plan_fc.copy()

    # Chuẩn hóa VINTAGE_DATE để merge
    sp = sale_plan_df.copy()
    if "VINTAGE_DATE" in sp.columns:
        sp["VINTAGE_DATE"] = pd.to_datetime(sp["VINTAGE_DATE"])

    # Lấy DISB_TOTAL từ EAD_PLAN
    disb_map = (
        sp[["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "EAD_PLAN"]]
        .rename(columns={"EAD_PLAN": "DISB_TOTAL"})
    )

    df = df.merge(
        disb_map,
        on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"],
        how="left",
    )

    # Tính DEL30/60/90 amount từ các bucket EAD
    df["DEL30_AMT"] = df[BUCKETS_30P].sum(axis=1)
    df["DEL60_AMT"] = df[BUCKETS_60P].sum(axis=1)
    df["DEL90_AMT"] = df[BUCKETS_90P].sum(axis=1)

    # Mẫu số an toàn (tránh chia 0)
    denom = df["DISB_TOTAL"].replace(0, np.nan)

    df["DEL30_PCT"] = df["DEL30_AMT"] / denom
    df["DEL60_PCT"] = df["DEL60_AMT"] / denom
    df["DEL90_PCT"] = df["DEL90_AMT"] / denom

    # Flag source
    if "IS_FORECAST" not in df.columns:
        df["IS_FORECAST"] = 1
    else:
        df["IS_FORECAST"] = 1  # sale plan luôn là forecast

    df["SOURCE"] = "SALE_PLAN"

    return df
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


def build_full_lifecycle_amount(
    df_raw: pd.DataFrame,
    matrices_by_mob: Dict,
    max_mob: int = 29,
):
    """
    Pipeline:
        1) forecast_all_vintages → forecast_results (dict)
        2) get_actual_all_vintages_amount → actual_results (dict)
        3) combine_all_lifecycle_amount → lifecycle (dict)
        4) lifecycle_to_long_df_amount → DataFrame long-format

    Output:
        df_long với cột BUCKETS_CANON (EAD per state), PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, MOB
    """

    # 1️⃣ Forecast
    forecast_results = forecast_all_vintages(
        df_raw=df_raw,
        matrices_by_mob=matrices_by_mob,
        max_mob=max_mob,
        enable_macro=False,
    )

    # 2️⃣ Actual
    actual_results = get_actual_all_vintages_amount(df_raw)

    # 3️⃣ Merge actual + forecast
    lifecycle = combine_all_lifecycle_amount(actual_results, forecast_results)

    # 4️⃣ Long format
    df_long = lifecycle_to_long_df_amount(lifecycle)

    return df_long
