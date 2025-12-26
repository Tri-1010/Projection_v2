import pandas as pd
from pathlib import Path
from src.data_loader import load_data
from src.rollrate.segment import generate_all_transitions
from src.rollrate.forecast import forecast_report
from src.rollrate.transition import compute_transition
from src.config import OUT_ROOT, CFG, ALPHA_SMOOTH

def run_rollrate_pipeline(as_of_month: str, forecast_months: int = 12):
    """
    Pipeline chính:
      1️⃣ Load data (từ parquet hoặc Oracle)
      2️⃣ Sinh transition matrices theo PRODUCT_TYPE
      3️⃣ Forecast danh mục 12 tháng tới
      4️⃣ Xuất file Excel tổng hợp
    """
    print(f"=== Running Roll Rate Pipeline | AS_OF_MONTH={as_of_month} ===")

    # 1️⃣ Load dữ liệu
    df = load_data()  # mặc định đọc parquet theo config
    print(f"Loaded {len(df):,} rows from source.")

    # 2️⃣ Xác định cột thời gian (as_of / mob)
    as_of_col = CFG.get("as_of", CFG["mob"])
    latest_month = df[as_of_col].max()
    df_latest = df[df[as_of_col] == latest_month].copy()
    print(f"📆 Latest snapshot = {latest_month}")

    # 3️⃣ Tạo ma trận transition theo từng sản phẩm
    matrices = {}
    for subprod in df["PRODUCT_TYPE"].dropna().unique():
        sub_df = df[df["PRODUCT_TYPE"] == subprod].copy()
        if len(sub_df) < 100:
            print(f"⚠️ Skip {subprod}: sample quá nhỏ ({len(sub_df)}). Dùng fallback.")
            P = compute_transition(df, value_col=CFG["ead"])  # fallback global
        else:
            P = compute_transition(sub_df, value_col=CFG["ead"])

        # Kiểm tra smoothing
        if (P.sum().sum() == 0):
            print(f"⚠️ Empty transition matrix for {subprod}. Using smoothed global fallback.")
            P = compute_transition(df, value_col=CFG["ead"])

        matrices[subprod] = P

    print(f"✅ Built {len(matrices)} product-level transition matrices.")

    # 4️⃣ Forecast danh mục (12 tháng)
    reports, summary = forecast_report(
        df_latest, matrices,
        months=forecast_months,
        value_col=CFG["ead"]
    )

    # 5️⃣ Xuất kết quả
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_summary = OUT_ROOT / f"forecast_portfolio_summary_{as_of_month}.xlsx"

    with pd.ExcelWriter(out_summary, engine="openpyxl") as writer:
        for k, v in reports.items():
            v.to_excel(writer, sheet_name=str(k)[:31], index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"✅ Pipeline hoàn tất. Output summary: {out_summary}")
