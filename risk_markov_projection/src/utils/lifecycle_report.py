from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def build_lifecycle_for_report(df_actual: pd.DataFrame, df_plan_fc: pd.DataFrame, buckets: Iterable[str]) -> pd.DataFrame:
    """
    Combine actual and forecast lifecycle data into a single DataFrame for reporting.

    Args:
        df_actual: DataFrame with actual metrics.
        df_plan_fc: DataFrame with forecast metrics.
        buckets: Iterable of bucket column names to retain (e.g., ["DPD30+", "DPD60+", "DPD90+"]).

    Returns:
        DataFrame with columns PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE, MOB, bucket columns, and is_forecast flag.
    """
    keep_cols = ["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"] + list(buckets)

    actual = df_actual.copy()
    actual["is_forecast"] = 0
    actual = actual[keep_cols + ["is_forecast"]]

    fc = df_plan_fc.copy()
    fc["is_forecast"] = 1
    fc = fc[keep_cols + ["is_forecast"]]

    df_all = pd.concat([actual, fc], ignore_index=True)
    df_all = df_all.sort_values(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"]).reset_index(drop=True)
    return df_all


def export_lifecycle_all_products_one_file_extended(
    df_lifecycle: pd.DataFrame,
    actual_info: Dict[Tuple[str, pd.Timestamp], int],
    filename: str | Path,
    metric_map: Dict[str, str] | None = None,
) -> Path:
    """
    Export lifecycle pivot for each product/metric to a single Excel file (xlsxwriter).

    Args:
        df_lifecycle: DataFrame from build_lifecycle_for_report (contains is_forecast, bucket columns).
        actual_info: Dict with key (PRODUCT_TYPE, VINTAGE_DATE) -> MOB boundary for actual (to highlight boundary).
        filename: Output Excel file path.
        metric_map: Optional map metric_name -> bucket column. Default DEL30/60/90 to DPD30+/60+/90+.
    """
    metric_map = metric_map or {"DEL30": "DPD30+", "DEL60": "DPD60+", "DEL90": "DPD90+"}
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    df = df_lifecycle.copy()
    df["VINTAGE_DATE"] = pd.to_datetime(df["VINTAGE_DATE"])

    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        for product, prod_df in df.groupby("PRODUCT_TYPE"):
            for metric, bucket_col in metric_map.items():
                sheet_name = f"{product}_{metric}"[:31]
                if bucket_col not in prod_df.columns:
                    continue

                pivot = (
                    prod_df.pivot_table(
                        index="VINTAGE_DATE",
                        columns="MOB",
                        values=bucket_col,
                        aggfunc="first",
                        fill_value=0,
                    )
                    .sort_index()
                )

                # Formats
                workbook = writer.book
                header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9D9D9", "border": 1, "align": "center"})
                data_fmt = workbook.add_format({"num_format": "0.0000", "border": 1})
                forecast_fmt = workbook.add_format({"num_format": "0.0000", "border": 1, "bg_color": "#FFC000"})
                boundary_fmt = workbook.add_format(
                    {
                        "num_format": "0.0000",
                        "border": 1,
                        "right": 2,
                        "top": 2,
                        "bottom": 2,
                        "right_color": "#FF0000",
                        "top_color": "#FF0000",
                        "bottom_color": "#FF0000",
                    }
                )
                cohort_fmt = workbook.add_format({"num_format": "yyyy-mm-dd", "border": 1})

                ws = workbook.add_worksheet(sheet_name)
                ws.hide_gridlines(2)

                # Headers
                ws.write(0, 0, "Cohort", header_fmt)
                mobs = list(pivot.columns)
                for j, mob in enumerate(mobs, start=1):
                    ws.write(0, j, mob, header_fmt)

                # Data rows
                for i, cohort in enumerate(pivot.index, start=1):
                    ws.write_datetime(i, 0, cohort, cohort_fmt)
                    for j, mob in enumerate(mobs, start=1):
                        val = float(pivot.iloc[i - 1, j - 1])
                        # Determine if forecast
                        mask = (prod_df["VINTAGE_DATE"] == cohort) & (prod_df["MOB"] == mob)
                        row_slice = prod_df.loc[mask]
                        is_fc = False
                        if not row_slice.empty:
                            is_fc = bool(row_slice["is_forecast"].max() == 1)
                        fmt = forecast_fmt if is_fc else data_fmt
                        # Boundary check
                        boundary_mob = actual_info.get((product, cohort))
                        if boundary_mob is not None and boundary_mob == mob:
                            fmt = boundary_fmt
                        ws.write(i, j, val, fmt)

                # Column widths
                ws.set_column(0, 0, 12)
                for j, mob in enumerate(mobs, start=1):
                    ws.set_column(j, j, max(8, len(str(mob)) + 2))

    return filename
