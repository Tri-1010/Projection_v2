from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.config import (
    DATA_SOURCE,
    PARQUET_DIR,
    PARQUET_FILE,
    EXCEL_FILE,
    EXCEL_SHEET,
)

def _read_parquet_dir(parquet_dir: Path) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(str(parquet_dir), format="parquet")
        table = dataset.to_table()
        df = table.to_pandas()
        print(f"✅ Loaded {len(df):,} rows via pyarrow.dataset from {parquet_dir}")
        return df
    except Exception as e:
        print(f"⚠️ pyarrow.dataset not used ({e}). Falling back to glob concat...")
        files = sorted(parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Không tìm thấy *.parquet trong {parquet_dir}")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(df):,} rows from {len(files)} files in {parquet_dir}")
        return df


def load_data(sql_or_file: str = None, params: dict | None = None) -> pd.DataFrame:
    ds = DATA_SOURCE.lower()

    # ------------------- Oracle -------------------
    if ds == "oracle":
        print("?? Loading data from Oracle...")
        if sql_or_file is None:
            raise ValueError("C?n ch? d?nh t?n SQL file ho?c c?u SQL khi d?ng Oracle.")
        from src.db import load_df
        return load_df(sql_or_file, params=params)
    elif ds == "parquet":
        parquet_dir = PARQUET_DIR if sql_or_file is None else Path(sql_or_file)
        print(f"📦 Loading Parquet from: {parquet_dir.resolve()}")
        if parquet_dir.is_dir():
            df = _read_parquet_dir(parquet_dir)
        elif parquet_dir.suffix.lower() == ".parquet" and parquet_dir.exists():
            df = pd.read_parquet(parquet_dir)
            print(f"✅ Loaded {len(df):,} rows from {parquet_dir.name}")
        else:
            raise FileNotFoundError(f"Không tìm thấy thư mục/file parquet: {parquet_dir}")

    # ------------------- Excel -------------------
    elif ds == "excel":
        excel_path = Path(EXCEL_FILE)
        if sql_or_file:
            excel_path = Path(sql_or_file)
        if not excel_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file Excel: {excel_path}")
        print(f"📗 Loading Excel data from {excel_path} (sheet='{EXCEL_SHEET}')")
        df = pd.read_excel(excel_path, sheet_name=EXCEL_SHEET)
        print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns from Excel")

    else:
        raise ValueError(f"DATA_SOURCE không hợp lệ: {DATA_SOURCE}. Chọn 'oracle', 'parquet', hoặc 'excel'.")

    # ------------------- Chuẩn hóa cột và thêm PRODUCT_TYPE -------------------
    df.columns = [c.upper() for c in df.columns]
    if "PRODUCT_TYPE" not in df.columns:
        df["PRODUCT_TYPE"] = "A"
        print("ℹ️ Added default column PRODUCT_TYPE = 'A'")
    return df
