from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

import config
from src.data import db
from src.data.schema import DataSchema, default_schema


def load_parquet(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {path}")
    files = sorted(list(path.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def load_oracle(sql: Optional[str] = None, params: Optional[dict] = None, sql_dir: Optional[str] = None, debug: bool = False) -> pd.DataFrame:
    cfg = config.ORACLE_CONFIG
    sql_text = sql or cfg.get("sql")
    if not sql_text:
        raise ValueError("SQL text must be provided for Oracle data source.")
    return db.load_df(sql_text, params=params or cfg.get("params"), sql_dir=sql_dir or cfg.get("sql_dir"), debug=debug)


def load_raw_data(schema: Optional[DataSchema] = None, source: Optional[str] = None, parquet_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw data according to config. Supports parquet directory or Oracle query."""
    schema = schema or default_schema()
    source_choice = (source or config.DATA_SOURCE).lower()

    if source_choice == "parquet":
        data = load_parquet(parquet_path or config.PARQUET_PATH)
    elif source_choice == "oracle":
        data = load_oracle()
    else:
        raise ValueError(f"Unsupported DATA_SOURCE: {source_choice}")

    required_cols = list(schema.required_columns)
    if schema.cohort_col and schema.cohort_col not in data.columns:
        required_cols = [c for c in required_cols if c != schema.cohort_col]
    missing_cols = set(required_cols) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Input data missing required columns: {missing_cols}")

    return data
