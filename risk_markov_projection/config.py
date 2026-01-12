from pathlib import Path
from typing import List

# Data source selection: "oracle" or "parquet"
DATA_SOURCE = "parquet"

# Oracle connection settings (used only when DATA_SOURCE == "oracle")
ORACLE_CONFIG = {
    "sql": "SELECT * FROM RISK_MARKOV_VIEW",
    "params": {},
    "sql_dir": "sql",
}

# Parquet input (used only when DATA_SOURCE == "parquet")
PARQUET_PATH = Path("data/parquet_sample")

# Schema mapping
SCHEMA = {
    "id_col": "AGREEMENT_ID",
    "mob_col": "MOB",
    "state_col": "STATE_MODEL",
    "ead_col": "PRINCIPLE_OUTSTANDING",
    "date_col": "CUTOFF_DATE",
    "segment_cols": ["RISK_SCORE", "PRODUCT_TYPE"],
    # Optional: disbursal/cohort column for cohort reporting
    "cohort_col": None,  # e.g., "DISBURSAL_DATE"
}

# States and absorbing rules
STATE_ORDER: List[str] = [
    "DPD0",       # current-equivalent
    "CURRENT",
    "DPD1+",
    "DPD30+",
    "DPD60+",
    "DPD90+",
    "DPD120+",
    "DPD180+",
    "WRITEOFF",
    "PREPAY",
    "CLOSED",
]
ABSORBING_STATES: List[str] = ["WRITEOFF", "PREPAY", "CLOSED"]
CLOSED_ABSORBING: bool = True

# Thresholds
MIN_OBS: int = 100
MIN_EAD: float = 1e2
MAX_MOB: int = 24

# Transition estimation
TRANSITION_WEIGHT_MODE: str = "ead"  # supported: "ead", "count"
SMOOTHING = {
    "strength": 5.0,  # higher -> more shrink toward parent
    "parent_order": ["segment_all", "global_mob", "global_all"],
    "min_prob": 1e-12,
}
FALLBACK_ORDER = [
    "segment_mob",
    "segment_nearest_previous_mob",
    "segment_all",
    "global_mob",
    "global_all",
    "identity",
]

# Buckets for delinquency metrics
BUCKETS_30P = ["DPD30+", "DPD60+", "DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]
BUCKETS_60P = ["DPD60+", "DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]
BUCKETS_90P = ["DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]

# Calibration
CALIBRATION = {
    "enabled": True,
    "method": "scalar_by_mob",
    "target_indicator": "DEL_30P_ON_EAD0",
    "lower_bound": 0.6,
    "upper_bound": 1.4,
}

# Outputs
OUTPUT = {
    "dir": Path("outputs"),
    "csv_name": "projection.csv",
    "parquet_name": "projection.parquet",
    "report_name": "indicator_report.csv",
    "cohort_report_name": "cohort_del30_report.csv",
    "cohort_report_excel_name": "cohort_del30_report.xlsx",
}

# Logging
LOGGING = {
    "level": "INFO",
}
