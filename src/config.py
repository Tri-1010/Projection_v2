from pathlib import Path


# ===== Resolve project root from this file path (stable across notebooks/scripts) =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../RR_model
OUT_ROOT     = PROJECT_ROOT / "outputs"

# Data source defaults to parquet for offline runs
DATA_SOURCE  = "parquet"  # options: "parquet" | "oracle"
PARQUET_DIR  = PROJECT_ROOT / "data" / "parquet"       # <-- FIXED: absolute path
PARQUET_FILE = None  # or "rollrate_base.parquet" if bạn dùng 1 file duy nhất

EXCEL_FILE   = PROJECT_ROOT / "data" / "rollrate_input.xlsx"   # 👈 đường dẫn mặc định nếu dùng Excel
EXCEL_SHEET  = "Data"    
# === COLUMNS CONFIG & others giữ nguyên ===

# ===========================
# B. Model parameters
# ===========================
MIN_OBS = 100         # Số quan sát tối thiểu
MIN_EAD = 1e2         # Tổng dư nợ tối thiểu để build transition
BUCKETS_30P = ["DPD30+", "DPD60+", "DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]
BUCKETS_60P = ["DPD60+", "DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]
BUCKETS_90P = ["DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]
# === COLUMNS CONFIG ===
CFG = dict(
    loan="AGREEMENT_ID",
    mob="MOB",
    state="STATE_MODEL",
    orig_date="DISBURSAL_DATE",
    ead="PRINCIPLE_OUTSTANDING",
    disb="DISBURSAL_AMOUNT",
    cutoff="CUTOFF_DATE",
)

# === SEGMENTATION CONFIG ===
SEGMENT_COLS = ["RISK_SCORE", "PRODUCT_TYPE"]
#SEGMENT_COLS = ["RISK_SCORE"]
SEGMENT_MAP = {
    "RISK_SCORE": ["LOW", "MEDIUM", "HIGH"],
    "PRODUCT_TYPE": ["PL", "CC"],
}


# === SMOOTHING CONFIG ===
ALPHA_SMOOTH = 0.5

# === STATE DEFINITIONS ===
BUCKETS_CANON = [
    "DPD0", "DPD1+", "DPD30+", "DPD60+", "DPD90+", "DPD120+", "DPD180+",
    "PREPAY", "WRITEOFF", "SOLDOUT"
]

#ABSORBING_BASE = ["WRITEOFF", "PREPAY", "SOLDOUT"]
ABSORBING_BASE = ["DPD90+", "WRITEOFF", "PREPAY", "SOLDOUT"] # PD model

DEFAULT = {"DPD90+"}

# === MODEL CONFIG ===
#WEIGHT_METHOD = "exp"
WEIGHT_METHOD = None
ROLL_WINDOW = 6
CFG["ROLL_WINDOW"] = ROLL_WINDOW

# === MACRO & COLLX ADJUSTMENT CONFIG (optional, not wired by default) ===
MACRO_INDICATORS = {
    "GDP_GROWTH": {"weight": -0.3},
    "UNEMPLOYMENT_RATE": {"weight": +0.5},
    "CPI": {"weight": +0.2},
    "POLICY_RATE": {"weight": +0.3},
}
COLLX_CONFIG = {
    "COLLX_INDEX": {
        "weight": -0.4,
        "ref_value": 1.0,
        "min_adj": -0.3,
        "max_adj": +0.3,
    }
}
ADJUST_METHOD = "multiplicative"
MACRO_LAG = 1
MACRO_SOURCE = "sql/macro_data.sql"
COLLX_SOURCE = "sql/collx_index.sql"
