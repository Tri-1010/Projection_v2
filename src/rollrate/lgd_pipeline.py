# ============================================================
#  LGD Pipeline – Multi-RW (12/18/24) + GDP Scenario
#  - Input: raw file LGD2.parquet (hoặc qua load_data)
#  - Output:
#       LGD_lookup_RW12.xlsx
#       LGD_lookup_RW18.xlsx
#       LGD_lookup_RW24.xlsx
#       LGD_lookup_all.xlsx
#       LGD_base_final.xlsx
#       LGD_scenario_GDP_adjusted.xlsx
# ============================================================

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1) Setup đường dẫn & import module project
# ------------------------------------------------------------

root = Path(".").resolve()
sys.path.append(str(root / "src"))

from src.data_loader import load_data
from src.config import CFG, OUT_ROOT


# ------------------------------------------------------------
# 2) CONFIG LGD
# ------------------------------------------------------------

# Giá bán nợ giả định: 0.02% * EAD_default
PRICE_RATE = 0.0002

# Chỉ giữ default từ kỳ này trở đi
CUTOFF_MIN_DATE = "2023-07-01"

# Bucket MOB theo tuổi tại thời điểm default
MOB_BUCKETS = [
    (0, 6),
    (7, 12),
    (13, 24),
    (25, 36),
    (37, 999),
]

# Ngưỡng tối thiểu để tin LGD tại 1 RW cho 1 segment
MIN_N_RW = 30

# Macro config (GDP)
GDP_NORMAL = 6.5    # GDP trạng thái "bình thường" (%)
BETA_LGD   = 0.01   # 1% GDP shock → LGD thay đổi 1% * beta

SCENARIOS = [
    {"SCENARIO": "Base",    "GDP_YOY": 6.0},
    {"SCENARIO": "Down",    "GDP_YOY": 4.0},
    {"SCENARIO": "Severe",  "GDP_YOY": 1.0},
    {"SCENARIO": "Upside",  "GDP_YOY": 7.0},
]


# ------------------------------------------------------------
# 3) Helper functions
# ------------------------------------------------------------

def months_between(d1, d2):
    """Tính chênh lệch số tháng giữa 2 date (d1 - d2)."""
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)


def assign_mob_bucket(mob: int) -> str:
    """Map MOB về MOB_BUCKET theo cấu hình MOB_BUCKETS."""
    for low, high in MOB_BUCKETS:
        if low <= mob <= high:
            return f"{low}-{high}"
    return "NA"


def get_product_type(row) -> str:
    """
    RULE:
    - Nếu PRODUCT_2 == 'POS_LOAN' → dùng PRODUCT
    - Ngược lại                  → dùng PRODUCT_2
    """
    if str(row.get("PRODUCT_2", "")).upper() == "POS_LOAN":
        return str(row.get("PRODUCT", "")).upper()
    return str(row.get("PRODUCT_2", "")).upper()


# ------------------------------------------------------------
# 4) PREPROCESS
# ------------------------------------------------------------

def preprocess_lgd_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu raw để tính LGD:
      - Parse date
      - Lọc cutoff min
      - Xác định PRODUCT_TYPE / PRODUCT_SEGMENT
      - Tính EAD_default & price (soldout)
      - Tính MOB_default & MOB_BUCKET
    """
    df = df.copy()

    df["CUTOFF_DATE_M0"] = pd.to_datetime(df["CUTOFF_DATE_M0"])
    df["DISBURSAL_DATE"] = pd.to_datetime(df["DISBURSAL_DATE"])

    # Lọc tối thiểu theo cutoff
    df = df[df["CUTOFF_DATE_M0"] >= CUTOFF_MIN_DATE].copy()

    # PRODUCT TYPE + SEGMENT
    df["PRODUCT_TYPE"] = df.apply(get_product_type, axis=1)
    df["PRODUCT_SEGMENT"] = df["PRODUCT_TYPE"]

    # EAD DEFAULT & PRICE
    df["EAD_default"] = df["M0_POS"]
    df["price"] = 0.0
    df.loc[df["FLAG_SOLDOUT"] == 1, "price"] = df["EAD_default"] * PRICE_RATE

    # MOB tại thời điểm default
    df["MOB_default"] = df.apply(
        lambda r: months_between(r["CUTOFF_DATE_M0"], r["DISBURSAL_DATE"]), axis=1
    )
    df["MOB_BUCKET"] = df["MOB_default"].apply(assign_mob_bucket)

    return df


# ------------------------------------------------------------
# 5) Function: compute LGD cho 1 RW
# ------------------------------------------------------------

def compute_lgd_for_rw(df: pd.DataFrame, RW: int, min_ead_default: float = 0.0):
    """
    Tính LGD tại RW (M12/M18/M24):
      LGD_RW = EAD_remaining_RW / EAD_default

    Steps:
      1) Chỉ giữ loan có M{RW}_POS khác NaN
      2) EAD_RW = M{RW}_POS
      3) EAD_remaining:
             - Bình thường: = EAD_RW
             - Soldout    : = EAD_default - price
      4) LGD_loan = EAD_rem_RW / EAD_default  (clamp [0,1])
      5) Group by PRODUCT_SEGMENT × MOB_BUCKET
    """

    if f"M{RW}_POS" not in df.columns:
        print(f"⚠️ compute_lgd_for_rw: không thấy cột M{RW}_POS, RW={RW} → bỏ qua.")
        return pd.DataFrame(), pd.DataFrame()

    # 1. Only loans with valid POS at RW
    df_rw = df[~df[f"M{RW}_POS"].isna()].copy()

    if min_ead_default > 0:
        df_rw = df_rw[df_rw["EAD_default"] >= min_ead_default].copy()

    if df_rw.empty:
        print(f"⚠️ compute_lgd_for_rw: RW={RW} không còn khoản vay hợp lệ.")
        return df_rw, pd.DataFrame()

    # 2. EAD at RW
    df_rw[f"EAD_{RW}"] = df_rw[f"M{RW}_POS"]

    # 3. EAD remaining
    df_rw[f"EAD_rem_{RW}"] = df_rw[f"EAD_{RW}"]

    # Sold-out → EAD_remaining = EAD_default - price
    df_rw.loc[df_rw["FLAG_SOLDOUT"] == 1, f"EAD_rem_{RW}"] = (
        df_rw["EAD_default"] - df_rw["price"]
    )

    # 4. LGD loan-level
    denom = df_rw["EAD_default"].replace(0, np.nan)
    df_rw[f"LGD_{RW}"] = df_rw[f"EAD_rem_{RW}"] / denom

    # Clamp LGD loan-level về [0,1]
    df_rw[f"LGD_{RW}"] = df_rw[f"LGD_{RW}"].clip(lower=0.0, upper=1.0)

    # 5. GROUP summary theo segment
    lookup = (
        df_rw.groupby(["PRODUCT_SEGMENT", "MOB_BUCKET"])
             .agg(
                 LGD_POINT       = (f"LGD_{RW}", "mean"),
                 N_LOANS         = ("AGREEMENT_ID", "nunique"),
                 EAD_DEFAULT_SUM = ("EAD_default", "sum"),
                 EAD_RW_SUM      = (f"EAD_{RW}", "sum"),
                 EAD_REMAIN_SUM  = (f"EAD_rem_{RW}", "sum")
             )
             .reset_index()
    )

    lookup["RW_MONTH"] = RW
    return df_rw, lookup


# ------------------------------------------------------------
# 6) Build lookup_all & LGD_BASE (từ 12/18/24)
# ------------------------------------------------------------

def build_lgd_lookup_all(df: pd.DataFrame):
    """
    Chạy RW12/18/24, merge thành lookup_all + chọn LGD_BASE:
      - Ưu tiên RW18 nếu N_LOANS>=MIN_N_RW
      - Sau đó RW24
      - Sau đó RW12
      - Nếu vẫn không có → mean của LGD_POINT tồn tại
      - Cuối cùng clamp về [0,1]
    """
    loan12, lookup12 = compute_lgd_for_rw(df, 12)
    loan18, lookup18 = compute_lgd_for_rw(df, 18)
    loan24, lookup24 = compute_lgd_for_rw(df, 24)

    # ----------
    # Merge 12 & 18
    # ----------
    if lookup12.empty and lookup18.empty and lookup24.empty:
        print("⚠️ Không có lookup nào cho RW12/18/24, trả về rỗng.")
        return lookup12, lookup18, lookup24, pd.DataFrame()

    # đảm bảo DataFrame không None
    if lookup12.empty:
        lookup12 = pd.DataFrame(columns=["PRODUCT_SEGMENT", "MOB_BUCKET",
                                         "LGD_POINT", "N_LOANS",
                                         "EAD_DEFAULT_SUM", "EAD_RW_SUM", "EAD_REMAIN_SUM", "RW_MONTH"])
    if lookup18.empty:
        lookup18 = pd.DataFrame(columns=lookup12.columns)
    if lookup24.empty:
        lookup24 = pd.DataFrame(columns=lookup12.columns)

    lookup_all = lookup12.merge(
        lookup18,
        on=["PRODUCT_SEGMENT", "MOB_BUCKET"],
        how="outer",
        suffixes=("_12", "_18")
    )

    lookup24 = lookup24.rename(columns={
        "LGD_POINT": "LGD_POINT_24",
        "N_LOANS": "N_LOANS_24",
        "EAD_DEFAULT_SUM": "EAD_DEFAULT_SUM_24",
        "EAD_RW_SUM": "EAD_RW_SUM_24",
        "EAD_REMAIN_SUM": "EAD_REMAIN_SUM_24"
    })

    lookup_all = lookup_all.merge(
        lookup24,
        on=["PRODUCT_SEGMENT", "MOB_BUCKET"],
        how="outer"
    )

    # Reorder & đảm bảo đủ cột
    cols = [
        "PRODUCT_SEGMENT", "MOB_BUCKET",
        "LGD_POINT_12", "LGD_POINT_18", "LGD_POINT_24",
        "N_LOANS_12", "N_LOANS_18", "N_LOANS_24",
        "EAD_DEFAULT_SUM_12", "EAD_DEFAULT_SUM_18", "EAD_DEFAULT_SUM_24",
        "EAD_RW_SUM_12", "EAD_RW_SUM_18", "EAD_RW_SUM_24",
        "EAD_REMAIN_SUM_12", "EAD_REMAIN_SUM_18", "EAD_REMAIN_SUM_24",
    ]

    for c in cols:
        if c not in lookup_all.columns:
            lookup_all[c] = np.nan

    lookup_all = lookup_all[cols]

    # ---- chọn LGD_BASE ----
    def choose_lgd_base(row):
        # ưu tiên RW18 → 24 → 12
        for rw in [18, 24, 12]:
            lgd_col = f"LGD_POINT_{rw}"
            n_col   = f"N_LOANS_{rw}"
            lgd = row.get(lgd_col, np.nan)
            n   = row.get(n_col, 0)

            if pd.notna(lgd) and n >= MIN_N_RW:
                return float(lgd)

        # fallback: trung bình các LGD_POINT không NaN
        vals = [v for v in [
            row.get("LGD_POINT_12", np.nan),
            row.get("LGD_POINT_18", np.nan),
            row.get("LGD_POINT_24", np.nan),
        ] if pd.notna(v)]
        if vals:
            return float(np.mean(vals))

        # không có gì tin được → default 1.0
        return 1.0

    lookup_all["LGD_BASE"] = lookup_all.apply(choose_lgd_base, axis=1)
    lookup_all["LGD_BASE"] = lookup_all["LGD_BASE"].clip(lower=0.0, upper=1.0)

    return loan12, lookup12, loan18, lookup18, loan24, lookup24, lookup_all


# ------------------------------------------------------------
# 7) Build LGD_base & LGD_scenario theo GDP
# ------------------------------------------------------------

def build_lgd_scenario(lgd_base: pd.DataFrame,
                       g_normal: float = GDP_NORMAL,
                       beta: float = BETA_LGD,
                       scenarios=None) -> pd.DataFrame:
    """
    Input:
      lgd_base: DataFrame có cột PRODUCT_SEGMENT, MOB_BUCKET, LGD_BASE
      scenarios: list dict [{"SCENARIO":..,"GDP_YOY":..}, ...]

    Output:
      DF: thêm SCENARIO, GDP_YOY, GDP_SHOCK, LGD_FACTOR, LGD_ADJ
    """
    if scenarios is None:
        scenarios = SCENARIOS

    lgd_base = lgd_base.copy()
    rows = []

    for sc in scenarios:
        shock = g_normal - sc["GDP_YOY"]      # GDP thấp hơn bình thường → shock dương
        factor = 1 + beta * shock            # LGD tăng khi GDP thấp

        df_temp = lgd_base.copy()
        df_temp["SCENARIO"] = sc["SCENARIO"]
        df_temp["GDP_YOY"] = sc["GDP_YOY"]
        df_temp["GDP_SHOCK"] = shock
        df_temp["LGD_FACTOR"] = factor
        df_temp["LGD_ADJ"] = (df_temp["LGD_BASE"] * df_temp["LGD_FACTOR"]).clip(0.0, 1.0)

        rows.append(df_temp)

    LGD_scenario = pd.concat(rows, ignore_index=True)
    return LGD_scenario


# ------------------------------------------------------------
# 8) Main pipeline
# ------------------------------------------------------------

def run_lgd_pipeline(
    data_path: str | None = None,
    use_loader: bool = True,
):
    """
    Chạy full pipeline LGD:
      1) Load dữ liệu
      2) Preprocess
      3) Tính LGD cho RW12/18/24
      4) Build lookup_all + LGD_BASE
      5) Build LGD_scenario theo GDP
      6) Export các file Excel vào OUT_ROOT / 'LGD'
    """

    # 1) Load data
    if use_loader:
        # load_data có thể tự hiểu path (SQL / parquet / v.v.)
        if data_path is None:
            df = load_data()
        else:
            df = load_data(data_path)
    else:
        if data_path is None:
            raise ValueError("Nếu use_loader=False thì phải truyền data_path (parquet/csv...).")
        df = pd.read_parquet(data_path)

    print(f"📂 Loaded {len(df):,} rows for LGD.")

    # 2) Preprocess
    df_prep = preprocess_lgd_raw(df)
    print(f"✅ After preprocess: {len(df_prep):,} rows.")

    # 3–4) LGD lookup & LGD_BASE
    loan12, lookup12, loan18, lookup18, loan24, lookup24, lookup_all = build_lgd_lookup_all(df_prep)

    if lookup_all.empty:
        print("⚠️ lookup_all rỗng → dừng pipeline LGD.")
        return {
            "df_raw": df,
            "df_prep": df_prep,
            "lookup12": lookup12,
            "lookup18": lookup18,
            "lookup24": lookup24,
            "lookup_all": lookup_all,
            "lgd_base": pd.DataFrame(),
            "LGD_scenario": pd.DataFrame(),
        }

    lgd_base = lookup_all[["PRODUCT_SEGMENT", "MOB_BUCKET", "LGD_BASE"]].copy()

    # 5) LGD_scenario theo GDP
    LGD_scenario = build_lgd_scenario(lgd_base)

    # 6) Export Excel
    out_dir = OUT_ROOT / "LGD"
    out_dir.mkdir(parents=True, exist_ok=True)

    path_l12 = out_dir / "LGD_lookup_RW12.xlsx"
    path_l18 = out_dir / "LGD_lookup_RW18.xlsx"
    path_l24 = out_dir / "LGD_lookup_RW24.xlsx"
    path_all = out_dir / "LGD_lookup_all.xlsx"
    path_base = out_dir / "LGD_base_final.xlsx"
    path_scen = out_dir / "LGD_scenario_GDP_adjusted.xlsx"

    if not lookup12.empty:
        lookup12.to_excel(path_l12, index=False)
    if not lookup18.empty:
        lookup18.to_excel(path_l18, index=False)
    if not lookup24.empty:
        lookup24.to_excel(path_l24, index=False)

    lookup_all.to_excel(path_all, index=False)
    lgd_base.to_excel(path_base, index=False)
    LGD_scenario.to_excel(path_scen, index=False)

    print("✓ DONE — Created:")
    if not lookup12.empty:
        print(f"  {path_l12}")
    if not lookup18.empty:
        print(f"  {path_l18}")
    if not lookup24.empty:
        print(f"  {path_l24}")
    print(f"  {path_all}")
    print(f"  {path_base}")
    print(f"  {path_scen}")

    return {
        "df_raw": df,
        "df_prep": df_prep,
        "lookup12": lookup12,
        "lookup18": lookup18,
        "lookup24": lookup24,
        "lookup_all": lookup_all,
        "lgd_base": lgd_base,
        "LGD_scenario": LGD_scenario,
    }


# ------------------------------------------------------------
# 9) Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    # 🔧 chỉnh lại path bên dưới theo môi trường của bạn
    DATA_PATH = r"C:/Users/MAFC4709/Python_work/RR_model_1911/data/parquet/LGD2.parquet"

    results = run_lgd_pipeline(
        data_path=DATA_PATH,
        use_loader=True,   # True: dùng load_data(DATA_PATH), False: đọc parquet trực tiếp
    )
