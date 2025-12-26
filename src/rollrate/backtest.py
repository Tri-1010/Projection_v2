import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import CFG, BUCKETS_CANON

# 1️⃣ Compute actual transition matrix
def compute_actual_matrix(df: pd.DataFrame, cutoff_start: str, cutoff_end: str) -> pd.DataFrame:
    """
    Tính ma trận chuyển trạng thái thực tế giữa 2 cutoff date liên tiếp.

    Args:
        df: Dữ liệu lịch sử có cột CFG["cutoff"], CFG["loan"], CFG["state"]
        cutoff_start: kỳ bắt đầu (vd '2023-06-30')
        cutoff_end: kỳ kết thúc (vd '2023-07-31')

    Returns:
        DataFrame ma trận [from_state x to_state]
    """
    cutoff_col, loan_col, state_col = CFG["cutoff"], CFG["loan"], CFG["state"]

    # Chọn dữ liệu 2 kỳ
    df_start = df[df[cutoff_col] == cutoff_start]
    df_end   = df[df[cutoff_col] == cutoff_end]

    if df_start.empty or df_end.empty:
        raise ValueError(f"Không tìm thấy cutoff {cutoff_start} hoặc {cutoff_end} trong dữ liệu.")

    # Join theo loan_id
    merged = pd.merge(
        df_start[[loan_col, state_col]],
        df_end[[loan_col, state_col]],
        on=loan_col,
        suffixes=("_t0", "_t1"),
        how="inner",
    )

    # Pivot sang ma trận chuyển trạng thái
    trans = (
        merged.groupby([f"{state_col}_t0", f"{state_col}_t1"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=BUCKETS_CANON, columns=BUCKETS_CANON, fill_value=0)
    )

    # Chuẩn hóa theo hàng (xác suất)
    trans = trans.div(trans.sum(axis=1), axis=0).fillna(0)
    return trans


# 2️⃣ Độ ổn định giữa hai ma trận (Matrix Stability)
def matrix_stability_score(mat_a: pd.DataFrame, mat_b: pd.DataFrame) -> float:
    """
    Đo mức khác biệt trung bình tuyệt đối giữa 2 ma trận (Markov stability score).
    """
    diff = (mat_a - mat_b).abs().mean().mean()
    return float(diff)


# 3️⃣ Roll-forward validation (Markov forecast vs thực tế)
def rollforward_validation(df: pd.DataFrame, mat_train: pd.DataFrame,
                           start_cutoff: str, horizon: int = 1) -> pd.DataFrame:
    """
    Thực hiện backtest Markov: dự báo phân phối trạng thái sau N tháng và so sánh thực tế.

    Args:
        df: Dữ liệu có cột CUTOFF_DATE
        mat_train: ma trận Markov 1-step
        start_cutoff: cutoff bắt đầu
        horizon: số bước chuyển tiếp (tháng)

    Returns:
        DataFrame gồm Predicted, Actual và Diff per state
    """
    cutoff_col, state_col = CFG["cutoff"], CFG["state"]

    # Cắt dữ liệu gốc
    df_start = df[df[cutoff_col] == start_cutoff]
    if df_start.empty:
        raise ValueError(f"Không tìm thấy cutoff {start_cutoff} trong dữ liệu.")

    # Tính cutoff đích
    start_dt = pd.to_datetime(start_cutoff)
    target_dt = (start_dt + pd.DateOffset(months=horizon))
    target_cutoff = target_dt.strftime("%Y-%m-%d")

    if target_cutoff not in df[cutoff_col].astype(str).unique():
        print(f"⚠️ Không tìm thấy cutoff mục tiêu {target_cutoff} trong dữ liệu.")
        return pd.DataFrame()

    # Vector trạng thái ban đầu (distribution)
    dist0 = (
        df_start[state_col]
        .value_counts(normalize=True)
        .reindex(BUCKETS_CANON, fill_value=0)
        .values
    )

    # N-step Markov projection
    mat_h = np.linalg.matrix_power(mat_train.values, horizon)
    dist_pred = dist0 @ mat_h

    # Vector trạng thái thực tế tại cutoff đích
    dist_actual = (
        df[df[cutoff_col].astype(str) == str(target_cutoff)][state_col]
        .value_counts(normalize=True)
        .reindex(BUCKETS_CANON, fill_value=0)
        .values
    )

    # Sai số
    mae = np.abs(dist_pred - dist_actual).mean()

    res = pd.DataFrame({
        "STATE": BUCKETS_CANON,
        "Predicted": dist_pred,
        "Actual": dist_actual,
        "Diff": dist_pred - dist_actual,
    })
    print(f"📈 Roll-forward horizon={horizon} tháng | MAE={mae:.4f}")
    return res


# 4️⃣ Plot độ khác biệt giữa 2 ma trận
def plot_matrix_diff(mat_a: pd.DataFrame, mat_b: pd.DataFrame, title="Matrix Difference (%)"):
    diff = (mat_b - mat_a).fillna(0)
    plt.figure(figsize=(8,6))
    sns.heatmap(diff * 100, annot=True, fmt=".2f", cmap="RdYlBu", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# 5️⃣ Plot phân phối dự báo vs thực tế
def plot_distribution_compare(res_df: pd.DataFrame, title="Predicted vs Actual"):
    plt.figure(figsize=(8,5))
    x = np.arange(len(res_df))
    plt.bar(x - 0.15, res_df["Predicted"], width=0.3, label="Predicted")
    plt.bar(x + 0.15, res_df["Actual"], width=0.3, label="Actual")
    plt.xticks(x, res_df["STATE"], rotation=45)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
