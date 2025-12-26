# ============================================================
#  forecast.py – Full Forecast Engine (BUCKETS_CANON, EAD-based)
# ============================================================

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Tuple

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

from src.config import CFG, BUCKETS_CANON


# ============================================================
# 0️⃣ Macro Adjustment (optional)
# ============================================================

def apply_macro_adjustment(P: pd.DataFrame,
                           macro_params: dict | None = None,
                           enable_macro: bool = False) -> pd.DataFrame:
    """
    Macro / stress layer (hiện tại chưa dùng, chỉ passthrough).
    P là ma trận transition theo BUCKETS_CANON.
    """
    if not enable_macro:
        return P
    # TODO: implement macro adjustments later
    return P


# ============================================================
# 1️⃣ Helper: Initial EAD vector theo state
# ============================================================

def get_initial_ead_vector(df_latest: pd.DataFrame) -> pd.Series:
    """
    Tính vector EAD ban đầu theo từng state (BUCKETS_CANON), dùng EAD thực tế.
    df_latest: dữ liệu của 1 vintage tại cutoff mới nhất (lọc sẵn theo
               PRODUCT_TYPE, RISK_SCORE, DISBURSAL_DATE, CUTOFF_DATE max).

    Output:
        Series index = BUCKETS_CANON
               value = tổng EAD (CFG["ead"]) ở state đó
    """
    state_col = CFG["state"]
    ead_col   = CFG["ead"]

    # Tổng EAD theo state
    ead_by_state = (
        df_latest
        .groupby(state_col, observed=True)[ead_col]
        .sum()
    )

    # Đảm bảo đầy đủ BUCKETS_CANON, thiếu state nào thì = 0
    ead_vec = ead_by_state.reindex(BUCKETS_CANON, fill_value=0.0)

    return ead_vec


# ============================================================
# 2️⃣ FORECAST SEGMENT (EAD vector × Transition Matrix)
# ============================================================

def forecast_segment(
    matrices_by_mob: Dict,
    product: str,
    score: str,
    start_mob: int,
    initial_ead: pd.Series,
    max_mob: int,
    enable_macro: bool = False,
    macro_params: dict | None = None,
):
    """
    Forecast EAD theo state từ start_mob → max_mob, sử dụng transition matrices.

    Ý nghĩa:
      - initial_ead: vector EAD theo BUCKETS_CANON tại start_mob
      - P: transition matrix amount-weighted (đã build từ ead_t trong transition module)
      - EAD_{t+1} = EAD_t @ P

    Return:
        result: dict
            key   = MOB
            value = Series (index=BUCKETS_CANON, value = EAD theo state)
    """

    result: Dict[int, pd.Series] = {}

    # Khởi tạo EAD tại MOB hiện tại
    cur_ead = initial_ead.astype(float).copy()
    result[start_mob] = cur_ead.copy()

    for mob in range(start_mob, max_mob):

        # Lấy matrix đúng MOB; nếu không có thì fallback MOB lớn nhất
        if (
            product in matrices_by_mob
            and mob in matrices_by_mob[product]
            and score in matrices_by_mob[product][mob]
        ):
            P: pd.DataFrame = matrices_by_mob[product][mob][score]["P"]
        else:
            last_mob = max(matrices_by_mob[product].keys())
            P = matrices_by_mob[product][last_mob][score]["P"]

        # Macro layer (nếu có)
        P_adj = apply_macro_adjustment(P, macro_params, enable_macro)

        # cur_ead: row vector (1 × n), P_adj: (n × n) → EAD_{t+1}
        cur_ead_values = cur_ead.values @ P_adj.values

        cur_ead = pd.Series(cur_ead_values, index=BUCKETS_CANON)

        # Lưu EAD theo state tại MOB+1
        result[mob + 1] = cur_ead.copy()

    return result


# ============================================================
# 3️⃣ FORECAST CHO 1 VINTAGE (EAD-based)
# ============================================================

# ============================================================
# 3️⃣ FORECAST CHO 1 VINTAGE (EAD-based) – BẢN TỐI ƯU
# ============================================================

def forecast_vintage(
    df_raw: pd.DataFrame,
    matrices_by_mob: Dict,
    product: str,
    score: str,
    vintage_date,
    max_mob: int = 29,
    enable_macro: bool = False,
    macro_params: dict | None = None,
    df_vintage: pd.DataFrame | None = None,  # ✅ NEW: cho phép truyền sẵn subset
):
    """
    Forecast cho 1 vintage (cohort):
      - Vintage xác định bằng DISBURSAL_DATE (CFG["orig_date"])
      - Dùng dữ liệu thực tế tại cutoff mới nhất để:
          + xác định MOB hiện tại
          + xây initial EAD vector theo state (EAD-weighted)
      - Sau đó EAD được forecast bằng:
          EAD_{t+1} = EAD_t @ P

    Input:
        - df_raw: full dataset
        - df_vintage (optional): nếu đã lọc sẵn (product, score, vintage)
          thì truyền vào để tránh lọc lại (tối ưu performance)

    Output:
        dict {mob: Series(EAD theo state)}
    """

    orig_col   = CFG["orig_date"]   # DISBURSAL_DATE
    cutoff_col = CFG["cutoff"]
    mob_col    = CFG["mob"]

    # ✅ Nếu caller chưa truyền sẵn df_vintage thì vẫn giữ logic cũ
    if df_vintage is None:
        df_vintage = df_raw[
            (df_raw["PRODUCT_TYPE"] == product)
            & (df_raw["RISK_SCORE"] == score)
            & (df_raw[orig_col] == vintage_date)
        ]

    if df_vintage.empty:
        raise ValueError(
            f"No loans found for vintage {vintage_date} "
            f"(product={product}, score={score})"
        )

    # Snapshot gần nhất
    latest_cutoff = df_vintage[cutoff_col].max()
    df_latest = df_vintage[df_vintage[cutoff_col] == latest_cutoff]

    if df_latest.empty:
        raise ValueError(
            f"No rows at latest cutoff for vintage={vintage_date}, "
            f"product={product}, score={score}"
        )

    # MOB hiện tại của vintage
    start_mob = int(df_latest[mob_col].max())

    # ⭐ Initial EAD vector theo state (amount-based)
    init_ead = get_initial_ead_vector(df_latest)

    # Run forecast chain trên EAD
    fc = forecast_segment(
        matrices_by_mob=matrices_by_mob,
        product=product,
        score=score,
        start_mob=start_mob,
        initial_ead=init_ead,
        max_mob=max_mob,
        enable_macro=enable_macro,
        macro_params=macro_params,
    )

    return fc


# ============================================================
# 4️⃣ FORECAST FULL PORTFOLIO (EAD-based) – BẢN TỐI ƯU
# ============================================================

def forecast_all_vintages(
    df_raw: pd.DataFrame,
    matrices_by_mob: Dict,
    max_mob: int = 29,
    enable_macro: bool = False,
    macro_params: dict | None = None,
):
    """
    Forecast EAD cho TẤT CẢ SEGMENTS:
        - mọi Product trong matrices_by_mob
        - mọi Score trong matrices_by_mob
        - mọi Vintage (DISBURSAL_DATE) thật sự có trong df_raw

    ✅ BẢN TỐI ƯU:
      - Group sẵn df_raw theo (PRODUCT_TYPE, RISK_SCORE, orig_date)
      - Dùng index group để lấy subset thay vì lọc boolean nhiều lần
      - Chỉ loop qua những (product, score) thật sự có data
      - Giữ nguyên output:
          results[(product, score, vintage_date)] = {mob: Series(EAD)}
    """

    from collections import defaultdict

    orig_col = CFG["orig_date"]
    results: Dict[Tuple[str, str, object], Dict[int, pd.Series]] = {}

    # ---------------------------------------------
    # 1️⃣ Group sẵn dataset theo (product, score, vintage)
    # ---------------------------------------------
    group_cols = ["PRODUCT_TYPE", "RISK_SCORE", orig_col]

    grouped = df_raw.groupby(
        group_cols,
        observed=True,
        sort=False,
    )

    # groups: dict key -> index (Index of rows)
    group_index = grouped.groups  # type: Dict[Tuple, pd.Index]

    # Map product -> set(vintage), product -> set(score)
    vintages_by_product: Dict[str, set] = defaultdict(set)
    scores_by_product: Dict[str, set] = defaultdict(set)

    for (p, s, v) in group_index.keys():
        vintages_by_product[p].add(v)
        scores_by_product[p].add(s)

    # ---------------------------------------------
    # 2️⃣ Loop qua các product có matrix
    #    và match với các score/vintage có data
    # ---------------------------------------------
    for product, mob_dict in matrices_by_mob.items():

        # Nếu product không có data trong df_raw → bỏ qua
        if product not in vintages_by_product:
            continue

        # Score list bên phía matrices
        sample_mob = next(iter(mob_dict.keys()))
        scores_in_matrices = set(mob_dict[sample_mob].keys())

        # Score list thực tế từ data
        scores_in_data = scores_by_product.get(product, set())

        # Chỉ giữ những score vừa có matrix vừa có data
        score_list = sorted(scores_in_matrices & scores_in_data)
        if not score_list:
            continue

        # Các vintages thực sự có trong data cho product này
        vintages = sorted(vintages_by_product[product])

        for score in score_list:
            for v in vintages:

                key = (product, score, v)
                idx = group_index.get(key)

                # Không có data cho (product, score, vintage) này → skip
                if idx is None:
                    continue

                # Lấy subset nhanh bằng index thay vì lọc boolean
                df_seg = df_raw.loc[idx]

                try:
                    # Forecast cho 1 segment duy nhất
                    fc = forecast_vintage(
                        df_raw=df_raw,
                        matrices_by_mob=matrices_by_mob,
                        product=product,
                        score=score,
                        vintage_date=v,
                        max_mob=max_mob,
                        enable_macro=enable_macro,
                        macro_params=macro_params,
                        df_vintage=df_seg,   # ✅ Truyền subset đã lọc sẵn
                    )

                    results[(product, score, v)] = fc

                except Exception as e:
                    # Giữ nguyên behavior: log và skip segment lỗi
                    print(f"⚠️ Skip ({product}, {score}, vintage={v}) due to error:")
                    print("   ", e)

    return results


# ============================================================
# 5️⃣ Helper: Convert forecast → DataFrame
# ============================================================

def forecast_to_dataframe(forecast_dict: Dict[int, pd.Series]) -> pd.DataFrame:
    """
    Convert 1 forecast segment (dict {mob: Series(EAD)}) → DataFrame:
        index  = MOB
        columns= BUCKETS_CANON (EAD theo state)
    """
    df = pd.DataFrame(forecast_dict).T
    df.index.name = "MOB"
    return df


# ============================================================
# 6️⃣ Helpers: Convert EAD → tỷ lệ (distribution, PD)
# ============================================================

def forecast_to_distribution(forecast_dict: Dict[int, pd.Series]) -> Dict[int, pd.Series]:
    """
    Convert forecast EAD theo state → distribution (% trên tổng EAD ban đầu).

    Giữ nguyên tổng EAD ban đầu (MOB nhỏ nhất) làm mẫu số.
    """
    if not forecast_dict:
        return {}

    # Lấy MOB đầu tiên (start_mob)
    first_mob = min(forecast_dict.keys())
    total_ead0 = forecast_dict[first_mob].sum()

    dist_result: Dict[int, pd.Series] = {}
    for mob, ead_vec in forecast_dict.items():
        dist = ead_vec / total_ead0
        dist_result[mob] = dist

    return dist_result


def marginal_loss_rate(
    forecast_dict: Dict[int, pd.Series],
    default_state: str = "WRITEOFF",
) -> pd.Series:
    """
    Marginal loss rate theo MOB (trên tổng EAD ban đầu),
    dựa trên EAD nằm ở default_state (thường là WRITEOFF).
    """
    if not forecast_dict:
        return pd.Series(dtype=float)

    first_mob = min(forecast_dict.keys())
    total_ead0 = forecast_dict[first_mob].sum()

    loss_series = {
        mob: (dist[default_state] / total_ead0)
        for mob, dist in forecast_dict.items()
    }

    return pd.Series(loss_series)


def cumulative_loss_rate(
    forecast_dict: Dict[int, pd.Series],
    default_state: str = "WRITEOFF",
) -> pd.Series:
    """
    Cumulative loss rate (trên EAD ban đầu) theo MOB,
    dựa trên EAD ở default_state.
    """
    m = marginal_loss_rate(forecast_dict, default_state=default_state)
    return m.cumsum().clip(upper=1.0)


# ============================================================
# 7️⃣ Vintage Table – DPD30+ aggregate trên EAD
# ============================================================

def vintage_table_ead(forecast_dict: Dict[int, pd.Series]) -> pd.DataFrame:
    """
    Bảng theo MOB, mỗi cell là EAD theo state.
    Thêm cột DPD30+_AGG_EAD = tổng EAD ở các bucket 30+ (30–180).
    """
    df = forecast_to_dataframe(forecast_dict)
    dpd_cols = [c for c in df.columns if c in ["DPD30+", "DPD60+", "DPD90+", "DPD120+", "DPD180+"]]
    if dpd_cols:
        df["DPD30+_AGG_EAD"] = df[dpd_cols].sum(axis=1)
    return df


# ============================================================
# 8️⃣ Plot Helper (vẽ theo EAD hoặc tỷ lệ tuỳ bạn dùng input)
# ============================================================

def plot_curve(
    forecast_df: pd.DataFrame,
    columns,
    title: str | None = None,
):
    """
    V? curve theo MOB cho 1 ho?c nhi?u c?t (EAD ho?c t? l?).
    """
    if not _HAS_MPL:
        print("matplotlib is not available; skipping plot.")
        return
    plt.figure(figsize=(10, 5))
    for col in columns:
        if col not in forecast_df.columns:
            print(f"?? Column missing: {col}")
            continue
        plt.plot(forecast_df.index, forecast_df[col], label=col)

    plt.xlabel("MOB")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(title or "Forecast Curve")
    plt.tight_layout()
    plt.show()
def validate_matrices(matrices_by_mob, atol: float = 1e-6):
    """
    Kiểm tra:
      - Thứ tự state trong matrix có khớp BUCKETS_CANON không
      - Mỗi hàng có sum ≈ 1 (row-stochastic)
    """
    issues = []
    for product, mob_dict in matrices_by_mob.items():
        for mob, score_dict in mob_dict.items():
            for score, entry in score_dict.items():
                P = entry["P"]

                # Check state order
                if list(P.index) != BUCKETS_CANON or list(P.columns) != BUCKETS_CANON:
                    issues.append((product, mob, score, "State mismatch"))
                    continue

                # Check normalization
                row_sums = P.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=atol):
                    issues.append(
                        (product, mob, score,
                         f"Row sum error (min={row_sums.min()}, max={row_sums.max()})")
                    )

    return issues

def forecast_full_history(
    df_raw: pd.DataFrame,
    matrices_by_mob: Dict,
    max_mob: int = 36,
):
    """
    Backtest forecast full-history:
        forecast từ MOB0 → max_mob
    cho mọi (product, score, vintage) đã có trong df_raw.

    Output: DataFrame long-format
    """

    orig_col = CFG["orig_date"]
    records = []

    grouped = df_raw.groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col])

    for (prod, score, vintage), df_seg in grouped:

        try:
            fc_dict = forecast_vintage(
                df_raw=df_raw,
                matrices_by_mob=matrices_by_mob,
                product=prod,
                score=score,
                vintage_date=vintage,
                max_mob=max_mob,
                enable_macro=False,
                df_vintage=df_seg,   # subset để chạy nhanh
            )

            for mob, series in fc_dict.items():
                rec = {
                    "PRODUCT_TYPE": prod,
                    "RISK_SCORE": score,
                    "VINTAGE_DATE": vintage,
                    "MOB": mob,
                }
                rec.update(series.to_dict())
                records.append(rec)

        except Exception as e:
            print(f"⚠️ Skip ({prod}, {score}, {vintage}) → {e}")

    df = pd.DataFrame(records)
    df = df.sort_values(
        ["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"]
    ).reset_index(drop=True)

    return df
