import pandas as pd
import numpy as np
from typing import Dict

from src.config import CFG, BUCKETS_CANON
from src.rollrate.forecast import forecast_segment


def _get_disb_total_for_cohort(df_seg: pd.DataFrame) -> float:
    """
    DISB_TOTAL cho 1 cohort (PRODUCT_TYPE, RISK_SCORE, VINTAGE_DATE).
    Ưu tiên dùng cột DISB trong CFG; nếu không có thì fallback về EAD tại MOB0.
    """
    disb_col = CFG.get("disb")
    loan_col = CFG.get("loan")
    mob_col  = CFG["mob"]
    ead_col  = CFG["ead"]

    # Trường hợp có DISB
    if disb_col and disb_col in df_seg.columns:
        if loan_col and loan_col in df_seg.columns:
            disb = (
                df_seg.groupby(loan_col)[disb_col]
                .first()
                .sum()
            )
        else:
            disb = df_seg[disb_col].sum()
        return float(disb)

    # Fallback: dùng tổng EAD tại MOB0 làm mẫu số
    if mob_col not in df_seg.columns or ead_col not in df_seg.columns:
        raise KeyError(
            f"Missing mob_col ({mob_col}) hoặc ead_col ({ead_col}) trong df_seg.columns"
        )

    df0 = df_seg[df_seg[mob_col] == 0]
    if df0.empty:
        return 0.0
    return float(df0[ead_col].sum())


def forecast_full_history(
    df_raw: pd.DataFrame,
    matrices_by_mob: Dict,
    max_mob: int = 36,
) -> pd.DataFrame:
    """
    Backtest forecast full-history:
        forecast từ MOB0 → max_mob
    cho mọi (product, score, vintage) đã có trong df_raw.

    Giả định:
        - Tại MOB0, toàn bộ EAD nằm ở bucket BUCKETS_CANON[0] (thường là 'DPD0')
        - Tổng EAD ban đầu = DISB_TOTAL của cohort
    """

    orig_col = CFG["orig_date"]
    if orig_col not in df_raw.columns:
        raise KeyError(
            f"CFG['orig_date'] = {orig_col}, nhưng df_raw không có cột này. "
            f"(df_raw.columns = {list(df_raw.columns)[:20]} ...)"
        )

    records = []

    grouped = df_raw.groupby(["PRODUCT_TYPE", "RISK_SCORE", orig_col])

    print(f"⚙️ forecast_full_history(): có {len(grouped)} cohort để backtest")

    for (prod, score, vintage), df_seg in grouped:
        prod_str  = str(prod)
        score_str = str(score)

        # 1️⃣ Lấy DISB_TOTAL cho cohort này
        try:
            disb_total = _get_disb_total_for_cohort(df_seg)
        except Exception as e:
            print(f"⚠️ Lỗi DISB_TOTAL tại ({prod_str}, {score_str}, {vintage}): {e}")
            continue

        if disb_total <= 0:
            print(f"⚠️ Skip ({prod_str}, {score_str}, {vintage}) vì DISB_TOTAL <= 0")
            continue

        # 2️⃣ Khởi tạo vector EAD tại MOB0: toàn bộ nằm ở DPD0
        if not BUCKETS_CANON:
            raise ValueError("BUCKETS_CANON rỗng – kiểm tra lại src.config")

        init_ead = pd.Series(0.0, index=BUCKETS_CANON, dtype=float)
        init_ead.iloc[0] = disb_total  # BUCKETS_CANON[0] thường là "DPD0"

        # 3️⃣ Chạy chuỗi Markov từ MOB0 → max_mob
        try:
            fc_dict = forecast_segment(
                matrices_by_mob=matrices_by_mob,
                product=prod_str,
                score=score_str,
                start_mob=0,
                initial_ead=init_ead,
                max_mob=max_mob,
                enable_macro=False,
                # ❗ Nếu version cũ không có macro_params, bỏ dòng này:
                macro_params=None,
            )
        except TypeError as e:
            # Trường hợp forecast_segment không nhận macro_params
            if "macro_params" in str(e):
                print("ℹ️ forecast_segment không có tham số macro_params – gọi lại không tham số này.")
                fc_dict = forecast_segment(
                    matrices_by_mob=matrices_by_mob,
                    product=prod_str,
                    score=score_str,
                    start_mob=0,
                    initial_ead=init_ead,
                    max_mob=max_mob,
                    enable_macro=False,
                )
            else:
                print(f"⚠️ Skip ({prod_str}, {score_str}, {vintage}) → {e}")
                continue
        except Exception as e:
            print(f"⚠️ Skip ({prod_str}, {score_str}, {vintage}) → {e}")
            continue

        # 4️⃣ Ghi lại long-format
        for mob, series in fc_dict.items():
            rec = {
                "PRODUCT_TYPE": prod_str,
                "RISK_SCORE": score_str,
                "VINTAGE_DATE": vintage,
                "MOB": mob,
            }
            rec.update(series.to_dict())
            records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️ forecast_full_history(): không tạo được record nào – trả về DataFrame rỗng.")
        return df

    df = df.sort_values(
        ["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"]
    ).reset_index(drop=True)

    print(
        f"✅ forecast_full_history(): output shape = {df.shape}, "
        f"MOB_min={df['MOB'].min()}, MOB_max={df['MOB'].max()}"
    )

    return df
