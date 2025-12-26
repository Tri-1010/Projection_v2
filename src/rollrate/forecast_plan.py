import pandas as pd
import numpy as np
from typing import Dict, List

from src.config import BUCKETS_CANON, ABSORBING_BASE
from src.rollrate.transition import STATE_SPACE


def forecast_sale_plan_by_mob(
    sale_plan_df: pd.DataFrame,
    matrices_by_mob: Dict[str, Dict[int, Dict[str, Dict[str, pd.DataFrame]]]],
    parent_fallback: Dict[tuple, pd.DataFrame],
    mob_target: int = 12,
    start_state: str | None = None,
    states: List[str] | None = None,
) -> pd.DataFrame:
    """
    Forecast lifecycle cho các future vintages dựa trên Sale Plan,
    dùng đúng output của compute_transition_by_mob(df):

        matrices_by_mob[product][mob][score] = { "P": DataFrame, ... }
        parent_fallback[(product, score)]    = P_parent (DataFrame)

    sale_plan_df cần có:
        - PRODUCT_TYPE
        - RISK_SCORE
        - VINTAGE_DATE
        - EAD_PLAN
    """

    # -----------------------------
    # Chuẩn tham số
    # -----------------------------
    if states is None:
        states = list(STATE_SPACE)

    if start_state is None:
        start_state = BUCKETS_CANON[0]  # vd: "DPD0"

    if start_state not in states:
        raise ValueError(f"start_state='{start_state}' không nằm trong STATE_SPACE.")

    state_idx = {s: i for i, s in enumerate(states)}

    # Bảo đảm datetime
    sale_plan_df = sale_plan_df.copy()
    if "VINTAGE_DATE" in sale_plan_df.columns:
        sale_plan_df["VINTAGE_DATE"] = pd.to_datetime(sale_plan_df["VINTAGE_DATE"])

    records = []

    # -----------------------------
    # Helper: chọn ma trận P cho (product, score, mob)
    # -----------------------------
    def _get_P(product: str, score: str, mob: int) -> pd.DataFrame:
        prod_str = str(product)
        score_str = str(score)

        # 1) Thử lấy đúng theo (product, mob, score)
        mob_dict = matrices_by_mob.get(prod_str, {})
        score_block = mob_dict.get(mob)

        if score_block is not None and score_str in score_block:
            P_df = score_block[score_str]["P"]
            return P_df.reindex(index=states, columns=states, fill_value=0.0)

        # 2) Fallback parent_fallback exact (product, score)
        key_exact = (prod_str, score_str)
        if key_exact in parent_fallback:
            P_df = parent_fallback[key_exact]
            return P_df.reindex(index=states, columns=states, fill_value=0.0)

        # 3) Fallback bất kỳ score có thật của product đó
        candidate_keys = [k for k in parent_fallback.keys() if k[0] == prod_str]
        if candidate_keys:
            use_key = candidate_keys[0]
            print(
                f"⚠️ Fallback (product={prod_str}, score={score_str}, mob={mob}) "
                f"→ dùng parent_fallback với score='{use_key[1]}'"
            )
            P_df = parent_fallback[use_key]
            return P_df.reindex(index=states, columns=states, fill_value=0.0)

        # 4) Không có bất cứ ma trận nào cho product → dùng identity (giữ nguyên state)
        print(
            f"⚠️ Không tìm thấy bất kỳ parent_fallback nào cho product='{prod_str}'. "
            f"Dùng identity matrix cho MOB={mob}."
        )
        eye = np.eye(len(states))
        P_df = pd.DataFrame(eye, index=states, columns=states)
        return P_df

    # -----------------------------
    # LOOP TỪNG DÒNG SALE PLAN
    # -----------------------------
    for _, row in sale_plan_df.iterrows():

        product = str(row["PRODUCT_TYPE"])
        score   = str(row["RISK_SCORE"])  # đã normalize ngoài notebook
        vintage = row["VINTAGE_DATE"]
        ead0    = float(row["EAD_PLAN"])

        # Khởi tạo vector EAD tại MOB 0
        ead_vec = np.zeros(len(states))
        ead_vec[state_idx[start_state]] = ead0

        # Chạy từ MOB 0 → MOB_target
        for mob in range(mob_target + 1):

            rec = {
                "PRODUCT_TYPE": product,
                "RISK_SCORE": score,
                "VINTAGE_DATE": vintage,
                "MOB": mob,
            }
            for i, st in enumerate(states):
                rec[st] = ead_vec[i]
            records.append(rec)

            if mob == mob_target:
                break

            P_df = _get_P(product, score, mob)
            P = P_df.values

            ead_vec = ead_vec @ P

    result = pd.DataFrame(records).sort_values(
        by=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE", "MOB"]
    ).reset_index(drop=True)

    return result
