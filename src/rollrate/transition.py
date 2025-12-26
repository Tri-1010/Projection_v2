# ============================================================
#  transition.py – Bản HOÀN CHỈNH (WHA + EAD + Product×MOB×Score)
# ============================================================

from __future__ import annotations

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple

from src.config import (
    CFG,             # mapping tên cột: loan, mob, state, cutoff, ead, ...
    BUCKETS_CANON,   # ["DPD0","DPD1+","DPD30+","DPD60+","DPD90+","DPD120+","DPD180+", ...]
    ABSORBING_BASE,  # ["PREPAY","WRITEOFF","SOLDOUT"]
    MIN_OBS,         # ngưỡng tối thiểu số cặp
    MIN_EAD,         # ngưỡng tối thiểu EAD (tổng weight)
)

# Optional alpha smoothing (không dùng nếu =0)
try:
    from src.config import ALPHA_SMOOTH
except Exception:
    ALPHA_SMOOTH = 0.0

# ===========================
#  State-space đầy đủ
# ===========================
STATE_SPACE = list(dict.fromkeys(list(BUCKETS_CANON) + list(ABSORBING_BASE)))


# ============================================================
# Helper utilities
# ============================================================

def _safe_int_series(x: pd.Series) -> pd.Series:
    """
    Ép về Int64 an toàn, xử lý cả NaN + số thực.
    """
    x2 = pd.to_numeric(x, errors="coerce")
    x2 = x2.round(0)
    return x2.astype("Int64")


def _normalize_rows(mat: pd.DataFrame) -> pd.DataFrame:
    s = mat.sum(axis=1).replace(0, np.nan)
    return mat.div(s, axis=0).fillna(0.0)


def _warn_unknown_states(states_seen, allowed=None):
    """
    Cảnh báo nếu có state không nằm trong STATE_SPACE.
    """
    if allowed is None:
        allowed = STATE_SPACE
    cleaned = {s for s in states_seen if pd.notna(s)}
    unknown = set(cleaned).difference(set(allowed))
    if unknown:
        print(f"⚠️ Phát hiện state ngoài STATE_SPACE: {sorted(map(str, unknown))} → sẽ bị reindex về 0.")


def _backfill_zero_rows(
    P: pd.DataFrame,
    counts: pd.DataFrame,
    fallback_P: pd.DataFrame | None,
    policy: str = "parent",
) -> pd.DataFrame:
    """
    Chỉ xử lý các hàng có tổng weight = 0.
    - parent: copy hàng từ fallback_P (nếu có).
    - uniform: chia đều toàn hàng.
    - identity: tự hút vào chính nó.
    """
    zero_mask = (counts.sum(axis=1) == 0)
    if not zero_mask.any():
        return P

    zero_states = P.index[zero_mask].tolist()
    print(f"⚠️ Có {len(zero_states)} hàng có tổng weight = 0: {zero_states}")

    k = P.shape[1]

    for st in zero_states:
        if policy == "parent" and fallback_P is not None and st in fallback_P.index:
            P.loc[st] = fallback_P.loc[st].values
        elif policy == "uniform":
            P.loc[st] = 1.0 / k
        else:  # identity
            P.loc[st] = 0.0
            if st in P.columns:
                P.loc[st, st] = 1.0
    return P


def _enforce_absorbing(P: pd.DataFrame, absorbing) -> pd.DataFrame:
    """
    Ép các absorbing state có probability ở lại = 100%.
    """
    for st in absorbing:
        if st in P.index and st in P.columns:
            P.loc[st, :] = 0.0
            P.loc[st, st] = 1.0
    return P



def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
    loan   = CFG["loan"]
    mob    = CFG["mob"]
    state  = CFG["state"]
    cutoff = CFG["cutoff"]
    eadcol = CFG.get("ead")

    ROLL_WINDOW   = CFG.get("ROLL_WINDOW", 12)
    WEIGHT_METHOD = CFG.get("WEIGHT_METHOD", "exp")
    DECAY_LAMBDA  = CFG.get("DECAY_LAMBDA", 0.9)

    product_col = "PRODUCT_TYPE" if "PRODUCT_TYPE" in df.columns else None
    score_col   = "RISK_SCORE"   if "RISK_SCORE" in df.columns else None

    # Chuẩn hoá
    work = df.copy()
    work[mob]    = _safe_int_series(work[mob])
    work[cutoff] = pd.to_datetime(work[cutoff], errors="coerce")
    work = work.dropna(subset=[mob, state, cutoff])
    work = work.sort_values([loan, mob])

    # ====== WHA PREP ======
    month_idx = work[cutoff].dt.to_period("M").astype(int)
    max_idx   = month_idx.max()
    age_raw   = max_idx - month_idx

    # ====== ❗ LỌC THEO ROLL_WINDOW ======
    mask = age_raw <= ROLL_WINDOW
    work = work[mask].copy()
    age  = age_raw[mask].clip(0, ROLL_WINDOW)

    # ====== TIME WEIGHT ======
    if WEIGHT_METHOD == "exp":
        time_w = (DECAY_LAMBDA ** age).astype(float)
    elif WEIGHT_METHOD == "linear":
        time_w = (1 - age/ROLL_WINDOW).clip(0)
    else:
        time_w = pd.Series(1.0, index=work.index)

    # EAD WEIGHT
    if eadcol and eadcol in work.columns:
        ead_raw = pd.to_numeric(work[eadcol], errors="coerce").fillna(0.0)
    else:
        ead_raw = pd.Series(1.0, index=work.index)

    work["ead_raw"] = ead_raw
    work["time_weight"] = time_w
    work["ead_t"] = ead_raw * time_w

    # ====== TẠO PAIRS ======
    mob_next = work.groupby(loan)[mob].shift(-1)
    st_t     = work.groupby(loan)[state].shift(0)
    st_t1    = work.groupby(loan)[state].shift(-1)

    valid = (mob_next - work[mob] == 1)
    pairs = work[valid].copy()

    if pairs.empty:
        print("⚠️ make_pairs(): Không có cặp hợp lệ.")
        return pd.DataFrame()

    pairs["mob_t"] = work.loc[pairs.index, mob].astype(int)
    pairs["mob_t1"] = mob_next.loc[pairs.index].astype(int)
    pairs["state_t"] = st_t.loc[pairs.index]
    pairs["state_t1"] = st_t1.loc[pairs.index]

    pairs["product_t"] = work[product_col].loc[pairs.index] if product_col else "ALL"
    pairs["score_t"]   = work[score_col].loc[pairs.index]   if score_col   else "ALL"

    return pairs[
        [loan, "mob_t", "mob_t1", "state_t", "state_t1",
         "ead_raw", "time_weight", "ead_t", "product_t", "score_t"]
    ]


# ============================================================
# 2️⃣ compute_transition_from_pairs – ma trận từ 1 lát cắt pairs
# ============================================================

def compute_transition_from_pairs(
    pair_slice: pd.DataFrame,
    value_col: str = "ead_t",
    parent_P: pd.DataFrame | None = None,
    zero_row_policy: str = "parent",
    alpha_smooth: float | None = None,
) -> pd.DataFrame:
    """
    Tính ma trận transition từ 1 lát cắt pairs (vd: 1 product, 1 score, 1 MOB hoặc all MOB).
    - value_col: cột dùng làm trọng số (mặc định ead_t đã WHA).
    - parent_P: dùng để backfill các hàng không có quan sát.
    """
    if pair_slice.empty:
        return pd.DataFrame(0.0, index=STATE_SPACE, columns=STATE_SPACE)

    if value_col not in pair_slice.columns:
        raise KeyError(f"compute_transition_from_pairs(): thiếu cột '{value_col}' trong pair_slice.")

    # Trọng số
    weights = pd.to_numeric(pair_slice[value_col], errors="coerce").fillna(0.0)

    # Ma trận đếm/trọng số
    mat_counts = (
        pd.crosstab(
            pair_slice["state_t"],
            pair_slice["state_t1"],
            values=weights,
            aggfunc="sum",
            dropna=False,
        )
        .reindex(index=STATE_SPACE, columns=STATE_SPACE, fill_value=0.0)
    )

    # Chuẩn hoá theo hàng
    P = _normalize_rows(mat_counts)

    # Chỉ backfill hàng có tổng weight = 0
    P = _backfill_zero_rows(P, counts=mat_counts, fallback_P=parent_P, policy=zero_row_policy)

    # Ép absorbing
    P = _enforce_absorbing(P, ABSORBING_BASE)

    # Check tổng hàng
    bad = ~np.isclose(P.sum(axis=1).values, 1.0, atol=1e-6)
    if bad.any():
        bad_states = [P.index[i] for i, b in enumerate(bad) if b]
        print(f"⚠️ Một số hàng không cộng về 1 sau backfill: {bad_states} → normalize lại.")
        P = _normalize_rows(P)

    # Optional smoothing (chỉ nếu ALPHA_SMOOTH > 0 và hàng = 0)
    _alpha = ALPHA_SMOOTH if alpha_smooth is None else alpha_smooth
    if _alpha and _alpha > 0:
        zero_rows = P.index[P.sum(axis=1) == 0].tolist()
        if zero_rows:
            k = len(P.columns)
            for st in zero_rows:
                P.loc[st] = 1.0 / k
            P = _normalize_rows(P)

    return P


# ============================================================
# 3️⃣ compute_transition_by_mob – Product × MOB × Score
# ============================================================

def compute_transition_by_mob(
    df: pd.DataFrame,
) -> Tuple[Dict[str, Dict[int, Dict[str, Dict[str, pd.DataFrame]]]],
           Dict[tuple, pd.DataFrame]]:
    """
    Xây dựng:
        matrices_by_mob[product][mob][score] = {
            "P": ma trận transition,
            "is_fallback": bool,
            "reason": str
        }

        parent_fallback[(product, score)] = ma trận parent (không phân tách MOB).

    Logic:
      - Bước 1: make_pairs(df) → pairs.
      - Bước 2: group theo (product_t, score_t) → tính P_parent cho từng nhóm.
      - Bước 3: trong mỗi (product, score), group tiếp theo mob_t:
          + nếu n_obs < MIN_OBS hoặc EAD < MIN_EAD → dùng parent (is_fallback=True).
          + nếu tính P_child ra ma trận toàn 0 → dùng parent.
    """
    pairs = make_pairs(df)
    if pairs.empty:
        print("⚠️ compute_transition_by_mob(): không có cặp → trả về rỗng.")
        return {}, {}

    matrices_by_mob: Dict[str, Dict[int, Dict[str, Dict[str, pd.DataFrame]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    parent_fallback: Dict[tuple, pd.DataFrame] = {}

    # ===========================
    # 1) Parent FALLBACK theo (product, score)
    # ===========================
    for (prod, score), grp in pairs.groupby(["product_t", "score_t"]):
        prod_str  = str(prod)
        score_str = str(score)

        P_parent = compute_transition_from_pairs(
            grp,
            value_col="ead_t",
            parent_P=None,
            zero_row_policy="parent",
            alpha_smooth=0.0,
        )
        parent_fallback[(prod_str, score_str)] = P_parent
        print(f"⚙️ Built parent fallback for (product={prod_str}, score={score_str})")

    # ===========================
    # 2) MOB-level trong từng (product, score)
    # ===========================
    for (prod, score), grp in pairs.groupby(["product_t", "score_t"]):
        prod_str  = str(prod)
        score_str = str(score)
        parent_P  = parent_fallback[(prod_str, score_str)]

        for mob in sorted(grp["mob_t"].unique()):
            mob_int = int(mob)
            mob_grp = grp[grp["mob_t"] == mob_int]

            total_ead = mob_grp["ead_t"].sum()
            n_obs     = len(mob_grp)

            # Kiểm tra sufficiency
            if n_obs < MIN_OBS or total_ead < MIN_EAD:
                reason = f"insufficient data (n_obs={n_obs}, total_ead={total_ead:,.0f})"
                print(f"⚠️ Fallback MOB={mob_int} for (product={prod_str}, score={score_str}) → {reason}")
                matrices_by_mob[prod_str][mob_int][score_str] = {
                    "P": parent_P,
                    "is_fallback": True,
                    "reason": reason,
                }
                continue

            P_child = compute_transition_from_pairs(
                mob_grp,
                value_col="ead_t",
                parent_P=parent_P,
                zero_row_policy="parent",
                alpha_smooth=0.0,
            )

            if (P_child.sum(axis=1) == 0).all():
                reason = "no valid pairs at MOB-level (all rows = 0)"
                print(f"⚠️ MOB={mob_int} (product={prod_str}, score={score_str}) → {reason}, dùng parent.")
                matrices_by_mob[prod_str][mob_int][score_str] = {
                    "P": parent_P,
                    "is_fallback": True,
                    "reason": reason,
                }
            else:
                matrices_by_mob[prod_str][mob_int][score_str] = {
                    "P": P_child,
                    "is_fallback": False,
                    "reason": "",
                }

    total_blocks = sum(len(mob_dict) for mob_dict in matrices_by_mob.values())
    print(f"✅ Generated {total_blocks} MOB-level matrices across products (real + fallback).")
    return matrices_by_mob, parent_fallback
