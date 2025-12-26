# ============================================================
#  transition.py – Bản HOÀN CHỈNH, ĐÃ KIỂM TRA LOGIC
# ============================================================

from __future__ import annotations
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict

from src.config import (
    CFG,
    BUCKETS_CANON,
    ABSORBING_BASE,
    SEGMENT_COLS,
    MIN_OBS,
    MIN_EAD,
)

# ==============================================
# α smoothing an toàn
# ==============================================
try:
    from src.config import ALPHA_SMOOTH
except Exception:
    ALPHA_SMOOTH = 0.0


# ============================================================
# Helper utilities
# ============================================================

def _safe_int_series(x: pd.Series) -> pd.Series:
    x2 = pd.to_numeric(x, errors="coerce")
    x2 = x2.round(0)
    return x2.astype("Int64")


def _normalize_rows(mat: pd.DataFrame) -> pd.DataFrame:
    s = mat.sum(axis=1).replace(0, np.nan)
    return mat.div(s, axis=0).fillna(0.0)


def _warn_unknown_states(states_seen, allowed=BUCKETS_CANON):
    cleaned = {s for s in states_seen if pd.notna(s)}
    unknown = set(cleaned).difference(set(allowed))
    if unknown:
        print(f"⚠️ Phát hiện state ngoài BUCKETS_CANON: {sorted(map(str, unknown))} → sẽ bị reindex về 0.")


def _backfill_zero_rows(P: pd.DataFrame, counts: pd.DataFrame,
                        fallback_P: pd.DataFrame | None,
                        policy="parent") -> pd.DataFrame:

    zero_mask = (counts.sum(axis=1) == 0)
    if not zero_mask.any():
        return P

    zero_states = P.index[zero_mask].tolist()
    print(f"⚠️ Có {len(zero_states)} hàng =0: {zero_states}")

    k = P.shape[1]

    for st in zero_states:
        if policy == "parent" and fallback_P is not None:
            P.loc[st] = fallback_P.loc[st].values
        elif policy == "uniform":
            P.loc[st] = 1.0 / k
        else:  # identity
            P.loc[st] = 0.0
            if st in P.columns:
                P.loc[st, st] = 1.0
    return P


def _enforce_absorbing(P: pd.DataFrame, absorbing):
    for st in absorbing:
        if st in P.index and st in P.columns:
            P.loc[st] = 0.0
            P.loc[st, st] = 1.0
    return P


# ============================================================
# 1️⃣ make_pairs – chuẩn hoá và tạo cặp MOB → MOB+1
# ============================================================

def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng cặp (t, t+1) CHUẨN HOÁ từ toàn dataset.
    Không lọc theo product hay score trước.
    Trả về DataFrame gồm:
        loan, mob_t, mob_t1, state_t, state_t1,
        ead_t, product_t, score_t
    """
    loan = CFG["loan"]
    mob = CFG["mob"]
    state = CFG["state"]
    ead = CFG.get("ead")

    # Tên cột segment
    product_col = "PRODUCT_TYPE" if "PRODUCT_TYPE" in df.columns else None
    score_col   = "RISK_SCORE"   if "RISK_SCORE" in df.columns   else None

    # --- chuẩn hoá ---
    work = df.copy()
    work[mob] = _safe_int_series(work[mob])
    work = work.dropna(subset=[mob, state])
    work = work.sort_values([loan, mob])

    # shift next
    mob_next   = work.groupby(loan)[mob].transform(lambda x: x.shift(-1))
    state_t    = work.groupby(loan)[state].transform(lambda x: x.shift(0))
    state_t1   = work.groupby(loan)[state].transform(lambda x: x.shift(-1))

    # weight
    if ead and ead in work.columns:
        ead_t = pd.to_numeric(work[ead], errors="coerce").fillna(0.0)
    else:
        ead_t = pd.Series(1.0, index=work.index)

    # chỉ lấy cặp liền kề
    pairs = work[(mob_next - work[mob]) == 1].copy()
    if pairs.empty:
        print("⚠️ make_pairs(): Không tìm thấy cặp hợp lệ trên toàn dataset.")
        return pd.DataFrame(columns=[
            loan, "mob_t", "mob_t1", "state_t", "state_t1",
            "ead_t", "product_t", "score_t"
        ])

    pairs["mob_t"]  = work.loc[pairs.index, mob].astype(int)
    pairs["mob_t1"] = mob_next.loc[pairs.index].astype(int)
    pairs["state_t"]  = state_t.loc[pairs.index]
    pairs["state_t1"] = state_t1.loc[pairs.index]
    pairs["ead_t"]    = ead_t.loc[pairs.index].astype(float)

    pairs["product_t"] = work[product_col].loc[pairs.index] if product_col else "ALL"
    pairs["score_t"]   = work[score_col].loc[pairs.index]   if score_col   else "ALL"

    # state-check
    _warn_unknown_states(set(pairs["state_t"]).union(set(pairs["state_t1"])))

    print(f"🔍 make_pairs(): created {len(pairs):,} pairs from {len(work):,} rows.")
    return pairs[
        [loan, "mob_t", "mob_t1", "state_t", "state_t1",
         "ead_t", "product_t", "score_t"]
    ]


# ============================================================
# 2️⃣ compute_transition_from_pairs – tính ma trận từ 1 lát cắt pairs
# ============================================================

def compute_transition_from_pairs(
    pair_slice: pd.DataFrame,
    value_col: str = "ead_t",
    parent_P: pd.DataFrame | None = None,
    zero_row_policy: str = "parent",
    alpha_smooth: float | None = None,
) -> pd.DataFrame:

    if pair_slice.empty:
        return pd.DataFrame(0.0, index=BUCKETS_CANON, columns=BUCKETS_CANON)

    weights = pd.to_numeric(pair_slice[value_col], errors="coerce").fillna(0.0)

    mat_counts = (
        pd.crosstab(pair_slice["state_t"],
                    pair_slice["state_t1"],
                    values=weights,
                    aggfunc="sum",
                    dropna=False)
        .reindex(index=BUCKETS_CANON, columns=BUCKETS_CANON, fill_value=0.0)
    )

    P = _normalize_rows(mat_counts)

    P = _backfill_zero_rows(P, mat_counts, fallback_P=parent_P, policy=zero_row_policy)
    P = _enforce_absorbing(P, ABSORBING_BASE)

    bad = ~np.isclose(P.sum(axis=1).values, 1.0, atol=1e-6)
    if bad.any():
        bad_states = [P.index[i] for i, b in enumerate(bad) if b]
        print(f"⚠️ Hàng không cộng về 1: {bad_states} → normalize lại.")
        P = _normalize_rows(P)

    # selective smoothing
    _alpha = ALPHA_SMOOTH if alpha_smooth is None else alpha_smooth
    if _alpha > 0:
        zero_rows = P.index[P.sum(axis=1) == 0].tolist()
        if zero_rows:
            k = len(P.columns)
            for st in zero_rows:
                P.loc[st] = 1.0 / k
            P = _normalize_rows(P)

    return P


# ============================================================
# 3️⃣ compute_transition_by_mob – build cấu trúc 3 tầng
# ============================================================

def compute_transition_by_mob(df: pd.DataFrame):
    """
    Xây ma trận chuyển trạng thái 3 tầng:
        matrices[product][mob][score] = {
            "P": <DataFrame ma trận>,
            "is_fallback": True/False,
            "reason": str
        }
    Đồng thời trả thêm:
        parent_fallback[(product, score)] = ma trận parent (product×score)
    """

    pairs = make_pairs(df)
    if pairs.empty:
        print("⚠️ compute_transition_by_mob(): không có cặp → return empty.")
        return {}, {}

    # 3 tầng: product → mob → score
    matrices = defaultdict(lambda: defaultdict(dict))

    # Tầng parent: product × score
    parent_fallback: dict[tuple, pd.DataFrame] = {}

    # 1) Build ma trận parent cho từng (product, score)
    for (prod, score), grp in pairs.groupby(["product_t", "score_t"]):
        P_parent = compute_transition_from_pairs(
            grp,
            value_col="ead_t",
            parent_P=None,
            alpha_smooth=0.0
        )
        parent_fallback[(str(prod), str(score))] = P_parent
        print(f"⚙️ Built parent fallback for (product={prod}, score={score})")

    # 2) Build ma trận theo MOB, dùng parent khi dữ liệu yếu
    for (prod, score), grp in pairs.groupby(["product_t", "score_t"]):
        prod_str = str(prod)
        score_str = str(score)
        parent_P = parent_fallback[(prod_str, score_str)]

        for mob in sorted(grp["mob_t"].unique()):
            mob_grp = grp[grp["mob_t"] == mob]
            total_ead = mob_grp["ead_t"].sum()
            n_obs = len(mob_grp)

            # Nếu dữ liệu quá ít → dùng fallback parent
            if n_obs < MIN_OBS or total_ead < MIN_EAD:
                reason = f"fallback: insufficient data (obs={n_obs}, ead={total_ead:,.0f})"
                print(f"⚠️ Fallback MOB={mob} for (product={prod}, score={score}) → {reason}")
                matrices[prod_str][int(mob)][score_str] = {
                    "P": parent_P,
                    "is_fallback": True,
                    "reason": reason,
                }
                continue

            # Ma trận thực nghiệm at MOB-level
            P_child = compute_transition_from_pairs(
                mob_grp,
                value_col="ead_t",
                parent_P=parent_P,
                zero_row_policy="parent",
                alpha_smooth=0.0,
            )

            # Nếu mọi hàng đều bằng 0 (không có cặp) → fallback
            if (P_child.sum(axis=1) == 0).all():
                reason = "fallback: no valid pairs at MOB-level"
                print(f"⚠️ MOB={mob} (product={prod}, score={score}) → {reason}")
                matrices[prod_str][int(mob)][score_str] = {
                    "P": parent_P,
                    "is_fallback": True,
                    "reason": reason,
                }
            else:
                matrices[prod_str][int(mob)][score_str] = {
                    "P": P_child,
                    "is_fallback": False,
                    "reason": "ok",
                }

    # Đếm tổng số ma trận
    total = sum(
        1
        for prod_dict in matrices.values()
        for mob_dict in prod_dict.values()
        for _ in mob_dict.values()
    )
    print(f"✅ Generated {total} MOB-level matrices (including fallback).")

    return matrices, parent_fallback


