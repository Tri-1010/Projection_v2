import pandas as pd
from src.rollrate.transition import make_pairs
from src.config import CFG, BUCKETS_CANON, MIN_OBS, MIN_EAD

def debug_transition_segment(
    df: pd.DataFrame,
    product: str | None = None,
    score: str | None = None,
    mob: int | None = None,
    show_sample_loans: int = 5,
):
    """
    Debug chi tiết 1 segment:
      - (product, score) parent, hoặc
      - (product, score, mob) MOB-level.
    Giúp trả lời:
      - Có bao nhiêu cặp (state_t→state_t1)?
      - Tổng EAD, số hợp đồng bao nhiêu?
      - Vi phạm ngưỡng MIN_OBS / MIN_EAD hay không?
      - State nào có hàng = 0 trong ma trận (không có cặp)?
    """

    loan_col  = CFG["loan"]
    mob_col   = CFG["mob"]
    state_col = CFG["state"]
    ead_col   = CFG.get("ead")

    print("=" * 80)
    print("🎯 DEBUG TRANSITION SEGMENT")
    print(f"• product = {product}")
    print(f"• score   = {score}")
    print(f"• mob     = {mob}")
    print(f"• MIN_OBS = {MIN_OBS:,}, MIN_EAD = {MIN_EAD:,}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1) Kiểm tra trên df GỐC (chưa ghép cặp)
    # ------------------------------------------------------------------
    raw_mask = pd.Series(True, index=df.index)

    if product is not None and "PRODUCT_TYPE" in df.columns:
        raw_mask &= (df["PRODUCT_TYPE"] == product)

    if score is not None and "RISK_SCORE" in df.columns:
        raw_mask &= (df["RISK_SCORE"] == score)

    if mob is not None and mob_col in df.columns:
        raw_mask &= (pd.to_numeric(df[mob_col], errors="coerce").round(0) == mob)

    df_seg = df[raw_mask].copy()

    print("\n[1] Trên dữ liệu GỐC (chưa ghép cặp):")
    print(f"  - Số dòng (records)  : {len(df_seg):,}")
    if len(df_seg) == 0:
        print("  ⚠️ Segment này không có dòng nào trong df → không thể có cặp transition.")
        return

    n_loans_raw = df_seg[loan_col].nunique()
    print(f"  - Số hợp đồng (loans): {n_loans_raw:,}")

    # Summary state trên snapshot gốc
    state_counts = (
        df_seg[state_col]
        .value_counts()
        .reindex(BUCKETS_CANON, fill_value=0)
    )
    print("\n  - Phân bố STATE trên df gốc:")
    print(state_counts.to_string())

    # Xem thử MOB min/max per loan (để check tính liên tục)
    mob_stats = (
        df_seg.groupby(loan_col)[mob_col]
        .agg(["min", "max", "count"])
        .head(show_sample_loans)
    )
    print(f"\n  - MOB min/max/count trên {show_sample_loans} hợp đồng đầu tiên:")
    print(mob_stats.to_string())

    # ------------------------------------------------------------------
    # 2) Tạo Pairs toàn hệ thống rồi lọc theo segment
    # ------------------------------------------------------------------
    pairs = make_pairs(df)
    if pairs.empty:
        print("\n[2] make_pairs(): không tạo được cặp nào trên toàn dataset → dừng.")
        return

    seg_mask = pd.Series(True, index=pairs.index)

    if product is not None:
        seg_mask &= (pairs["product_t"] == product)
    if score is not None:
        seg_mask &= (pairs["score_t"] == score)
    if mob is not None:
        seg_mask &= (pairs["mob_t"] == mob)

    seg_pairs = pairs[seg_mask].copy()

    print("\n[2] Trên bảng Pairs (đã ghép cặp MOB→MOB+1):")
    print(f"  - Số cặp (rows trong pairs): {len(seg_pairs):,}")
    if len(seg_pairs) == 0:
        print("  ⚠️ Không có cặp MOB→MOB+1 nào trong segment này.")
        print("     → Lý do fallback: 'Không tìm thấy cặp hợp lệ'.")
        return

    n_loans_pairs = seg_pairs[loan_col].nunique()
    total_ead_pairs = seg_pairs["ead_t"].sum()
    print(f"  - Số hợp đồng có cặp  : {n_loans_pairs:,}")
    print(f"  - Tổng EAD trong cặp  : {total_ead_pairs:,.0f}")

    # ------------------------------------------------------------------
    # 3) Kiểm tra NGƯỠNG MIN_OBS / MIN_EAD (logic giống compute_transition_by_mob)
    # ------------------------------------------------------------------
    print("\n[3] Kiểm tra điều kiện ngưỡng MIN_OBS / MIN_EAD:")

    if mob is not None:
        # Đây là logic tương tự khi build MOB-level matrix
        if len(seg_pairs) < MIN_OBS:
            print(f"  ❌ Vi phạm MIN_OBS: n_pairs = {len(seg_pairs):,} < {MIN_OBS:,}")
        else:
            print(f"  ✅ Thỏa MIN_OBS: n_pairs = {len(seg_pairs):,} ≥ {MIN_OBS:,}")

        if total_ead_pairs < MIN_EAD:
            print(f"  ❌ Vi phạm MIN_EAD: total_ead = {total_ead_pairs:,.0f} < {MIN_EAD:,.0f}")
        else:
            print(f"  ✅ Thỏa MIN_EAD: total_ead = {total_ead_pairs:,.0f} ≥ {MIN_EAD:,.0f}")

        if len(seg_pairs) < MIN_OBS or total_ead_pairs < MIN_EAD:
            print("  👉 Kết luận: MOB-level này dùng FALLBACK parent (product, score).")
        else:
            print("  👉 Kết luận: MOB-level này ĐỦ DATA để tính ma trận riêng (không fallback do ngưỡng).")
    else:
        print("  (Không truyền mob → đang debug parent-level (product,score), không áp MIN_OBS/MIN_EAD ở đây.)")

    # ------------------------------------------------------------------
    # 4) Summary theo STATE_T: n_pairs, total_ead, n_loans
    # ------------------------------------------------------------------
    print("\n[4] Phân rã theo STATE_T (trước khi normalize):")

    state_summary = (
        seg_pairs
        .groupby("state_t")
        .agg(
            n_pairs=("state_t", "size"),
            total_ead=("ead_t", "sum"),
            n_loans=(loan_col, "nunique"),
        )
        .reindex(BUCKETS_CANON, fill_value=0)
    )

    print(state_summary.to_string())

    zero_states = state_summary.index[state_summary["n_pairs"] == 0].tolist()
    if zero_states:
        print(f"\n  ⚠️ Các trạng thái có HÀNG = 0 trong ma trận (không có cặp chuyển ra): {zero_states}")
        print("     → Đây chính là những hàng mà hàm _backfill_zero_rows() sẽ xử lý (copy fallback/uniform/identity).")
    else:
        print("\n  ✅ Không có trạng thái nào có hàng = 0 trong ma trận (mọi STATE_T đều có cặp).")

    # ------------------------------------------------------------------
    # 5) Cross-tab EAD state_t → state_t1
    # ------------------------------------------------------------------
    print("\n[5] Cross-tab EAD theo state_t → state_t1 (ma trận thô trước chuẩn hoá):")
    ct = pd.crosstab(
        index=seg_pairs["state_t"],
        columns=seg_pairs["state_t1"],
        values=seg_pairs["ead_t"],
        aggfunc="sum",
        dropna=False
    ).reindex(index=BUCKETS_CANON, columns=BUCKETS_CANON, fill_value=0.0)

    print(ct.to_string(float_format=lambda x: f"{x:,.0f}"))

    print("\n✅ DEBUG DONE.\n")
