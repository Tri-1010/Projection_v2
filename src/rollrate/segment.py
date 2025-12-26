import pandas as pd
from pathlib import Path
from src.config import CFG, OUT_ROOT, MIN_OBS  # dùng SEGMENT_MAP ở dưới nếu bạn có
from src.rollrate.transition import compute_transition

def _matrix_to_long(P: pd.DataFrame, type_label: str, segment_label: str) -> pd.DataFrame:
    """Chuyển ma trận vuông sang long-form để lưu kèm TYPE/SEGMENT."""
    out = P.copy()
    out["FROM_STATE"] = out.index
    out = out.melt(id_vars="FROM_STATE", var_name="TO_STATE", value_name="PROB")
    out["TYPE"] = type_label
    out["SEGMENT"] = segment_label
    # Sắp xếp đẹp
    return out[["SEGMENT", "TYPE", "FROM_STATE", "TO_STATE", "PROB"]]

def generate_all_transitions(
    df: pd.DataFrame,
    segment_map: dict[str, list] | None = None,
    by_product_file: bool = True,
) -> pd.DataFrame:
    """
    Tính và lưu toàn bộ ma trận transition theo các segment đã cấu hình.
    - CONTRACT (đếm) và AMOUNT (theo EAD)
    - Chuẩn long-form để ghi Excel

    Args:
        df: dữ liệu panel có cột loan/mob/state/ead theo CFG
        segment_map: dict như {"PRODUCT_TYPE": ["A","B"], "RISK_SCORE":[1,2,3]}.
                     Nếu None, mặc định duyệt theo cột PRODUCT_TYPE đang có trong data.
        by_product_file: True → ghi 1 file/PRODUCT_TYPE, False → ghi 1 file tổng.

    Returns:
        DataFrame long-form gộp tất cả (SEGMENT, TYPE, FROM_STATE, TO_STATE, PROB)
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    # --- xác định segment_map mặc định nếu không truyền vào
    if segment_map is None:
        if "PRODUCT_TYPE" in df.columns:
            segment_map = {"PRODUCT_TYPE": sorted(df["PRODUCT_TYPE"].dropna().unique())}
        else:
            segment_map = {"_ALL_": ["ALL"]}

    # --- vòng lặp theo từng cột segment và từng giá trị
    for seg_col, seg_vals in segment_map.items():
        for seg_val in seg_vals:
            if seg_col == "_ALL_":
                subset = df.copy()
                seg_label = "ALL"
                product_for_file = "ALL"
            else:
                subset = df[df[seg_col] == seg_val].copy()
                seg_label = f"{seg_col}={seg_val}"
                product_for_file = str(seg_val) if seg_col == "PRODUCT_TYPE" else "ALL"

            if len(subset) < MIN_OBS:
                print(f"⚠️ Bỏ qua {seg_label} (obs={len(subset)} < MIN_OBS={MIN_OBS})")
                continue

            # --- CONTRACT (đếm)
            subset["__COUNT__"] = 1.0
            P_count = compute_transition(subset, value_col="__COUNT__")
            long_count = _matrix_to_long(P_count, "CONTRACT", seg_label)

            # --- AMOUNT (EAD)
            if CFG["ead"] in subset.columns:
                P_amt = compute_transition(subset, value_col=CFG["ead"])
            else:
                print(f"⚠️ Không thấy cột EAD ({CFG['ead']}) trong {seg_label}, dùng COUNT thay thế.")
                P_amt = compute_transition(subset, value_col="__COUNT__")
            long_amt = _matrix_to_long(P_amt, "AMOUNT", seg_label)

            # --- gộp hai loại
            res = pd.concat([long_count, long_amt], ignore_index=True)
            results.append(res)

            # --- xuất Excel
            if by_product_file:
                out_file = OUT_ROOT / f"rollrate_{product_for_file}.xlsx"
                sheet = seg_label[:31]  # excel limit
            else:
                out_file = OUT_ROOT / "rollrate_all_segments.xlsx"
                sheet = seg_label[:31]

            with pd.ExcelWriter(out_file, engine="openpyxl", mode="a" if out_file.exists() else "w") as writer:
                # mỗi sheet ghi long-form (dễ đọc + pivot lại khi cần)
                res.to_excel(writer, sheet_name=sheet, index=False)

            print(f"✅ Saved matrices (CONTRACT/AMOUNT) for {seg_label} → {out_file.name}:{sheet}")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
