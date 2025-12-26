from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import oracledb

# TỐI ƯU MẶC ĐỊNH CHO oracledb
# ----------------------------------------------------------------
# Tăng số dòng fetch mỗi lần (mặc định ~100)
oracledb.defaults.fetch_array_size = 50000
oracledb.defaults.prefetch_rows = 50000

# Nếu mạng hoặc DB chậm → tăng lên 100k
# oracledb.defaults.fetch_array_size = 100000
# oracledb.defaults.prefetch_rows = 100000
# ----------------------------------------------------------------


# === Oracle config (only needed if using DB) ===
ORA_HOST    = os.getenv("ORA_HOST", "prd-datamart-01.mafc.vn")
ORA_PORT    = int(os.getenv("ORA_PORT", "1521"))
ORA_SERVICE = os.getenv("ORA_SERVICE", "datamart")
ORA_USER    = os.getenv("ORA_USER", "RISK")
ORA_PASS    = os.getenv("ORA_PASS", "CongDuc$0luong")


def _connect():
    """
    Tạo kết nối Oracle đã tối ưu.
    Chạy ở chế độ Thin (nhanh, không cần Instant Client).
    """
    dsn = oracledb.makedsn(ORA_HOST, ORA_PORT, service_name=ORA_SERVICE)

    conn = oracledb.connect(
        user=ORA_USER,
        password=ORA_PASS,
        dsn=dsn,
        # tăng tốc network round-trip
        stmtcachesize=50,        # tăng cache câu lệnh
        edition=None,
    )

    return conn


def _resolve_sql_text(sql: str, sql_dir: str | None) -> tuple[str, Path | None]:
    p = Path(sql)
    if p.suffix.lower() == ".sql":
        candidates = [p]
        if sql_dir:
            candidates.append(Path(sql_dir) / p.name)
        candidates.append(Path("src/sql") / p.name)
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError(f"Không tìm thấy file SQL: {p}")
        text = found.read_text(encoding="utf-8-sig")
        return text, found
    return sql, None


def _clean_sql(text: str) -> str:
    s = text.strip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def load_df(sql: str, params: dict | None = None, sql_dir: str | None = None) -> pd.DataFrame:
    raw_sql, src_file = _resolve_sql_text(sql, sql_dir)
    final_sql = _clean_sql(raw_sql)
    bind_params = params or {}

    print("=== SQL DEBUG ===")
    if src_file:
        print("File:", src_file)
    print("First 200 chars:\n", final_sql[:200].replace("\n", " ") + ("..." if len(final_sql) > 200 else ""))
    print("Params:", {k: (str(v)[:40] + ("..." if len(str(v)) > 40 else "")) for k, v in bind_params.items()})
    print("=================")

    conn = _connect()

    try:
        # pandas + oracledb đã được hưởng tối ưu fetch_array_size
        df = pd.read_sql_query(
            final_sql,
            conn,
            params=bind_params,
            chunksize=None     # đọc nguyên khối → nhanh nhất
        )
    finally:
        conn.close()

    return df
