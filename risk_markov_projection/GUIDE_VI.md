# Hướng dẫn Risk Markov Projection (Tiếng Việt)

Tài liệu này hướng dẫn cấu hình, chạy, kiểm thử và xuất báo cáo cho pipeline Markov. Mọi tham số tập trung trong `config.py`.

## Yêu cầu dữ liệu
- Cột bắt buộc (theo `config.SCHEMA`): `AGREEMENT_ID`, `MOB`, `STATE_MODEL`, `PRINCIPLE_OUTSTANDING`, `CUTOFF_DATE`, `RISK_SCORE`, `PRODUCT_TYPE`.
- State phải nằm trong `STATE_ORDER` (mặc định: `DPD0`, `CURRENT`, `DPD1+`, `DPD30+`, `DPD60+`, `DPD90+`, `DPD120+`, `DPD180+`, `WRITEOFF`, `PREPAY`, `CLOSED`).
- Absorbing mặc định: `WRITEOFF`, `PREPAY`, `CLOSED` (điều chỉnh qua `ABSORBING_STATES`, `CLOSED_ABSORBING`).

## Cấu hình (config.py)
- Chọn nguồn dữ liệu: `DATA_SOURCE = "parquet"` hoặc `"oracle"`.
- Parquet: đặt `PARQUET_PATH` trỏ tới thư mục chứa file parquet.
- Oracle: cấu hình `ORACLE_CONFIG["sql"]`, `params`, `sql_dir` và đảm bảo biến môi trường `ORA_*`.
- Ngưỡng/khác: `MIN_OBS`, `MIN_EAD`, `MAX_MOB`, `TRANSITION_WEIGHT_MODE` (mặc định `"ead"`), `SMOOTHING`, `FALLBACK_ORDER`.
- Buckets chỉ tiêu: `BUCKETS_30P`, `BUCKETS_60P`, `BUCKETS_90P`.
- Calibration: bật/tắt qua `CALIBRATION["enabled"]`, dải clamp qua `lower_bound`/`upper_bound`.
- Output: thư mục và tên file trong `OUTPUT` (`csv_name`, `parquet_name`, `report_name`).

## Chạy pipeline (CLI)
Từ thư mục `risk_markov_projection/`:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m src.pipelines.run_projection \
  --asof-date 2025-10-01 \
  --target-mob 24 \
  --source parquet \
  --parquet-path "C:/duong_dan/parquet_dir"
```
Kết quả: CSV + Parquet trong `OUTPUT["dir"]` với `EAD_*`, `DIST_*`, `DEL_*_ON_EAD0`, audit (`matrix_source`, `mob_used`, `n_obs_used`, `ead_sum_used`, `calibration_factor`). File `indicator_report.csv` chứa actual vs predict `DEL_30P_ON_EAD0` theo MOB.

## Xuất Excel
Sau khi có `projection_df` (kết quả pipeline):
```python
from pathlib import Path
import pandas as pd
from src.utils.export_excel import export_projection_excel

output_dir = Path("outputs")
proj_path = output_dir / "projection.parquet"
projection_df = pd.read_parquet(proj_path)

rep_df = pd.read_csv(output_dir / "indicator_report.csv")
actual_series = rep_df.set_index("MOB")["ACTUAL_DEL30P_ON_EAD0"]

export_projection_excel(
    projection_df,
    output_path=output_dir / "projection_report.xlsx",
    actual_del30p_by_mob=actual_series,
)
```
- Mỗi segment có sheet riêng: DEL%, audit, actual (nếu có), sai số.
- Sheet Summary: fallback rate, mean calibration factor, MAE/WAPE theo DEL_30P.

## Notebook
- Mở `notebooks/interactive_projection.ipynb`.
- Chạy cell thiết lập môi trường (in ra project root).
- Chỉnh `DATA_SOURCE`, `PARQUET_PATH`, ngưỡng trong cell cấu hình.
- Chạy các cell để load/validate, project, visualize, và cell cuối để xuất Excel.
- Dữ liệu lớn: bật cache (`USE_RAW_CACHE`, `USE_PROJ_CACHE`) và đường dẫn cache (`RAW_CACHE_PATH`, `PROJ_CACHE_PATH`) trong notebook để tránh phải chạy lại bước nặng; xóa/đổi cache khi thay đổi nguồn dữ liệu hoặc config.

## Kiểm thử
- Chạy: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`
- Bao phủ: validator, fallback, projection/distribution/indicator, calibration, integration.

## Xử lý sự cố
- “States not in state_order”: bổ sung state vào `STATE_ORDER` (và buckets/absorbing nếu cần) hoặc map trước khi validate.
- “MOB out of range”: tăng `MAX_MOB` hoặc lọc dữ liệu.
- Kết quả trống: kiểm tra `MIN_OBS`/`MIN_EAD` và thứ tự fallback.
