# RR_Model_v3 — Roll Rate / Markov Chain (Full Package)

**Ngôn ngữ / Languages:** [Tiếng Việt](#tiếng-việt) | [English](#english)

---

## Tiếng Việt

### 🎯 Mục tiêu
Bộ công cụ mô hình **Roll Rate / Markov Chain** để:
- Tính ma trận chuyển trạng thái DPD (theo số hợp đồng & số dư)
- Dự báo phân phối rủi ro 12 tháng tới cho từng *subproduct*
- Backtest: kiểm định ổn định ma trận & roll-forward validation
- Xuất báo cáo Excel theo *subproduct* và sheet Summary cho toàn danh mục

### 🗂️ Cấu trúc
```
RR_Model_v3/
├── README.md
├── src/
│   ├── db.py
│   ├── config.py
│   ├── data_loader.py
│   └── rollrate/
│        ├── transition.py
│        ├── segment.py
│        ├── forecast.py
│        ├── backtest.py
│        └── model.py
├── data/
│   └── parquet/
│        └── (đặt file parquet của bạn tại đây)
└── notebooks/
    └── RR_Model_Demo.ipynb
```

### ⚙️ Cấu hình (`src/config.py`)
```python
OUT_ROOT = Path("./outputs")
DATA_SOURCE = "parquet"
PARQUET_DIR = Path("./data/parquet")
PARQUET_FILE = "rollrate_base.parquet"
```

### 🚀 Cách chạy nhanh
1. Cài đặt thư viện:
   ```bash
   pip install pandas numpy matplotlib seaborn openpyxl
   ```
2. Đặt file parquet của bạn vào `./data/parquet/rollrate_base.parquet`
3. Mở notebook:
   ```bash
   jupyter notebook notebooks/RR_Model_Demo.ipynb
   ```
4. Chạy từng cell → outputs sẽ được tạo tại `./outputs/`

### 🧩 Thành phần chính
- `transition.py`: tính ma trận Markov (contract/amount)
- `segment.py`: loop segment/subproduct, lưu Excel
- `forecast.py`: dự báo 12 tháng + xuất report & Summary
- `backtest.py`: stability & roll-forward validation
- `model.py`: orchestrator end-to-end
- `data_loader.py`: chọn Parquet/Oracle (mặc định Parquet)

---

## English

### 🎯 Purpose
A **Roll Rate / Markov Chain** toolkit to:
- Estimate DPD transition matrices (by contract & amount)
- Forecast 12‑month risk distribution by subproduct
- Backtest: matrix stability & roll‑forward validation
- Export Excel reports per subproduct + portfolio Summary sheet

### 🗂️ Structure
(see the same tree above)

### ⚙️ Configuration (`src/config.py`)
```python
OUT_ROOT = Path("./outputs")
DATA_SOURCE = "parquet"
PARQUET_DIR = Path("./data/parquet")
PARQUET_FILE = "rollrate_base.parquet"
```

### 🚀 Quickstart
1. Install deps:
   ```bash
   pip install pandas numpy matplotlib seaborn openpyxl
   ```
2. Place your parquet file at `./data/parquet/rollrate_base.parquet`
3. Open the notebook:
   ```bash
   jupyter notebook notebooks/RR_Model_Demo.ipynb
   ```
4. Run cells → outputs land in `./outputs/`

### 🧩 Core modules
- `transition.py`: Markov matrix by contract/amount
- `segment.py`: iterate segments/subproducts, save Excel
- `forecast.py`: 12‑month forecast + summary export
- `backtest.py`: stability & roll‑forward validation
- `model.py`: end‑to‑end pipeline
- `data_loader.py`: Parquet/Oracle switch (default Parquet)
