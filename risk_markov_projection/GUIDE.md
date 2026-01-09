# Risk Markov Projection – Detailed Guide

This guide shows how to configure, run, and validate the time-inhomogeneous Markov projection pipeline. All parameters live in `config.py`; avoid hard-coding elsewhere.

## Data requirements
- Mandatory columns (see `config.SCHEMA`): `AGREEMENT_ID`, `MOB`, `STATE_MODEL`, `PRINCIPLE_OUTSTANDING`, `CUTOFF_DATE`, `RISK_SCORE`, `PRODUCT_TYPE`.
- States must be in `config.STATE_ORDER` (current defaults include `DPD0`, `DPD1+`, `DPD30+`, `DPD60+`, `DPD90+`, `DPD120+`, `DPD180+`, `WRITEOFF`, `PREPAY`, `CLOSED`, `CURRENT`).
- Absorbing states: `WRITEOFF`, `PREPAY`, `CLOSED` (configurable via `ABSORBING_STATES` and `CLOSED_ABSORBING`).
- Segmentation: `RISK_SCORE`, `PRODUCT_TYPE` (configurable via `SCHEMA.segment_cols`).

## Configuration (config.py)
- Data source: set `DATA_SOURCE = "parquet"` or `"oracle"`.
- Parquet path: `PARQUET_PATH` (directory containing one or more parquet files).
- Oracle: define `ORACLE_CONFIG["sql"]`, optional `params`, `sql_dir`; environment must provide `ORA_*` credentials.
- Thresholds: `MIN_OBS`, `MIN_EAD`, `MAX_MOB` (horizon).
- Transition: `TRANSITION_WEIGHT_MODE` (`"ead"` default), smoothing strength/parents in `SMOOTHING`, fallback order in `FALLBACK_ORDER`.
- Buckets: `BUCKETS_30P`, `BUCKETS_60P`, `BUCKETS_90P`.
- Calibration: toggle via `CALIBRATION["enabled"]`, bounds via `lower_bound`/`upper_bound`.
- Outputs: set `OUTPUT["dir"]`, `csv_name`, `parquet_name`.

## Running the pipeline (CLI)
From `risk_markov_projection/`:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m src.pipelines.run_projection \
  --asof-date 2025-10-01 \
  --target-mob 24 \
  --source parquet \
  --parquet-path "C:/path/to/parquet_dir"
```
Outputs: CSV and Parquet in `OUTPUT["dir"]` with `EAD_*`, `DIST_*`, `DEL_*_ON_EAD0`, and audit columns (`matrix_source`, `mob_used`, `n_obs_used`, `ead_sum_used`, `calibration_factor`).

## Notebook workflow
- Open `notebooks/interactive_projection.ipynb`.
- Ensure the first cell prints the detected project root; adjust `DATA_SOURCE`, `PARQUET_PATH`, and thresholds in the config cell.
- Run the cells in order to load, validate, project, and visualize delinquency trajectories. Calibration is controlled by `config.CALIBRATION["enabled"]`.

## Testing
- Run all tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`
- Key tests: validators, transition fallback, projection/distribution/indicators, calibration, and end-to-end integration.

## Calibration notes
- Method: scalar-by-MOB on `DEL_30P_ON_EAD0`; factors are clamped to `CALIBRATION["lower_bound"]`/`upper_bound` and applied to delinquent buckets with renormalization.
- Disable by setting `CALIBRATION["enabled"] = False` if you want raw projected distributions.

## Troubleshooting
- `ValidationError: States not in state_order`: add the missing states to `STATE_ORDER` (and buckets/absorbing lists if needed) or map them upstream.
- `MOB out of range`: increase `MAX_MOB` in config or filter the data.
- Empty outputs: check `MIN_OBS`/`MIN_EAD` thresholds and fallback order.
