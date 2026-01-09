# Risk Markov Projection

Projection of risk indicators using time-inhomogeneous Markov chains segmented by MOB and customer/product attributes. The pipeline builds EAD-weighted transition matrices with fallbacks, projects EAD vectors across states, converts them to percentages over the initial EAD (MOB 0), derives delinquency indicators, and optionally calibrates by MOB.

## Quick start
- Python 3.9+ with `numpy`, `pandas`, `pyarrow`, `pytest` (see `pyproject.toml`).
- Configure parameters in `config.py`; no logic should be hard-coded outside config.
- Prepare synthetic parquet data or a SQL query for Oracle; column mappings are defined in `src/data/schema.py`.
- Run pipeline end-to-end:  
  `python -m src.pipelines.run_projection --asof-date 2024-12-31 --target-mob 24`
- Run tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` (disables external plugins for a clean run)

## Project structure
- `config.py` centralizes runtime parameters, column mappings, segmentation, thresholds, buckets, and calibration settings.
- `src/data`: loading (Oracle/parquet), schema, and validation utilities.
- `src/models/markov`: transition estimation, projection, and calibration modules.
- `src/pipelines`: CLI for full projection run.
- `src/utils`: logging and metric helpers.
- `tests`: unit and integration coverage with synthetic data.

## Outputs
- The pipeline writes projection results to CSV and Parquet with audit columns, EAD per state, distributions over EAD0, delinquency indicators, and fallback metadata.
