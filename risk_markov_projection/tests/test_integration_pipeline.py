import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.pipelines.run_projection import run


@pytest.mark.integration
def test_pipeline_end_to_end(tmp_path, synthetic_df, schema, monkeypatch):
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = parquet_dir / "data.parquet"
    synthetic_df.to_parquet(parquet_file, index=False)

    # Soften thresholds to keep test data small and redirect outputs to temp
    monkeypatch.setattr(config, "MIN_OBS", 10)
    monkeypatch.setattr(config, "MIN_EAD", 1.0)
    output_dir = tmp_path / "outputs"
    monkeypatch.setitem(config.OUTPUT, "dir", output_dir)
    monkeypatch.setitem(config.OUTPUT, "csv_name", "proj.csv")
    monkeypatch.setitem(config.OUTPUT, "parquet_name", "proj.parquet")

    result = run(target_mob=6, source="parquet", parquet_path=parquet_dir)
    assert not result.empty

    expected_cols = {"EAD0", "matrix_source", "mob_used", "n_obs_used", "ead_sum_used", schema.mob_col}
    expected_cols.update({f"EAD_{s}" for s in config.STATE_ORDER})
    expected_cols.update({f"DIST_{s}" for s in config.STATE_ORDER})
    expected_cols.update({"DEL_30P_ON_EAD0", "DEL_60P_ON_EAD0", "DEL_90P_ON_EAD0"})
    assert expected_cols.issubset(set(result.columns))

    csv_path = output_dir / config.OUTPUT["csv_name"]
    parquet_path_out = output_dir / config.OUTPUT["parquet_name"]
    assert csv_path.exists()
    assert parquet_path_out.exists()

    # Fallback should trigger when thresholds exceed available transitions
    assert (result["matrix_source"] != "segment_mob").any()

    # WRITEOFF absorbing: distribution should not decline across mob for a segment
    for _, group in result.groupby(schema.segment_cols):
        writeoff_series = group[f"DIST_WRITEOFF"].values
        assert np.all(np.diff(writeoff_series) >= -1e-6)

    # Indicators within bounds
    assert ((result["DEL_30P_ON_EAD0"] >= 0) & (result["DEL_30P_ON_EAD0"] <= 1)).all()
