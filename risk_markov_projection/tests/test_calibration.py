import numpy as np
import pandas as pd

import config
from src.models.markov.calibration import apply_scalar_to_distributions, fit_scalar_by_mob


def test_calibration_clamp_and_non_negative():
    state_order = config.STATE_ORDER
    # Build distributions aligned to the current state_order length
    base_dists = [
        {
            "DPD0": 0.5,
            "CURRENT": 0.3,
            "DPD1+": 0.1,
            "DPD30+": 0.05,
            "DPD60+": 0.05,
        },
        {
            "DPD0": 0.4,
            "CURRENT": 0.3,
            "DPD1+": 0.1,
            "DPD30+": 0.1,
            "DPD60+": 0.1,
        },
    ]
    dist_rows = []
    for dist_map in base_dists:
        row = []
        for state in state_order:
            row.append(dist_map.get(state, 0.0))
        dist_rows.append(row)
    ead0 = [100.0, 100.0]
    df = pd.DataFrame({"MOB": [0, 1], "EAD0": ead0})
    for idx, state in enumerate(state_order):
        df[f"DIST_{state}"] = [row[idx] for row in dist_rows]
        df[f"EAD_{state}"] = df["EAD0"] * df[f"DIST_{state}"]

    actual = pd.Series({0: 0.2, 1: 0.05})
    predicted = pd.Series({0: 0.1, 1: 0.2})
    factors = fit_scalar_by_mob(actual, predicted, lower=config.CALIBRATION["lower_bound"], upper=config.CALIBRATION["upper_bound"])
    assert np.isclose(factors.loc[0], config.CALIBRATION["upper_bound"])
    adjusted = apply_scalar_to_distributions(df, factors.to_dict(), config.BUCKETS_30P, state_order, mob_col="MOB")
    dist_cols = [f"DIST_{s}" for s in state_order]
    assert (adjusted[dist_cols] >= 0).all().all()
    sums = adjusted[dist_cols].sum(axis=1).values
    assert np.allclose(sums, np.ones_like(sums))
