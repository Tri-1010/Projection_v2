import numpy as np

import config
from src.models.markov.projector import MarkovProjector
from src.models.markov.transition import TransitionModel


def test_indicator_matches_bucket_sum(synthetic_df, schema):
    model = TransitionModel(synthetic_df, schema=schema, state_order=config.STATE_ORDER, min_obs=1, min_ead=1)
    projector = MarkovProjector(model, schema=schema, state_order=config.STATE_ORDER, max_mob=3)
    result = projector.project(synthetic_df)
    for _, row in result.iterrows():
        dist_map = {state: row[f"DIST_{state}"] for state in config.STATE_ORDER}
        bucket_sum = sum(dist_map[s] for s in config.BUCKETS_30P if s in dist_map)
        assert np.isclose(row["DEL_30P_ON_EAD0"], bucket_sum)
        assert 0.0 <= row["DEL_30P_ON_EAD0"] <= 1.0
        assert 0.0 <= row["DEL_60P_ON_EAD0"] <= 1.0
        assert 0.0 <= row["DEL_90P_ON_EAD0"] <= 1.0
