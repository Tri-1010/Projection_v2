import numpy as np

import config
from src.models.markov.projector import MarkovProjector
from src.models.markov.transition import TransitionModel


def test_distribution_over_ead0_sums_to_one(synthetic_df, schema):
    model = TransitionModel(synthetic_df, schema=schema, state_order=config.STATE_ORDER, min_obs=1, min_ead=1)
    projector = MarkovProjector(model, schema=schema, state_order=config.STATE_ORDER, max_mob=3)
    result = projector.project(synthetic_df)
    dist_cols = [f"DIST_{s}" for s in config.STATE_ORDER]
    sums = result[dist_cols].sum(axis=1).values
    assert np.allclose(sums, np.ones_like(sums))
