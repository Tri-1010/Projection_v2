import numpy as np

import config
from src.models.markov.projector import MarkovProjector
from src.models.markov.transition import TransitionModel


def test_projection_preserves_total_ead(synthetic_df, schema):
    model = TransitionModel(
        synthetic_df,
        schema=schema,
        state_order=config.STATE_ORDER,
        min_obs=1,
        min_ead=1,
    )
    projector = MarkovProjector(model, schema=schema, state_order=config.STATE_ORDER, max_mob=4)
    result = projector.project(synthetic_df)
    for _, group in result.groupby(schema.segment_cols):
        ead0 = group["EAD0"].iloc[0]
        totals = group[[f"EAD_{s}" for s in config.STATE_ORDER]].sum(axis=1).values
        assert np.allclose(totals, ead0)
