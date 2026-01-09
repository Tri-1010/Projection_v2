import numpy as np

import config
from src.models.markov.transition import TransitionModel


def test_transition_rows_sum_to_one(synthetic_df, schema):
    model = TransitionModel(
        synthetic_df,
        schema=schema,
        state_order=config.STATE_ORDER,
        min_obs=1,
        min_ead=1,
    )
    matrix, meta = model.select_matrix(("A", "P1"), mob=0)
    assert matrix.shape[0] == matrix.shape[1] == len(config.STATE_ORDER)
    assert np.allclose(matrix.sum(axis=1), np.ones(len(config.STATE_ORDER)))
    assert meta["matrix_source"] in config.FALLBACK_ORDER + ["identity"]


def test_absorbing_states_enforced(synthetic_df, schema):
    model = TransitionModel(
        synthetic_df,
        schema=schema,
        state_order=config.STATE_ORDER,
        min_obs=1,
        min_ead=1,
        absorbing_states=["WRITEOFF", "CLOSED"],
        closed_absorbing=True,
    )
    matrix, _ = model.select_matrix(("A", "P1"), mob=3)
    idx_writeoff = config.STATE_ORDER.index("WRITEOFF")
    idx_closed = config.STATE_ORDER.index("CLOSED")
    assert np.allclose(matrix[idx_writeoff], np.eye(len(config.STATE_ORDER))[idx_writeoff])
    assert np.allclose(matrix[idx_closed], np.eye(len(config.STATE_ORDER))[idx_closed])
