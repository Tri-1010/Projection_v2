import config
from src.models.markov.transition import TransitionModel


def test_fallback_to_nearest_previous_mob(synthetic_df, schema):
    # Limit data to early mobs to force fallback for later horizon
    truncated = synthetic_df[synthetic_df[schema.mob_col] <= 2].copy()
    model = TransitionModel(
        truncated,
        schema=schema,
        state_order=config.STATE_ORDER,
        min_obs=1,
        min_ead=1,
    )
    matrix, meta = model.select_matrix(("A", "P1"), mob=5)
    assert meta["matrix_source"] == "segment_nearest_previous_mob"
    assert meta["mob_used"] == 1
    assert matrix.shape[0] == len(config.STATE_ORDER)


def test_fallback_to_global_all_when_segment_missing(synthetic_df, schema):
    model = TransitionModel(
        synthetic_df,
        schema=schema,
        state_order=config.STATE_ORDER,
        min_obs=1,
        min_ead=1,
    )
    matrix, meta = model.select_matrix(("UNKNOWN", "UNKNOWN"), mob=1)
    assert meta["matrix_source"] in {"global_mob", "global_all", "identity"}
    assert matrix.shape[0] == len(config.STATE_ORDER)
