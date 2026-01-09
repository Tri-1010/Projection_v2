import pandas as pd
import pytest

import config
from src.data.validators import ValidationError, validate_input


def test_validate_input_passes(synthetic_df, schema):
    validate_input(synthetic_df, schema=schema, state_order=config.STATE_ORDER, max_mob=6)


def test_validate_missing_columns(schema):
    df = pd.DataFrame({"AGREEMENT_ID": [1], "MOB": [0]})
    with pytest.raises(ValidationError):
        validate_input(df, schema=schema, state_order=config.STATE_ORDER, max_mob=6)


def test_validate_duplicates(synthetic_df, schema):
    dup = synthetic_df.iloc[[0]].copy()
    df = pd.concat([synthetic_df, dup], ignore_index=True)
    with pytest.raises(ValidationError):
        validate_input(df, schema=schema, state_order=config.STATE_ORDER, max_mob=6)


def test_validate_invalid_state(synthetic_df, schema):
    df = synthetic_df.copy()
    df.loc[0, schema.state_col] = "UNKNOWN_STATE"
    with pytest.raises(ValidationError):
        validate_input(df, schema=schema, state_order=config.STATE_ORDER, max_mob=6)


def test_validate_mob_range(synthetic_df, schema):
    df = synthetic_df.copy()
    df.loc[0, schema.mob_col] = 999
    with pytest.raises(ValidationError):
        validate_input(df, schema=schema, state_order=config.STATE_ORDER, max_mob=6)
