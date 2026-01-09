import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

import pandas as pd
import pytest

from src.data.schema import default_schema


@pytest.fixture
def schema():
    return default_schema()


def make_synthetic_dataset(num_loans_per_segment: int = 5, max_mob: int = 6) -> pd.DataFrame:
    records = []
    date_base = pd.Timestamp("2024-01-31")
    segments = [("A", "P1"), ("A", "P2"), ("B", "P1"), ("B", "P2")]
    state_cycle = ["CURRENT", "CURRENT", "DPD30+", "DPD60+", "DPD90+", "WRITEOFF", "WRITEOFF"]

    for risk_score, product in segments:
        for i in range(num_loans_per_segment):
            agreement_id = f"{risk_score}{product}{i}"
            ead0 = 1000 + 50 * i
            for mob in range(max_mob + 1):
                state = state_cycle[min(mob, len(state_cycle) - 1)]
                if mob == max_mob and i % 2 == 1:
                    state = "CLOSED"
                cutoff_date = date_base + pd.DateOffset(months=mob)
                ead_value = max(ead0 - mob * 20, 50)
                records.append(
                    {
                        "AGREEMENT_ID": agreement_id,
                        "MOB": mob,
                        "STATE_MODEL": state,
                        "PRINCIPLE_OUTSTANDING": float(ead_value),
                        "CUTOFF_DATE": cutoff_date,
                        "RISK_SCORE": risk_score,
                        "PRODUCT_TYPE": product,
                    }
                )
    return pd.DataFrame(records)


@pytest.fixture
def synthetic_df():
    return make_synthetic_dataset()
