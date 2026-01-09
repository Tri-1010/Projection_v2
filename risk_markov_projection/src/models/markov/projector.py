from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
from src.data.schema import DataSchema, default_schema
from src.models.markov.transition import TransitionModel
from src.utils.metrics import delinquency_indicator


def forecast_to_distribution(forecast_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    first_mob = min(forecast_dict.keys())
    total_ead0 = forecast_dict[first_mob].sum()
    if total_ead0 == 0:
        return {mob: np.zeros_like(vec) for mob, vec in forecast_dict.items()}
    return {mob: vec / total_ead0 for mob, vec in forecast_dict.items()}


class MarkovProjector:
    def __init__(
        self,
        transition_model: TransitionModel,
        schema: DataSchema | None = None,
        state_order: List[str] | None = None,
        max_mob: int | None = None,
    ) -> None:
        self.transition_model = transition_model
        self.schema = schema or default_schema()
        self.state_order = state_order or config.STATE_ORDER
        self.max_mob = max_mob if max_mob is not None else config.MAX_MOB
        self.segment_cols = list(self.schema.segment_cols)

    def _initial_ead_vectors(self, df: pd.DataFrame) -> Dict[Tuple, np.ndarray]:
        mob0 = df[df[self.schema.mob_col] == 0]
        if mob0.empty:
            return {}
        grouped = mob0.groupby(self.segment_cols + [self.schema.state_col])[self.schema.ead_col].sum().reset_index()
        vectors: Dict[Tuple, np.ndarray] = {}
        for key, group in grouped.groupby(self.segment_cols):
            key_tuple = key if isinstance(key, tuple) else (key,)
            vec = np.zeros(len(self.state_order))
            for _, row in group.iterrows():
                if row[self.schema.state_col] in self.state_order:
                    idx = self.state_order.index(row[self.schema.state_col])
                    vec[idx] = row[self.schema.ead_col]
            vectors[key_tuple] = vec
        return vectors

    def _project_segment(self, segment_key: Tuple, initial_vec: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        forecast: Dict[int, np.ndarray] = {0: initial_vec}
        audits: Dict[int, Dict] = {
            0: {"matrix_source": "initial", "mob_used": None, "n_obs_used": 0, "ead_sum_used": initial_vec.sum()}
        }
        for mob in range(self.max_mob):
            matrix, meta = self.transition_model.select_matrix(segment_key, mob)
            next_vec = forecast[mob] @ matrix
            forecast[mob + 1] = next_vec
            audits[mob + 1] = meta
        return forecast, audits

    def _indicator_values(self, dist_vec: np.ndarray) -> Dict[str, float]:
        del_30p = delinquency_indicator(dist_vec, self.state_order, config.BUCKETS_30P)
        del_60p = delinquency_indicator(dist_vec, self.state_order, config.BUCKETS_60P)
        del_90p = delinquency_indicator(dist_vec, self.state_order, config.BUCKETS_90P)
        return {
            "DEL_30P_ON_EAD0": del_30p,
            "DEL_60P_ON_EAD0": del_60p,
            "DEL_90P_ON_EAD0": del_90p,
        }

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_vectors = self._initial_ead_vectors(df)
        records = []

        for segment_key, init_vec in initial_vectors.items():
            forecast, audits = self._project_segment(segment_key, init_vec)
            dist_map = forecast_to_distribution(forecast)
            ead0 = forecast[min(forecast.keys())].sum()
            for mob, vec in forecast.items():
                dist_vec = dist_map[mob]
                record = {self.schema.mob_col: mob, "EAD0": ead0}
                for col_name, value in zip(self.segment_cols, segment_key):
                    record[col_name] = value
                for state, value in zip(self.state_order, vec):
                    record[f"EAD_{state}"] = value
                    record[f"DIST_{state}"] = dist_vec[self.state_order.index(state)]
                record.update(self._indicator_values(dist_vec))
                record.update(audits.get(mob, {}))
                records.append(record)

        result = pd.DataFrame(records)
        if not result.empty:
            result = result.sort_values(self.segment_cols + [self.schema.mob_col]).reset_index(drop=True)
        return result
